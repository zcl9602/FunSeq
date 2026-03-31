import sys
import torch
import polars as pl 
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from pyfastx import Fasta
import os
from tqdm import tqdm
from time import time
import pynvml
import argparse
from captum.attr import IntegratedGradients
from torch.utils.data import Dataset
from contextlib import nullcontext
import torch.nn as nn
import math
import h5py

disable_tqdm = (not sys.stderr.isatty()) or (not sys.stdout.isatty())

class PositionalEncoding(nn.Module):
    # Taken from: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    # "Implement the PE function."
    def __init__(self, d_model, dropout=0.2, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
       
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class cnn_bn(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dropout_prob=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class cnn_mp(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, mp_kernel_size, mp_stride, dropout_prob=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.dropout(x)
        return x

class cnn_amp(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dropout_prob=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        # self.mp = nn.AdaptiveMaxPool1d(1)
        self.mp = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.mp(x)
        x = self.dropout(x)
        x = x.squeeze(-1)
        return x

class GlobalAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.3):
        super().__init__()
        self.mh = nn.MultiheadAttention(input_dim, num_heads=1, dropout=dropout_prob, batch_first=True)
        self.to_out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out, _ = self.mh(x, x, x)
        out = self.to_out(out)
        return out

class minResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, skip_ks):
        super().__init__()
        self.prep = nn.Conv1d(input_dim, output_dim, kernel_size=skip_ks, padding='same')
        self.conv = cnn_bn(input_dim, output_dim, kernel_size=kernel_size)
        self.gatt = GlobalAttention(output_dim, output_dim)
    
    def forward(self, x):
        res = x
        res = self.prep(res)
        x = self.conv(x)
        x = x.transpose(1, 2) 
        x = self.gatt(x)
        x = x.transpose(1, 2)
        x = x + res
        return x 

class resblock(nn.Module):
    def __init__(self, input_dim, mid_dim, out_dim, mid_ks, out_ks):
        super().__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.mid_ks = mid_ks
        self.out_ks = out_ks

        self.res1 = minResBlock(self.input_dim, self.mid_dim, kernel_size=self.mid_ks, skip_ks=1)
        self.res2 = minResBlock(self.mid_dim, self.out_dim, kernel_size=self.out_ks, skip_ks=1)

        self.adap = nn.AdaptiveAvgPool1d(1) 
        self.admp = nn.AdaptiveMaxPool1d(1) 

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x1 = self.adap(x) 
        x2 = self.admp(x) 
        x = torch.cat([x1, x2], dim=1)  
        x = x.squeeze(-1) 
        return x

class CATG(nn.Module):
    def __init__(self,
                seq_len = 1000,
                conv1_outdim = 512, 
                conv1_ks = 11,
                conv2_outdim = 256,
                conv2_ks = 7,
                mp_ks = 23,
                mp_sd = 3,
                res_conv1_dim = 256,
                res_conv1_ks = 21,
                res_skip_ks1 = 1,
                res_conv2_dim = 128,
                res_conv2_ks = 71,
                res_skip_ks2 = 1,
                l1_dim = 4096,
                # l2_dim = 2048,
                tfs_num = 2226
                ):
        super().__init__()
        self.seq_len = seq_len
        self.conv1_outdim = conv1_outdim
        self.conv1_ks = conv1_ks
        self.conv2_outdim = conv2_outdim
        self.conv2_ks = conv2_ks
        self.mp_ks = mp_ks
        self.mp_sd = mp_sd
        self.res_conv1_dim = res_conv1_dim
        self.res_conv1_ks = res_conv1_ks
        self.res_skip_ks1 = res_skip_ks1
        self.res_conv2_dim = res_conv2_dim
        self.res_conv2_ks = res_conv2_ks
        self.res_skip_ks2 = res_skip_ks2
        self.l1_dim = l1_dim
        # self.l2_dim = l2_dim
        self.tfs_num = tfs_num

        self.conv1 = cnn_bn(4, self.conv1_outdim, kernel_size=self.conv1_ks)
        self.position_embedding = PositionalEncoding(self.conv1_outdim)
        # self.conv2_0 = cnn_amp(self.conv1_outdim, self.res_conv2_dim, kernel_size=19)
        self.conv2_0 = cnn_amp(self.conv1_outdim, self.res_conv2_dim, kernel_size=35)
        # self.conv2_0 = cnn_amp(self.conv1_outdim, self.res_conv2_dim, kernel_size=131)
        self.conv2_1 = cnn_mp(self.conv1_outdim, self.conv2_outdim, kernel_size=self.conv2_ks, mp_kernel_size=self.mp_ks, mp_stride=self.mp_sd)
        self.res_block = resblock(self.conv2_outdim, self.res_conv1_dim, self.res_conv2_dim, self.res_conv1_ks, self.res_conv2_ks)

        self.classifier = nn.Sequential(
            nn.Linear(self.res_conv2_dim * 3, self.l1_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(self.l1_dim, self.l2_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(self.l1_dim, self.tfs_num)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x) 
        x = x.permute(0, 2, 1)  
        x = self.position_embedding(x)
        x = x.transpose(1,2)
        x1 = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.res_block(x)
        x = torch.cat([x, x1], dim=1)
        x = self.classifier(x)
        return x

class Bed2Code(Dataset):
    def __init__(self, bed_file, genome_file, logfile='stdout', target_length=1000, train=True):
        print(f"Loading {bed_file}...")
        
        # 1. 读取数据
        self.train = train
        if self.train:
            df = pl.read_csv(bed_file, separator='\t', has_header=False, 
                         new_columns=['chrom', 'start', 'end', 'label'])
        else:
            df = pl.read_csv(bed_file, separator='\t', has_header=False, columns=[0, 1, 2],
                         new_columns=['chrom', 'start', 'end'], schema_overrides={'chrom': pl.Utf8, 'start': pl.Int64, 'end': pl.Int64})
            
        # 2. 添加原始行号 (从 0 开始，方便定位)
        # 这一步极快，几乎不消耗额外时间
        df = df.with_row_index(name="line_idx", offset=0)

        # 3. 计算每一行的实际长度
        # 使用 alias 给这一列命名为 'actual_len'
        df = df.with_columns(
            (pl.col('end') - pl.col('start')).alias('actual_len')
        )
        
        # 4. 分离出 错误行 (Invalid) 和 正确行 (Valid)
        invalid_df = df.filter(pl.col('actual_len') != target_length)
        valid_df = df.filter(pl.col('actual_len') == target_length)
        
        # 5. 详细输出错误信息
        num_invalid = invalid_df.height

        if logfile == 'stdout':
            opener = nullcontext(sys.stdout)
        else:
            opener = open(logfile, 'a')

        with opener as logf:
            if num_invalid > 0:
                logf.write(f"\n[WARNING] Found {num_invalid} samples with length != {target_length}!\n")
                logf.write("-" * 60 + "\n")
                logf.write(f"{'Line #':<8} {'Chrom':<10} {'Start':<10} {'End':<10} {'Actual Len':<10}\n")
                logf.write("-" * 60 + "\n")
                
                # 只打印前 10 个错误，避免刷屏
                # iter_rows() 返回每一行的数据元组
                for row in invalid_df.iter_rows(named=True):
                    logf.write(f"{row['line_idx']:<8} {row['chrom']:<10} {row['start']:<10} {row['end']:<10} {row['actual_len']:<10}\n")
                
                logf.write("-" * 60 + "\n")
                
                # 可选：如果你想把错误行保存到文件以便检查
                # invalid_df.write_csv("error_lines.csv")
            else:
                logf.write(f"[INFO] All {valid_df.height} samples match target length {target_length}.\n")

            # 6. 检查是否还有剩余数据
            if valid_df.height == 0:
                logf.write(f"No valid samples found in {bed_file}! Check your BED file or target_length.\n")
                raise ValueError(f"No valid samples found! Check your BED file or target_length.")

        # 7. 提取正确的数据到内存
        # 此时 valid_df 已经是纯净的数据了
        self.chroms = valid_df['chrom'].to_list()
        self.starts = valid_df['start'].to_numpy()
        self.ends = valid_df['end'].to_numpy()
        if self.train:
            self.labels = valid_df['label'].to_numpy()
        
        self.genome_file = genome_file
        self.genome_fdx_state = None
        self._len = len(self.chroms)

    def __len__(self):
        return self._len
    
    def one_hot_encode(self, input_seq):
        mapping = { 'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 
                    'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        return np.array([mapping.get(base, [0, 0, 0, 0]) for base in input_seq], dtype=np.float32)

    def __getitem__(self, idx):
        if self.genome_fdx_state is None:
            self.genome_fdx_state = Fasta(self.genome_file)
        
        # 直接通过列表/数组索引，极快，无 overhead
        chrom = self.chroms[idx]
        start = self.starts[idx]
        end = self.ends[idx]
        if self.train:
            pred = self.labels[idx]

        # pyfastx 读取
        try:
            # 加上 try-except 以防染色体名称不匹配 (比如 chr1 vs 1)
            seq_obj = self.genome_fdx_state[str(chrom)][start:end]
        except KeyError:
            # 处理找不到染色体的情况，或者打印报错
            raise KeyError(f"Chrom {chrom} not found in fasta file.")

        seq_forward = seq_obj.seq.upper()
        seq_code = self.one_hot_encode(seq_forward)
        
        seq_reverse = seq_obj.antisense.upper()
        seq_reverse_code = self.one_hot_encode(seq_reverse)

        # 【关键修正 1】确保 Numpy 数组内存是连续的 (Contiguous)
        # 即使 one_hot_encode 看起来产生的是新数组，加上这个保险
        seq_code = np.ascontiguousarray(seq_code, dtype=np.float32)
        seq_reverse_code = np.ascontiguousarray(seq_reverse_code, dtype=np.float32)

        # 【关键修正 2】处理 Label
        # 如果直接用 torch.tensor(self.labels[idx])，有时会继承 numpy 的只读属性
        # 用 float() 强转一下，变成纯 Python 数字，最安全
        if self.train:
            pred = float(self.labels[idx])
            return torch.tensor(seq_code), torch.tensor(seq_reverse_code), torch.tensor(pred, dtype=torch.float32)
        else:
            return torch.tensor(seq_code), torch.tensor(seq_reverse_code), torch.tensor(0.0, dtype=torch.float32)  # 占位 label，测试时不使用

def parse_args():
    parser = argparse.ArgumentParser(description='''Predicting binding sites from sequence data
    Note:
    1. The model is trained on the sequence data with the sequence length of 600 bp, so the sequence length of the input data must be 600 bp.
    2. Any sequence containing N will be filtered out.
    ''')
    parser.add_argument('--bed_path', type=str, required=True, help='Path to BED file')
    parser.add_argument('--genome_path', type=str, required=True, help='Path to genome file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file, eg: /public/home/zhuchenglong/project/15.TF_Histone_dat/scripts/tf_distillation_model/distilled.pt')
    parser.add_argument('--output_prefix', type=str, default=f'{sys.argv[0]}', help=f'prefix for output file, default: {sys.argv[0]}')
    parser.add_argument('--output_dir', type=str, default='.', help='Path to output dir, default: current dir')
    parser.add_argument('--target_tf', type=str, default=None, help='which TFBS want to predicted, default: None (predict all TFs)')
    parser.add_argument('--thread', type=int, default=30, help='Number of cpu threads to use, default: 30')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for prediction, default: 512')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use, default: 0')
    parser.add_argument('--test', default=False, action='store_true', help='Test mode (use a small subset of labeled data for quick testing)')
    parser.add_argument('--explain', default=False, action='store_true', help='Enable model explanation (Based on IntegratedGradients method)')
    parser.add_argument('--baseline', type=str, default=None, help='Path to baseline bed file for explanation, should be a bed file with the same format as --bed_path, but can be unlabeled and with different coordinates. If not provided, a zero baseline will be used.')
    return parser.parse_args()

def predictor(dataloader, model, tf_idx, gpu_n, explainer, baseline_data):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_n}')
        device_type = 'cuda'
        handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_n)
        first_state = True
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        first_state = False
        print("Warning: CUDA is not available. Using CPU.")
    output_list = []
    attr_list = []

    if explainer and (tf_idx is not None):
        with torch.autocast(device_type=device_type):
            for batch_plus, batch_minus, _ in tqdm(dataloader, desc=f'Predicting'):
                batch = batch_plus.to(device)
                if len(baseline_data) > len(batch):
                    baseline_data = baseline_data[:len(batch)]
                output_y = model(batch)[:, tf_idx]
                preds = torch.sigmoid(output_y)
                
                attributions = explainer.attribute(batch, baselines=baseline_data, target=tf_idx)

                if first_state:
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
                    first_state = False

                output_list.append(preds.detach().cpu().numpy())
                attr_list.append((attributions * batch).sum(dim=2).detach().cpu().numpy())
    else:
        with torch.inference_mode(), torch.autocast(device_type=device_type):  
            for batch_plus, batch_minus, _ in tqdm(dataloader, desc=f'Predicting'):
                batch_plus = batch_plus.to(device)
                batch_minus = batch_minus.to(device)
                if (tf_idx is not None):
                    output_y_plus = model(batch_plus)[:, tf_idx]
                    output_y_minus = model(batch_minus)[:, tf_idx]
                else:
                    output_y_plus = model(batch_plus)
                    output_y_minus = model(batch_minus)
                output_y = (output_y_plus + output_y_minus) / 2
                preds = torch.sigmoid(output_y)

                if first_state:
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
                    first_state = False
                output_list.append(preds.cpu().numpy())

    if device_type == 'cuda':
        print(f'GPU memory usage: {meminfo.used / 1024**2:.2f} MB')
        torch.cuda.empty_cache()    

    return (
        np.concatenate(output_list, axis=0),
        np.concatenate(attr_list, axis=0) if explainer else None
    )

def main_func(args):
    start_time = time()
    Fasta(args.genome_path)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        pynvml.nvmlInit()
    else:
        device = torch.device('cpu')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    dataset = Bed2Code(args.bed_path, args.genome_path, logfile='stdout', target_length=1000, train=args.test)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.thread, pin_memory=True, prefetch_factor=2)

    batch_size = min(args.batch_size, len(dataset))
    explainer = None
    baseline_data = None

    model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model_path = torch.load(model_path)
    model_pt, tf_names = model_path['pt'], model_path['tf_names']
    tf_idx = None

    if args.target_tf:
        if args.target_tf in tf_names:
            tf_idx = tf_names.index(args.target_tf)
        else:
            raise ValueError(f"Target TF {args.target_tf} not found in model's TF list. Available TFs: {' '.join(tf_names)}")

    if args.explain:
        if args.target_tf is None:
            raise ValueError(f"argument '--target_tf' must be set if you want to explain some one TFBS.\ntarget_tfAvailable TFs: \n{' '.join(tf_names)}")

    if args.explain and (tf_idx is not None):
        if args.baseline:
            baseline_dataset = Bed2Code(args.baseline, args.genome_path, logfile='stdout', target_length=1000, train=args.test)
            baseline_dataloader = DataLoader(baseline_dataset, batch_size=batch_size, shuffle=True, num_workers=args.thread)
            baseline_data = next(iter(baseline_dataloader)).to(device)
        else:
            baseline_data = torch.zeros((1, 1000, 4), device=device)
            print("Warning: Baseline data is not provided. Using zero baseline shape.")
    
    model = CATG()
    model.load_state_dict(model_pt, strict=True)
    model = torch.compile(model)

    model = model.eval().to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.thread, pin_memory=True, prefetch_factor=2)

    if args.explain:
        explainer = IntegratedGradients(model, multiply_by_inputs=True)

    predictions, attributions = predictor(dataloader, model, tf_idx, args.gpu, explainer, baseline_data)

    result_df = h5py.File(f'{args.output_dir}/{args.output_prefix}.h5', 'w')
    result_df.create_dataset('chrom', data=np.array(dataset.chroms, dtype='S'), compression='lzf')
    result_df.create_dataset('start', data=dataset.starts, compression='lzf')
    result_df.create_dataset('end', data=dataset.ends, compression='lzf')

    if tf_idx is not None:
         result_df.create_dataset(tf_names[tf_idx], data=predictions, compression='lzf')
    else:
        # for i, tf_name in enumerate(tf_names):
            # result_df.create_dataset(tf_name, data=predictions[:, i], compression='lzf')
        result_df.create_dataset('tf_names', data=np.array(tf_names, dtype='S'), compression='lzf')
        result_df.create_dataset('all_predictions', data=predictions, compression='lzf')

    if args.explain:
        result_df.create_dataset('attribution', data=attributions, compression='lzf')
    if args.test:
        # y_score = result_df[tf_names[0]].values.astype(np.float32)
        y_score = predictions.astype(np.float32)
        y_true = dataset.labels.astype(np.int32)
        try:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            print(f"\nEvaluation Metrics:\nAUROC: {auroc:.4f}\nAUPRC: {auprc:.4f}")
        except ValueError as e:
            print(f"\nEvaluation failed: {str(e)} (可能因为标签缺少正/负样本)")

    result_df.close()

    torch.cuda.empty_cache()
    end_time = time()
    print(f"Used time: {(end_time - start_time) / 60 :.2f} minutes")
    print(f"Results saved to {args.output_dir}/{args.output_prefix}.h5")

if __name__ == '__main__':
    main_func(parse_args())
