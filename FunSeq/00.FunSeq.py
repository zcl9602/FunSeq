import sys
import torch
from bin.modelPE_C import FunSeq
from bin.prepData import Bed2Code
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import numpy as np
import h5py
from pyfastx import Fasta
import os
from tqdm import tqdm
from time import time
import pynvml
import argparse
from captum.attr import IntegratedGradients, LayerConductance

def parse_args():
    parser = argparse.ArgumentParser(description='''Predicting binding sites and histone modifications from sequence data
    Note:
    1. The model is trained on the sequence data with the sequence length of 600 bp, so the sequence length of the input data must be 600 bp.
    2. Any sequence containing N will be filtered out.
    ''')
    parser.add_argument('--bed_path', type=str, default=None, help='Path to BED file (Must used with --genome_path)')
    parser.add_argument('--genome_path', type=str, default=None, help='Path to genome file (Must used with --bed_path)')
    parser.add_argument('--fasta_path', type=str, default=None, help='Path to fasta file (Used without --bed_path and --genome_path)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file, set to: /public/home/zhuchenglong/project/13.final_preds/bin19/input.pt ')
    parser.add_argument('--file_prefix', type=str, default=f'{sys.argv[0]}', help='prefix for output file')
    parser.add_argument('--output_dir', type=str, default='.', help='Path to output dir')
    parser.add_argument('--thread', type=int, default=5, help='Number of cpu threads to use')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for prediction, default is 1024')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use, default is 0')
    parser.add_argument('--explain', default=False, action='store_true', help='Enable model explanation (Based on IntegratedGradients method)')
    # parser.add_argument('--strand', type=str, default='+', help='which strand to predict, + or -')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parser.parse_args().gpu)
    return parser.parse_args()

def part_nucl_explainer(explainer, chunk_size, input_f, input_r, baseline_data, target, nucl='nucl'):
    results_in_cpu = []
    batch_f = input_f.repeat(chunk_size, 1, 1)
    batch_r = input_r.repeat(chunk_size, 1, 1)

    for i in range(0, 416, chunk_size):
        chunk_end = min(i + chunk_size, 416)
        current_target_chunk = target[i:chunk_end]

        attr_f = explainer.attribute(batch_f, baselines=baseline_data, target=current_target_chunk)
        attr_r = explainer.attribute(batch_r, baselines=baseline_data, target=current_target_chunk)

        if nucl == 'nucl':
            attr_r = torch.flip(attr_r, [1])
        attr = (attr_f + attr_r) / 2
        attr = attr_f
        attr = attr.detach().cpu().numpy()
        del attr_f , attr_r
        torch.cuda.empty_cache()
        results_in_cpu.append(attr)

    attr = np.concatenate(results_in_cpu, axis=0)
    return attr

def predictor(dataloader, model, explainer, baseline_data, output_file_prefix, hist_names, tf_names, hist_std, hist_mean):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')
        device_type = 'cuda'
        handler = pynvml.nvmlDeviceGetHandleByIndex(0)
        first_state = True
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        first_state = False
        print("Warning: CUDA is not available. Using CPU.")


    if explainer:
        target = [x for x in range(416)]
        with torch.autocast(device_type=device_type):
            for batch_f, batch_r, batch_y in tqdm(dataloader, desc=f'Predicting '):
                batch_f = batch_f.to(device)
                batch_r = batch_r.to(device)
                if len(baseline_data) > len(batch_f):
                    baseline_data = baseline_data[:len(batch_f)]
                
                output_hist_f, output_tf_f = model(batch_f)
                output_hist_r, output_tf_r = model(batch_r)
                output_hist = (output_hist_f + output_hist_r) / 2
                output_tf = (output_tf_f + output_tf_r) / 2

                output_hist = output_hist.detach().cpu().numpy()
                output_hist = output_hist * hist_std + hist_mean

                output_tf = output_tf.detach().cpu().numpy()
                torch.cuda.empty_cache()

                chunk_size = 26
                nucl2hist_attr = part_nucl_explainer(explainer[0], chunk_size, batch_f, batch_r, baseline_data, target, 'nucl')
                nucl2tf_attr = part_nucl_explainer(explainer[1], chunk_size, batch_f, batch_r, baseline_data, target, 'nucl')
                tf2hist_attr = part_nucl_explainer(explainer[2], chunk_size, batch_f, batch_r, baseline_data, target, 'tf')

                if first_state:
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
                    first_state = False
                
                with h5py.File(f'{output_file_prefix}.{batch_y[0]}.h5', 'w') as f:
                    f.create_dataset('output_hist', data=output_hist, compression='gzip')
                    f.create_dataset('output_tf', data=output_tf, compression='gzip')
                    f.create_dataset('hist_attr', data=nucl2hist_attr, compression='gzip')
                    f.create_dataset('tf_attr', data=nucl2tf_attr, compression='gzip')
                    f.create_dataset('tf2hist_attr', data=tf2hist_attr, compression='gzip')
                    f.create_dataset('hist_names', data=hist_names, compression='gzip')
                    f.create_dataset('tf_names', data=tf_names, compression='gzip')

    else:
        BUFFER_SIZE_IN_BATCHES = 1000

        # 初始化空的缓冲区列表
        hist_buffer = []
        tf_buffer = []
        current_idx = 0
        num_samples = len(dataloader.dataset)
        hist_std_gpu = torch.tensor(hist_std).to(device)
        hist_mean_gpu = torch.tensor(hist_mean).to(device)

        with h5py.File(f'{output_file_prefix}_pred.h5', 'w') as f:
            f.create_dataset('hist_names', data=hist_names, compression='lzf')
            f.create_dataset('tf_names', data=tf_names, compression='lzf')
            dset_hist = f.create_dataset('hist_preds', 
                                 shape=(num_samples, 416), 
                                 dtype=np.float32, 
                                 chunks=True,
                                 compression='lzf')
            dset_tf = f.create_dataset('tf_preds', 
                                 shape=(num_samples, 2054), 
                                 dtype=np.float32, 
                                 chunks=True,
                                 compression='lzf')
            
            with torch.inference_mode(), torch.autocast(device_type=device_type):  
                for batch_f, batch_r, batch_y in tqdm(dataloader, desc=f'Predicting '):
                    batch_f = batch_f.to(device)
                    batch_r = batch_r.to(device)

                    output_hist_f, output_tf_f = model(batch_f)
                    output_hist_r, output_tf_r = model(batch_r)
                    output_hist = (output_hist_f + output_hist_r) / 2

                    output_hist = output_hist * hist_std_gpu + hist_mean_gpu
                    output_tf = (output_tf_f + output_tf_r) / 2

                    output_hist = output_hist.detach().cpu().numpy()
                    output_tf = output_tf.detach().cpu().numpy()

                    batch_size = batch_f.shape[0]
                    end_idx = current_idx + batch_size
                    hist_buffer.append(output_hist)
                    tf_buffer.append(output_tf)
                    if len(hist_buffer) >= BUFFER_SIZE_IN_BATCHES:
                        # 将缓冲区中的数据合并成一个大的Numpy数组
                        hist_chunk = np.concatenate(hist_buffer, axis=0)
                        tf_chunk = np.concatenate(tf_buffer, axis=0)
                        
                        # 获取这个大块数据的大小
                        chunk_size = hist_chunk.shape[0]
                        end_idx = current_idx + chunk_size

                        dset_hist[current_idx:end_idx] = hist_chunk
                        dset_tf[current_idx:end_idx] = tf_chunk
                        # 清空缓冲区并更新索引
                        hist_buffer.clear()
                        tf_buffer.clear()
                        current_idx = end_idx

                    if first_state:
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
                        first_state = False
                    
            if hist_buffer:
                hist_chunk = np.concatenate(hist_buffer, axis=0)
                tf_chunk = np.concatenate(tf_buffer, axis=0)
                
                chunk_size = hist_chunk.shape[0]
                end_idx = current_idx + chunk_size
                
                dset_hist[current_idx:end_idx] = hist_chunk
                dset_tf[current_idx:end_idx] = tf_chunk


            if device_type == 'cuda':
                print(f'GPU memory usage: {meminfo.used / 1024**2:.2f} MB')
                torch.cuda.empty_cache()

class model_wrapper(torch.nn.Module):
    def __init__(self, model, output_idx):
        super(model_wrapper, self).__init__()
        self.model = model
        self.output_idx = output_idx

    def forward(self, x):
        output = self.model(x)
        return output[self.output_idx]

def main_func(args):
    start_time = time()
    gpu_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = torch.device(gpu_type)
        pynvml.nvmlInit()
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    data_height = 0
    if args.fasta_path is not None:
        max_u = 0
        pos_list = []

        with open(args.fasta_path, 'r') as f:
            id, seq = '', ''
            for line in f:
                if line.startswith('>'):
                    if (id != '') and (seq != ''):
                        end = len(seq)
                        max_u = len(id) if len(id) > max_u else max_u
                        if end != 600:
                            print(f"Warning: Sequence {id} length is not 600. Skipped.")
                        else:
                            pos_list.append(('mm10', id, 0, end))
                    
                    line = line.strip().lstrip('>')
                    id = line
                    seq = ''
                else:
                    seq += line.strip()
            if (id != '') and (seq != ''):
                end = len(seq)
                max_u = len(id) if len(id) > max_u else max_u
                if end != 600:
                    print(f"Warning: Sequence {id} length is not 600. Skipped.")
                else:
                    pos_list.append(('mm10', id, 0, end))
                
        Fasta(args.fasta_path)
        print(f'pyfaidx indexing {args.fasta_path} ...')

        pos_np = np.array(pos_list, dtype=[('species', 'S5'), ('chrom', f'S{max_u+1}'), ('start', 'i4'), ('end', 'i4')])
        with h5py.File(f'{args.fasta_path}.h5', 'w') as f:
            f.create_dataset('pos', data=pos_np)

        data_height = len(pos_list)
        dataset = Bed2Code(f'{args.fasta_path}.h5', args.fasta_path, '/tmp/zcl_enPred/GRCh38.clean.fa', data_height)
    else:
        if (args.bed_path is not None) and (args.genome_path is not None):
            Fasta(args.genome_path)
            print(f'pyfaidx indexing {args.genome_path} ...')

            input_file = args.bed_path 
            if args.bed_path.endswith('.h5'):
                input_file = args.bed_path
                data_height = h5py.File(input_file, 'r')['pos'].shape[0]
            else:
                max_u = 0
                pos_list = []
                with open(args.bed_path, 'r') as f:
                    for line in f:
                        line = line.strip().split('\t')
                        chrom, start, end = line[0], int(line[1]), int(line[2])
                        max_u = len(chrom) if len(chrom) > max_u else max_u
                        pos_list.append(('mm10', chrom, start, end))
                pos_np = np.array(pos_list, dtype=[('species', 'S5'), ('chrom', f'S{max_u+1}'), ('start', 'i4'), ('end', 'i4')])
                with h5py.File(f'{args.bed_path}.h5', 'w') as f:
                    f.create_dataset('pos', data=pos_np)
                input_file = f'{args.bed_path}.h5'
                data_height = len(pos_list)
            dataset = Bed2Code(input_file, args.genome_path, '/tmp/zcl_enPred/GRCh38.clean.fa', data_height)
        else:
            raise ValueError("Please provide either --bed_path and --genome_path or --fasta_path")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    baseline_data = None
    batch_size=args.batch_size
    if args.explain:
        batch_size = 1
        baseline_data = torch.zeros((1, 600, 4), dtype=torch.float32, device=device)
        print("Warning: Baseline data is not provided. Using zero baseline shape.")

    batch_size = min(batch_size, data_height)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.thread, pin_memory=True, prefetch_factor=2)


    model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model_path = torch.load(model_path, weights_only=False)
    model_pt, tf_names, tf_thresholds, hist_names, hist_std, hist_mean = model_path['pt'], model_path['tf_names'], model_path['tf_thresholds'], model_path['hist_names'], model_path['hist_std'], model_path['hist_mean']
    
    model = FunSeq()
    model.load_state_dict(model_pt, strict=True)
    model = torch.compile(model)
    model = model.eval().to(device)


    HISTexplainer = None
    TFexplainer = None
    TF2HISTexplainer = None
    explainer = None
    if args.explain:
        hist_model1 = model_wrapper(model, 0)
        hist_model1 = torch.compile(hist_model1)
        hist_model1 = hist_model1.eval().to(device)

        hist_model2 = model_wrapper(model, 0)
        hist_model2 = torch.compile(hist_model2)
        hist_model2 = hist_model2.eval().to(device)

        hist_model3 = model_wrapper(model, 0)
        hist_model3 = torch.compile(hist_model3)
        hist_model3 = hist_model3.eval().to(device)

        TF2HISTexplainer = LayerConductance(hist_model1, hist_model1.model.catg)
        HISTexplainer = IntegratedGradients(hist_model2, multiply_by_inputs=True)
        TFexplainer = IntegratedGradients(hist_model3.model.catg, multiply_by_inputs=True)
        explainer = (HISTexplainer, TFexplainer, TF2HISTexplainer)
    
    output_file_prefix = f'{args.output_dir}/{args.file_prefix}'
    predictor(dataloader, model, explainer, baseline_data, output_file_prefix, hist_names, tf_names, hist_std, hist_mean)



    # with h5py.File(f'{args.fasta_path}.h5', 'w') as f:
    #     f.create_dataset('predictions', data=predictions)

    # if tf_idx is not None:
    #     predictions = (predictions > tf_thresholds[tf_idx]).astype(np.int8)
    # else:
    #     predictions = (predictions > tf_thresholds).astype(np.int8)

    # result_df = pd.DataFrame(
    #     data=predictions,
    #     columns=tf_names
    # )
    
    # if args.bed_path is not None:
    #     result_df[['chr','start', 'end']] = bed_data[['chr','start', 'end']]
    #     cols = ['chr','start', 'end'] + tf_names
    # elif args.fasta_path is not None:
    #     result_df['id'] = bed_data['id']
    #     cols = ['id'] + tf_names

    # result_df = result_df[cols]
    
    # result_df['label'] = bed_data['label']
    
    # if args.explain:
    #     result_df[f'attribution'] = attributions.tolist()
    # y_true = result_df['label'].values.astype(np.int32)
    # y_score = result_df[tf_names[0]].values.astype(np.float32)
    # try:
    #     auroc = roc_auc_score(y_true, y_score)
    #     auprc = average_precision_score(y_true, y_score)
    #     print(f"\nEvaluation Metrics:\nAUROC: {auroc:.4f}\nAUPRC: {auprc:.4f}")
    # except ValueError as e:
    #     print(f"\nEvaluation failed: {str(e)} (可能因为标签缺少正/负样本)")


    # result_df = pl.from_pandas(result_df)
    # result_df.write_csv(f'{args.output_dir}/{args.file_prefix}.out', include_header=True, separator='\t')

    # print(f"Prediction finished. Output saved to {args.output_dir}/{args.file_prefix}.out")

    end_time = time()
    print(f"Used time: {(end_time - start_time) / 60 :.2f} minutes")


if __name__ == '__main__':
    main_func(parse_args())
