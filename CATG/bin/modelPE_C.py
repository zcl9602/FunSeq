from torch.utils.data import Dataset
import torch
import torch.nn as nn
import math

class GenoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    # Taken from: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Implement the PE function."
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
        with torch.no_grad():
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
        
class GlobalAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.3):
        super().__init__()
        self.mh = nn.MultiheadAttention(input_dim, num_heads=1, dropout=dropout_prob, batch_first=True)
        self.to_out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = self.mh(x, x, x)[0]
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
                seq_len = 600,
                conv1_outdim = 512, 
                conv1_ks = 11,
                conv2_outdim = 256,
                conv2_ks = 7,
                mp_ks = 23,
                mp_sd = 3,
                res_conv1_dim = 64,
                res_conv1_ks = 21,
                res_skip_ks1 = 1,
                res_conv2_dim = 64,
                res_conv2_ks = 115,
                res_skip_ks2 = 1,
                l1_dim = 128,
                l2_dim = 64
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
        self.l2_dim = l2_dim
        self.res_num = 1

        self.conv1 = cnn_bn(4, self.conv1_outdim, kernel_size=self.conv1_ks)
        self.position_embedding = PositionalEncoding(self.conv1_outdim)
        self.conv2 = cnn_mp(self.conv1_outdim, self.conv2_outdim, kernel_size=self.conv2_ks, mp_kernel_size=self.mp_ks, mp_stride=self.mp_sd)
        self.res_block = resblock(self.conv2_outdim, self.res_conv1_dim, self.res_conv2_dim, self.res_conv1_ks, self.res_conv2_ks)
        self.classifier = nn.Sequential(
            nn.Linear(self.res_conv2_dim * 2, self.l2_dim),
#             nn.Linear(self.l1_dim, self.l2_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.l2_dim, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x) 
        x = x.permute(0, 2, 1)  
        x = self.position_embedding(x)
        x = x.transpose(1,2)
        x = self.conv2(x)
        x = self.res_block(x) 
        x = self.classifier(x)
        return x.squeeze(-1)

