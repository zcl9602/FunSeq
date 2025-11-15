import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import ConfusionMatrix, AUROC, AveragePrecision
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn as nn
import time 
import os
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_epoch = None 
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, epoch):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_epoch = epoch  
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif val_loss < self.best_score:
            self.best_score = val_loss
            self.best_epoch = epoch 
            self.counter = 0
        return self.early_stop

    def get_best_epoch(self):
        return self.best_epoch
    

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0, reduction='mean', eps=1e-8):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = torch.sigmoid(inputs)
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        pt = targets * pt + (1 - targets) * (1 - pt)   # pt = p if y=1 else 1-p
        pt = 1 - pt
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (pt ** self.gamma) * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def stat_metric(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    acc = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    
    TP /= total
    TN /= total
    FP /= total
    FN /= total
    return total, acc, precision, recall, f1, TP, TN, FP, FN

def name_output(output_dir):
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs(output_dir, exist_ok=True)
    return f'{output_dir}/{current_time}.log', f'{output_dir}/{current_time}.pth'

def train_epoch(model, optimizer, criterion, train_loader, device, threshold, epoch):
    model.train()
    total_loss = 0.0
    count = 0
    
    cm_metric = ConfusionMatrix(task="binary", num_classes=2).to(device)
    auroc_metric = AUROC(task="binary").to(device)
    auprc_metric = AveragePrecision(task="binary").to(device)

    for X, y in tqdm(train_loader, desc=f'Training epoch {epoch+1}'):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)

        outputs = model(X)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        count += X.size(0)
        with torch.no_grad():
            outputs = torch.sigmoid(outputs)
            preds = (outputs > threshold).long()
            cm_metric.update(preds, y.long())
            auroc_metric.update(outputs, y.long())
            auprc_metric.update(outputs, y.long())

    avg_loss = total_loss / count if count != 0 else 0.0

    cm = cm_metric.compute()
    TN, FP, FN, TP = cm[0, 0].item(), cm[0, 1].item(), cm[1, 0].item(), cm[1, 1].item()
    total, acc, precision, recall, f1, TP, TN, FP, FN = stat_metric(TP, TN, FP, FN)

    return (
        avg_loss, TP, TN, FP, FN, acc, precision, recall, f1,
        auroc_metric.compute().item(),
        auprc_metric.compute().item()
    )

def evaluate_epoch(model, criterion, val_loader, device, threshold, epoch):
    model.eval()
    total_loss = 0.0
    count = 0

    cm_metric = ConfusionMatrix(task="binary", num_classes=2).to(device)
    auroc_metric = AUROC(task="binary").to(device)
    auprc_metric = AveragePrecision(task="binary").to(device)
    all_probs = []
    all_trues = []
    
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc=f'Validing epoch {epoch+1}'):
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item() * X.size(0)
            count += X.size(0)
            
            probs = torch.sigmoid(outputs)
            # preds = (probs > threshold).long()
            
            # cm_metric.update(preds, y.long())
            # auroc_metric.update(probs, y.long())
            # auprc_metric.update(probs, y.long())
            all_probs.append(probs)
            all_trues.append(y.long())


    avg_loss = total_loss / count if count != 0 else 0.0
    probs = torch.cat(all_probs, dim=0)
    trues = torch.cat(all_trues, dim=0)
    # probs = np.concatenate(all_probs)
    # trues = np.concatenate(all_trues)
    preds = (probs > threshold).long()

    cm_metric.update(preds, trues)
    auroc_metric.update(probs, trues)
    auprc_metric.update(probs, trues)
    cm = cm_metric.compute()
    TN, FP, FN, TP = cm[0, 0].item(), cm[0, 1].item(), cm[1, 0].item(), cm[1, 1].item()
    total, acc, precision, recall, f1, TP, TN, FP, FN = stat_metric(TP, TN, FP, FN)

    return (
        avg_loss, TP, TN, FP, FN, acc, precision, recall, f1,
        auroc_metric.compute().item(),
        auprc_metric.compute().item(),
        probs.cpu().numpy()
    )

def perpare_sampler(train_set):
    train_labels = np.array([label for _, label in train_set])  # 示例：从训练集中提取标签
    # 确保标签是整数类型
    train_labels = train_labels.astype(int)
    # 获取每个类别的样本数
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    # 计算每个类别的权重：权重 = 1 / 类别样本数
    weight = 1. / class_sample_count.astype(float)
    # 为每个样本分配权重
    samples_weight = np.array([weight[label] for label in train_labels])  # 这里确保每个标签的索引是整数类型
    # 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    return sampler


def main_worker(input_model, train_set, valid_set,
                output_dir='./train_output',
                learning_rate=1e-4,
                num_epochs=200,
                batch_size=128,
                num_workers=4,
                early_stop=10,
                early_stop_delta=0.001,
                rank=0, 
                threshold=0.5,
                alpha=0.5,
                gamma=0):
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')

    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    model = input_model.to(device)
    # model = torch.compile(model)

    sampler = perpare_sampler(train_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler)
    val_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    log_file, model_file = name_output(output_dir)
    early_stopping = EarlyStopping(patience=early_stop, delta=early_stop_delta)
    auroc = 0

    with open(log_file, 'a') as log_p:
        # record all the hyperparameters
        log_line = (
            f"rank={rank}, "
            f"model={input_model.__class__.__name__}\n"
            f"train_data={len(train_set)}\n"
            f"valid_data={len(valid_set)}\n"
            f"learning_rate={learning_rate}\n"
            f"num_epochs={num_epochs}\n"
            f"batch_size={batch_size}\n"
            f"num_workers={num_workers}\n"
            f"early_stop={early_stop}\n"
            f"early_stop_delta={early_stop_delta}\n"
            f"threshold={threshold}\n"
            f"alpha={alpha}\n"
            f"gamma={gamma}\n"
            f"-----------\n"
#            f"hyperparameters:\n"
#            f"-----------\n"
#            f"{model.get_hyper_params()}\n"
#            f"-----------\n"
        )

        log_p.write(log_line)
        print(log_line)

        for epoch in range(num_epochs):
            
            train_metrics = train_epoch(model, optimizer, criterion, train_loader, device, threshold, epoch)
            val_metrics = evaluate_epoch(model, criterion, val_loader, device, threshold, epoch)
            
            current_lr = optimizer.param_groups[0]['lr']
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            log_line = (
                f"{current_time} => Epoch {epoch+1}:\n"
                f"Train Loss = {train_metrics[0]:.4g} | "
                f"TP: {train_metrics[1]:.4g} TN: {train_metrics[2]:.4g} "
                f"FP: {train_metrics[3]:.4g} FN: {train_metrics[4]:.4g} | "
                f"Acc = {train_metrics[5]:.4g} | Precision = {train_metrics[6]:.4g} | "
                f"Recall = {train_metrics[7]:.4g} | F1 = {train_metrics[8]:.4g} | "
                f"AUROC = {train_metrics[9]:.4g} | AUPRC = {train_metrics[10]:.4g}\n"
                f"Valid Loss = {val_metrics[0]:.4g} | "
                f"TP: {val_metrics[1]:.4g} TN: {val_metrics[2]:.4g} "
                f"FP: {val_metrics[3]:.4g} FN: {val_metrics[4]:.4g} | "
                f"Acc = {val_metrics[5]:.4g} | Precision = {val_metrics[6]:.4g} | "
                f"Recall = {val_metrics[7]:.4g} | F1 = {val_metrics[8]:.4g} | "
                f"AUROC = {val_metrics[9]:.4g} | AUPRC = {val_metrics[10]:.4g}\n"
                f"Learning Rate = {current_lr:.2e}\n"
            )

            log_p.write(log_line)
            print(log_line)

            scheduler.step()

            if early_stopping(val_metrics[0], epoch):
                break

            elif epoch == early_stopping.get_best_epoch():
                model_dict = model.state_dict()
                model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
                torch.save(model_dict, model_file)
                auroc = val_metrics[9]
                with open(f'{model_file}.probs.txt', 'w') as f:
                    np.savetxt(f, val_metrics[11], fmt='%.4g')
        
        best_epoch = early_stopping.get_best_epoch()
        log_line = f"Best Epoch = {best_epoch+1}\n"
        log_p.write(log_line)
        print(log_line)
        torch.cuda.empty_cache()
        
    return auroc

