![image](https://github.com/user-attachments/assets/22771231-7101-48a8-9201-a5b73a737d8f)

# RadioUNet第一阶段模型优化计划

## 当前理解与方案

根据论文描述，第一阶段网络(μθ1)的目标是预测奇异点分布，流程应为：

1. 使用离散化亥姆霍兹方程从真实RadioMap计算k²场
2. 根据k²<0的条件生成二值奇异点图(binary singularity map)
3. 训练网络预测这个二值奇异点图

您的问题很关键：**应该将k²场作为中间数据保存，还是直接保存二值奇异点图？**

答案是：**两者都需要，但训练目标是二值奇异点图**。

## 优化计划文档

### 1. 数据准备阶段

```python
# 在SingularPointsDataset类中

def generate_k_squared_and_singularity(self, building_mask, complete_radiomap):
    """从完整RadioMap生成k²场和奇异点掩码"""
    # 归一化RadioMap
    norm_radiomap = complete_radiomap.astype(np.float32) / 255.0
    
    # 转为PyTorch张量
    radiomap_tensor = torch.from_numpy(norm_radiomap).unsqueeze(0).unsqueeze(0)
    
    # 使用亥姆霍兹方程计算k²
    with torch.no_grad():
        k_squared, _ = extract_singularities(radiomap_tensor, self.delta_h)
    
    # 转回NumPy
    k_squared_np = k_squared.squeeze().numpy()
    
    # 生成二值奇异点掩码(k²<0的区域)
    singularity_mask = (k_squared_np < 0).astype(np.float32)
    
    # 确保建筑物内部被标记为奇异点
    k_squared_np[building_mask > 0] = -1.0 * (self.k0 ** 2)
    singularity_mask[building_mask > 0] = 1.0
    
    return k_squared_np, singularity_mask

def __getitem__(self, idx):
    # ...现有代码...
    
    # 生成k²场和奇异点掩码
    k_squared, singularity_mask = self.generate_k_squared_and_singularity(
        image_buildings, complete_radiomap)
    
    # 构建网络输入(建筑物掩码和稀疏采样)
    sparse_input = np.stack([image_buildings / 255.0, sparse_samples / 255.0], axis=2)
    
    # 返回样本，同时包含k²场和奇异点掩码
    sample = {
        'input': torch.from_numpy(sparse_input).float().permute(2, 0, 1),
        'k_squared': torch.from_numpy(k_squared[..., np.newaxis]).float().permute(2, 0, 1),
        'singularity_mask': torch.from_numpy(singularity_mask[..., np.newaxis]).float().permute(2, 0, 1),
        'true_field': torch.from_numpy(complete_radiomap[..., np.newaxis]).float().permute(2, 0, 1) / 255.0
    }
    
    return sample
```

### 2. 损失函数修改

```python
class SingularityPredictionLoss(nn.Module):
    def __init__(self, lambda_bce=1.0, lambda_mse=0.1):
        super(SingularityPredictionLoss, self).__init__()
        self.lambda_bce = lambda_bce  # 二值分类损失权重
        self.lambda_mse = lambda_mse  # k²回归损失权重
        self.bce_loss = nn.BCEWithLogitsLoss()  # Sigmoid+BCE
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_k_squared, target_k_squared, target_singularity):
        """
        参数:
            pred_k_squared: 预测的k²场
            target_k_squared: 目标k²场
            target_singularity: 目标奇异点掩码
        """
        # 从预测的k²导出奇异点预测(使用sigmoid变换)
        # 注意：使用-pred_k_squared因为k²<0对应奇异点为1
        pred_singularity_logits = -pred_k_squared  
        
        # 二值交叉熵损失 - 奇异点分类
        bce = self.bce_loss(pred_singularity_logits, target_singularity)
        
        # 均方误差损失 - k²回归
        mse = self.mse_loss(pred_k_squared, target_k_squared)
        
        # 总损失 - 强调二值分类
        total_loss = self.lambda_bce * bce + self.lambda_mse * mse
        
        return total_loss, mse, bce
```

### 3. 训练循环优化

```python
def train_model(model, optimizer, scheduler, criterion, num_epochs=500, 
                dataloaders=None, device="cuda", save_dir=None):
    # ...现有代码...
    
    for epoch in range(num_epochs):
        # ...现有代码...
        
        for phase in ['train', 'val']:
            # ...现有代码...
            
            for batch in dataloaders[phase]:
                inputs = batch['input'].to(device)
                k_squared_true = batch['k_squared'].to(device)
                singularity_mask = batch['singularity_mask'].to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # 前向传播
                    k_squared_pred = model(inputs)
                    
                    # 计算损失
                    loss, mse_loss, bce_loss = criterion(
                        k_squared_pred, k_squared_true, singularity_mask
                    )
                    
                    # 记录损失和指标
                    metrics['loss'] += loss.item() * inputs.size(0)
                    metrics['mse_loss'] += mse_loss.item() * inputs.size(0)
                    metrics['bce_loss'] += bce_loss.item() * inputs.size(0)
                    
                    # 计算评估指标
                    calc_metrics(k_squared_pred, k_squared_true, singularity_mask, metrics)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # ...现有代码...
                
            # 更新学习率 - 基于验证集性能
            if phase == 'val' and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics['loss'] / epoch_samples)
            elif phase == 'train' and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
```

### 4. 评估指标增强

```python
def calc_metrics(pred_k_squared, target_k_squared, target_singularity, metrics):
    """计算奇异点预测的评估指标"""
    # 从k²导出奇异点预测
    pred_singularity = (pred_k_squared < 0).float()
    
    # 准确率
    accuracy = (pred_singularity == target_singularity).float().mean()
    metrics['accuracy'] += accuracy.item() * target_singularity.size(0)
    
    # IoU (Intersection over Union)
    intersection = (pred_singularity * target_singularity).sum(dim=[1, 2, 3])
    union = pred_singularity.sum(dim=[1, 2, 3]) + target_singularity.sum(dim=[1, 2, 3]) - intersection
    iou = (intersection / (union + 1e-6)).mean()
    metrics['iou'] += iou.item() * target_singularity.size(0)
    
    # 精确度 (Precision)
    precision = (pred_singularity * target_singularity).sum(dim=[1, 2, 3]) / (pred_singularity.sum(dim=[1, 2, 3]) + 1e-6)
    metrics['precision'] += precision.mean().item() * target_singularity.size(0)
    
    # 召回率 (Recall)
    recall = (pred_singularity * target_singularity).sum(dim=[1, 2, 3]) / (target_singularity.sum(dim=[1, 2, 3]) + 1e-6)
    metrics['recall'] += recall.mean().item() * target_singularity.size(0)
    
    # F1分数
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    metrics['f1'] += f1.mean().item() * target_singularity.size(0)
    
    return accuracy
```

### 5. 超参数设置

```python
# 命令行参数修改
parser.add_argument('--lambda_bce', type=float, default=1.0, help='二值分类损失权重')
parser.add_argument('--lambda_mse', type=float, default=0.1, help='k²回归损失权重')
parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
parser.add_argument('--batch_size', type=int, default=8, help='批量大小')

# 优化器设置
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

# 学习率调度设置 - 性能驱动衰减
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, 
    min_lr=1e-6, verbose=True
)
```

### 6. 可视化脚本改进

```python
def visualize_singularity_predictions(model, dataset, save_dir, device='cuda', num_samples=5):
    """可视化奇异点预测结果"""
    # ...现有代码...
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_tensor = sample['input'].unsqueeze(0).to(device)
        k_squared_true = sample['k_squared']
        singularity_true = sample['singularity_mask']
        
        # 预测
        with torch.no_grad():
            k_squared_pred = model(input_tensor)
        
        # 从k²导出奇异点预测
        singularity_pred = (k_squared_pred < 0).float().cpu()
        
        # 创建可视化
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
        # 真实奇异点
        axs[0, 0].imshow(singularity_true.squeeze(), cmap='hot')
        axs[0, 0].set_title('真实奇异点')
        
        # 预测奇异点
        axs[0, 1].imshow(singularity_pred.squeeze(), cmap='hot')
        axs[0, 1].set_title('预测奇异点')
        
        # 真实k²场
        axs[1, 0].imshow(k_squared_true.squeeze(), cmap='RdBu', vmin=-1, vmax=1)
        axs[1, 0].set_title('真实k²场')
        
        # 预测k²场
        axs[1, 1].imshow(k_squared_pred.squeeze().cpu(), cmap='RdBu', vmin=-1, vmax=1)
        axs[1, 1].set_title('预测k²场')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'singularity_pred_{i}.png'))
        plt.close()
```

## 执行计划

1. **修改数据集类**：确保返回包含k²场和二值奇异点掩码
2. **更新损失函数**：实现新的`SingularityPredictionLoss`
3. **调整训练循环**：使用二值奇异点图作为主要训练目标
4. **增强评估指标**：添加分类任务相关的指标
5. **优化超参数**：调整学习率和批量大小
6. **改进可视化**：添加奇异点预测可视化功能

这些修改将确保您的第一阶段网络更加专注于奇异点分类任务，而不是k²场的精确回归。通过更加符合论文描述的实现，您的模型应该能够获得更好的收敛性能。

如需进一步讨论或调整，请随时提问！
