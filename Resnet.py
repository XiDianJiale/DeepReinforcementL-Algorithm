import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义基本的残差块（ResBlock），适用于 ResNet-18 和 ResNet-34
class BasicBlock(nn.Module):
    expansion = 1  # 该 block 不改变通道数
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 归一化层，防止梯度爆炸或消失
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 用于匹配输入/输出尺寸的跳跃连接
    
    def forward(self, x):
        identity = x  # 保存输入数据用于跳跃连接
        if self.downsample is not None:
            identity = self.downsample(x)  # 调整残差连接的通道数和尺寸
        
        out = self.conv1(x)  # 为什么conv1能改变通道数？见 conv1 初始化处的 C 变化
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # 残差连接, 这里 identity 需要与 out 形状匹配，否则需要 downsample 处理
        out = F.relu(out)
        
        return out

# ResNet 主体
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64  # 第一层卷积之后的通道数，C从3变64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # C=3 -> C=64，疑问点：为什么这里是 64？
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 下采样，降低计算量，但如何影响特征图尺寸？
        
        # 逐层堆叠 ResBlock
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，最终将特征图变为 1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接分类层
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )  # 为什么这里要做 downsample？
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion  # 更新通道数，以匹配后续层
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))  # 残差块堆叠，确保通道数匹配
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 构造不同版本的 ResNet
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

# 示例：构建 ResNet-18
model = resnet18(num_classes=1000)
print(model)  # 可以查看整个网络结构
