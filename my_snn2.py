import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, layer, functional, surrogate
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# 1. 基础配置与超参数
torch.manual_seed(0)  # 固定随机种子，确保结果可复现
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数设置
batch_size = 128
learning_rate = 1e-3
num_epochs = 10  # 训练总轮次
T = 8  # 脉冲序列时间步长
tau = 2.0  # LIF神经元时间常数


# 2. 脉冲编码工具
def poisson_encode(x, T):
    """将图像转换为泊松脉冲序列"""
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # 归一化到[0,1]
    # 生成脉冲序列，形状为 [N, T, C, H, W]
    spike = torch.rand([x.shape[0], T] + list(x.shape[1:]), device=x.device) < x.unsqueeze(1)
    return spike.float()


class PoissonEncodedDataset(Dataset):
    """包装数据集，自动添加泊松脉冲编码"""

    def __init__(self, dataset, T):
        self.dataset = dataset
        self.T = T

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # 生成脉冲序列（形状：[T, C, H, W]）
        spike_img = poisson_encode(img.unsqueeze(0), self.T).squeeze(0)
        return spike_img, label


# 3. 脉冲神经网络模型（修正膜电位初始化问题）
class SNN(nn.Module):
    def __init__(self, T, tau):
        super().__init__()
        self.T = T
        self.tau = tau
        self.surrogate = surrogate.ATan()  # 替代梯度函数

        # 定义独立的LIF神经元（便于膜电位初始化）
        self.lif1 = neuron.LIFNode(
            tau=tau,
            surrogate_function=self.surrogate,
            detach_reset=True
        )
        self.lif2 = neuron.LIFNode(
            tau=tau,
            surrogate_function=self.surrogate,
            detach_reset=True
        )
        self.lif3 = neuron.LIFNode(
            tau=tau,
            surrogate_function=self.surrogate,
            detach_reset=True
        )
        self.lif4 = neuron.LIFNode(
            tau=tau,
            surrogate_function=self.surrogate,
            detach_reset=True
        )

        # 网络层定义
        self.conv1 = layer.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(16)
        self.pool1 = layer.MaxPool2d(2, 2)

        self.conv2 = layer.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(32)
        self.pool2 = layer.MaxPool2d(2, 2)

        self.flatten = layer.Flatten()
        self.fc1 = layer.Linear(32 * 7 * 7, 128, bias=False)
        self.fc2 = layer.Linear(128, 10, bias=False)

    def forward(self, x):
        # x形状: [N, T, 1, 28, 28]
        functional.reset_net(self)  # 重置所有神经元状态

        # 首次前向传播时手动初始化膜电位（解决尺寸不匹配问题）
        if not hasattr(self.lif1, 'v'):
            with torch.no_grad():
                # 用第一个时间步的数据模拟传播，确定各层尺寸
                x_dummy = x[:, 0]  # [N, 1, 28, 28]

                x_dummy = self.conv1(x_dummy)
                x_dummy = self.bn1(x_dummy)
                self.lif1.v = torch.zeros_like(x_dummy, device=device)  # 适配conv1输出

                x_dummy = self.lif1(x_dummy)
                x_dummy = self.pool1(x_dummy)
                x_dummy = self.conv2(x_dummy)
                x_dummy = self.bn2(x_dummy)
                self.lif2.v = torch.zeros_like(x_dummy, device=device)  # 适配conv2输出

                x_dummy = self.lif2(x_dummy)
                x_dummy = self.pool2(x_dummy)
                x_dummy = self.flatten(x_dummy)
                x_dummy = self.fc1(x_dummy)
                self.lif3.v = torch.zeros_like(x_dummy, device=device)  # 适配fc1输出

                x_dummy = self.lif3(x_dummy)
                x_dummy = self.fc2(x_dummy)
                self.lif4.v = torch.zeros_like(x_dummy, device=device)  # 适配fc2输出

        # 时间步上的前向传播
        out_spikes = 0
        for t in range(self.T):
            x_t = x[:, t]  # 取第t个时间步的输入

            # 第一卷积块
            x_t = self.conv1(x_t)
            x_t = self.bn1(x_t)
            x_t = self.lif1(x_t)
            x_t = self.pool1(x_t)

            # 第二卷积块
            x_t = self.conv2(x_t)
            x_t = self.bn2(x_t)
            x_t = self.lif2(x_t)
            x_t = self.pool2(x_t)

            # 全连接层
            x_t = self.flatten(x_t)
            x_t = self.fc1(x_t)
            x_t = self.lif3(x_t)
            x_t = self.fc2(x_t)
            x_t = self.lif4(x_t)

            out_spikes += x_t  # 累加所有时间步的脉冲

        return out_spikes / self.T  # 返回平均脉冲数


# 4. 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # 数据移至设备
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 清空梯度

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播与参数更新
        loss.backward()
        optimizer.step()

        # 统计指标
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每100个batch打印一次进度
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {total_loss / (batch_idx + 1):.4f}, '
                  f'Train Accuracy: {100. * correct / total:.2f}%')

    return total_loss / len(train_loader), 100. * correct / total


# 5. 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 测试时不计算梯度
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计指标
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(test_loader), 100. * correct / total


# 6. 主程序
if __name__ == '__main__':
    # 数据预处理：确保输入为28×28的灰度图
    # Fashion - MNIST 归一化参数（与 MNIST 不同）
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # 加载 Fashion - MNIST 数据集
    train_dataset = datasets.FashionMNIST(
        root='./data2',
        train=True,
        transform=transform,
        download=True
    )
    test_dataset = datasets.FashionMNIST(
        root='./data2',
        train=False,
        transform=transform,
        download=True
    )

    # 转换为脉冲数据集
    train_dataset = PoissonEncodedDataset(train_dataset, T)
    test_dataset = PoissonEncodedDataset(test_dataset, T)

    # 数据加载器（Windows系统关闭多进程）
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 初始化模型、损失函数和优化器
    model = SNN(T, tau).to(device)
    criterion = nn.CrossEntropyLoss()  # 分类任务损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 学习率衰减

    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')

        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')

        # 测试
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

        # 学习率衰减
        scheduler.step()

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_snn_fashion_mnist.pth')
            print(f'Saved best model with accuracy: {best_acc:.2f}%')

    print(f'\n最佳测试准确率: {best_acc:.2f}%')