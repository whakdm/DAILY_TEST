import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, layer, functional, surrogate
import matplotlib.pyplot as plt
import random
import os



# 设置中文字体
import matplotlib.font_manager as fm  # 新增：用于字体管理
def set_chinese_font():
    chinese_fonts = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "SimSun", "WenQuanYi Micro Hei", "Heiti TC"]
    available_fonts = [f for f in chinese_fonts if any(f.lower() in font.lower() for font in fm.findSystemFonts())]
    if available_fonts:
        plt.rcParams["font.family"] = [available_fonts[0]]
    else:
        plt.rcParams["font.family"] = ["sans-serif"]
        print("警告：未找到中文字体，可能无法正常显示中文")


set_chinese_font()

plt.switch_backend('TkAgg')

# 1. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 超参数（与训练时保持一致）
T = 8  # 脉冲序列时间步长
tau = 2.0  # LIF神经元时间常数


# 3. 脉冲编码工具（与训练时保持一致）
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
        return spike_img, label, img  # 返回脉冲序列、标签和原始图像


# 4. 脉冲神经网络模型（与训练时保持一致）
class SNN(nn.Module):
    def __init__(self, T, tau):
        super().__init__()
        self.T = T
        self.tau = tau
        self.surrogate = surrogate.ATan()  # 替代梯度函数

        # 定义LIF神经元
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

        # 首次前向传播时手动初始化膜电位
        if not hasattr(self.lif1, 'v'):
            with torch.no_grad():
                x_dummy = x[:, 0]  # [N, 1, 28, 28]
                x_dummy = self.conv1(x_dummy)
                x_dummy = self.bn1(x_dummy)
                self.lif1.v = torch.zeros_like(x_dummy, device=device)

                x_dummy = self.lif1(x_dummy)
                x_dummy = self.pool1(x_dummy)
                x_dummy = self.conv2(x_dummy)
                x_dummy = self.bn2(x_dummy)
                self.lif2.v = torch.zeros_like(x_dummy, device=device)

                x_dummy = self.lif2(x_dummy)
                x_dummy = self.pool2(x_dummy)
                x_dummy = self.flatten(x_dummy)
                x_dummy = self.fc1(x_dummy)
                self.lif3.v = torch.zeros_like(x_dummy, device=device)

                x_dummy = self.lif3(x_dummy)
                x_dummy = self.fc2(x_dummy)
                self.lif4.v = torch.zeros_like(x_dummy, device=device)

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


# 5. 完整可视化函数：原图 + 编码过程 + 预测结果
def visualize_encoding_and_prediction(model, test_dataset, device, num_samples=5, show_time_steps=4):
    """
    可视化每张图片的完整信息：
    - 原始图像
    - 脉冲编码过程（时间步上的脉冲发放）
    - 模型预测结果
    """
    # Fashion-MNIST 类别名称
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # 随机选择样本索引
    indices = random.sample(range(len(test_dataset)), num_samples)

    # 创建画布：每行显示1个样本的原图 + 脉冲编码过程 + 预测结果
    # 布局：num_samples行，1(原图) + show_time_steps(脉冲步) 列
    fig, axes = plt.subplots(
        num_samples,
        1 + show_time_steps,
        figsize=(3 * (1 + show_time_steps), 3 * num_samples)
    )

    model.eval()  # 切换到评估模式

    with torch.no_grad():  # 关闭梯度计算
        for sample_idx, idx in enumerate(indices):
            # 获取数据：脉冲序列、标签、原始图像
            spike_img, true_label, original_img = test_dataset[idx]
            # 转换为模型输入形状 [1, T, 1, 28, 28]
            spike_input = spike_img.unsqueeze(0).to(device)

            # 模型预测
            output = model(spike_input)
            _, pred_label = torch.max(output, 1)
            pred_label = pred_label.item()

            # 1. 显示原始图像
            ax = axes[sample_idx, 0] if num_samples > 1 else axes[0]
            ax.imshow(original_img.squeeze().numpy(), cmap='gray')
            ax.set_title(
                f"原图\n真实: {class_names[true_label]}\n预测: {class_names[pred_label]}",
                fontsize=9
            )
            ax.axis('off')

            # 2. 显示脉冲编码过程（选择部分时间步）
            # 计算需要显示的时间步索引（均匀分布）
            step_indices = [int(T / show_time_steps * i) for i in range(show_time_steps)]

            for step_idx, t in enumerate(step_indices):
                # 获取第t个时间步的脉冲图像
                spike_t = spike_img[t].squeeze().numpy()  # 形状: [28, 28]

                # 显示脉冲图像（白色表示发放脉冲）
                ax = axes[sample_idx, step_idx + 1] if num_samples > 1 else axes[step_idx + 1]
                ax.imshow(spike_t, cmap='gray_r')  # 反转灰度，脉冲点为白色
                ax.set_title(f"时间步 {t + 1}", fontsize=8)
                ax.axis('off')

    plt.tight_layout()
    plt.show()


# 6. 主函数
if __name__ == '__main__':
    # 数据预处理（与训练时保持一致）
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST专用归一化参数
    ])

    # 加载Fashion-MNIST测试集
    original_test_dataset = datasets.FashionMNIST(
        root='./data2',
        train=False,
        transform=transform,
        download=True
    )

    # 转换为脉冲编码数据集
    test_dataset = PoissonEncodedDataset(original_test_dataset, T)

    # 初始化模型并加载最佳权重
    model = SNN(T, tau).to(device)
    model_path = 'best_snn_fashion_mnist.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"成功加载模型权重: {model_path}")
    else:
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    # 可视化编码过程和预测结果（显示5个样本，每个样本显示4个时间步）
    visualize_encoding_and_prediction(
        model,
        test_dataset,
        device,
        num_samples=5,
        show_time_steps=4  # 每个样本显示4个时间步的编码
    )
