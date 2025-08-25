import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, layer, functional, surrogate
import matplotlib.pyplot as plt
import random
import os
from PIL import Image  # 新增：用于加载自定义图片
import warnings
warnings.filterwarnings('ignore')  # 忽略无关警告


# -------------------------- 1. 工具函数：中文字体设置 --------------------------
import matplotlib.font_manager as fm
def set_chinese_font():
    chinese_fonts = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "SimSun", "WenQuanYi Micro Hei", "Heiti TC"]
    available_fonts = [f for f in chinese_fonts if any(f.lower() in font.lower() for font in fm.findSystemFonts())]
    if available_fonts:
        plt.rcParams["font.family"] = [available_fonts[0]]
    else:
        plt.rcParams["font.family"] = ["sans-serif"]
        print("警告：未找到中文字体，可能无法正常显示中文")

set_chinese_font()
plt.switch_backend('TkAgg')  # 确保图像正常显示


# -------------------------- 2. 基础配置：设备、超参数、类别名 --------------------------
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数（必须与训练时完全一致！）
T = 8  # 脉冲序列时间步长
tau = 2.0  # LIF神经元时间常数

# Fashion-MNIST 类别名称（模型训练的目标类别）
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# -------------------------- 3. 核心功能1：脉冲编码（与训练时一致） --------------------------
def poisson_encode(x, T):
    """将图像转换为泊松脉冲序列（输入：[N,C,H,W]，输出：[N,T,C,H,W]）"""
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # 归一化到[0,1]（确保脉冲概率合理）
    spike = torch.rand([x.shape[0], T] + list(x.shape[1:]), device=x.device) < x.unsqueeze(1)
    return spike.float()


class PoissonEncodedDataset(Dataset):
    """包装数据集（支持原始数据集/Fashion-MNIST），自动添加泊松脉冲编码"""
    def __init__(self, dataset, T):
        self.dataset = dataset
        self.T = T

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        spike_img = poisson_encode(img.unsqueeze(0), self.T).squeeze(0)  # [T,C,H,W]
        return spike_img, label, img  # 返回：脉冲序列、标签、原始图像


# -------------------------- 4. 核心功能2：脉冲神经网络模型（与训练时一致） --------------------------
class SNN(nn.Module):
    def __init__(self, T, tau):
        super().__init__()
        self.T = T
        self.tau = tau
        self.surrogate = surrogate.ATan()  # 替代梯度函数（与训练一致）

        # LIF神经元定义
        self.lif1 = neuron.LIFNode(tau=tau, surrogate_function=self.surrogate, detach_reset=True)
        self.lif2 = neuron.LIFNode(tau=tau, surrogate_function=self.surrogate, detach_reset=True)
        self.lif3 = neuron.LIFNode(tau=tau, surrogate_function=self.surrogate, detach_reset=True)
        self.lif4 = neuron.LIFNode(tau=tau, surrogate_function=self.surrogate, detach_reset=True)

        # 网络层（卷积+全连接，输入为28x28灰度图）
        self.conv1 = layer.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(16)
        self.pool1 = layer.MaxPool2d(2, 2)

        self.conv2 = layer.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(32)
        self.pool2 = layer.MaxPool2d(2, 2)

        self.flatten = layer.Flatten()
        self.fc1 = layer.Linear(32 * 7 * 7, 128, bias=False)  # 28→(池化2次)→7
        self.fc2 = layer.Linear(128, 10, bias=False)  # 10个类别

    def forward(self, x):
        """x输入形状：[N, T, 1, 28, 28]（N=批量数，T=时间步）"""
        functional.reset_net(self)  # 每次前向传播前重置神经元状态

        # 首次前向时初始化膜电位（避免维度不匹配）
        if not hasattr(self.lif1, 'v'):
            with torch.no_grad():
                x_dummy = x[:, 0].to(device)  # 取第0个时间步初始化
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

        # 时间步循环：累加所有时间步的脉冲输出
        out_spikes = 0.0
        for t in range(self.T):
            x_t = x[:, t].to(device)  # 取第t个时间步的输入

            # 卷积块1
            x_t = self.conv1(x_t)
            x_t = self.bn1(x_t)
            x_t = self.lif1(x_t)
            x_t = self.pool1(x_t)

            # 卷积块2
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

            out_spikes += x_t  # 累加脉冲

        return out_spikes / self.T  # 返回平均脉冲数（概率化输出）


# -------------------------- 5. 新增功能：自定义图片加载与预测 --------------------------
def load_custom_image(image_path, transform):
    """
    加载并预处理自定义图片
    Args:
        image_path: 自定义图片路径（如"./test_shirt.jpg"）
        transform: 与训练一致的预处理管道
    Returns:
        original_img: 原始PIL图像（用于显示）
        processed_img: 预处理后的张量（[1,1,28,28]，用于编码）
    """
    # 1. 加载图片（支持JPG/PNG等格式）
    try:
        original_img = Image.open(image_path).convert('L')  # 转为灰度图（'L'模式）
    except FileNotFoundError:
        raise FileNotFoundError(f"自定义图片不存在：{image_path}")
    except Exception as e:
        raise RuntimeError(f"加载图片失败：{str(e)}")

    # 2. 应用预处理（与训练时一致）
    processed_img = transform(original_img).unsqueeze(0)  # 增加批量维度 [1,1,28,28]
    return original_img, processed_img


def predict_custom_image(model, image_path, transform, T, device):
    """
    预测单张自定义图片
    Args:
        model: 加载好权重的SNN模型
        image_path: 自定义图片路径
        transform: 预处理管道
        T: 脉冲时间步长
        device: 计算设备
    Returns:
        original_img: 原始图片（PIL格式）
        spike_img: 脉冲序列（[T,1,28,28]）
        pred_label: 预测类别（int）
        pred_prob: 预测类别概率（float，0~1）
    """
    # 1. 加载并预处理图片
    original_img, processed_img = load_custom_image(image_path, transform)

    # 2. 生成脉冲序列
    spike_img = poisson_encode(processed_img, T).squeeze(0)  # [T,1,28,28]

    # 3. 模型预测
    model.eval()
    with torch.no_grad():
        input_tensor = spike_img.unsqueeze(0).to(device)  # [1, T, 1, 28, 28]
        output = model(input_tensor)  # [1, 10]
        pred_prob = torch.softmax(output, dim=1).squeeze(0)  # 概率归一化
        pred_label = torch.argmax(pred_prob).item()  # 预测类别
        pred_confidence = pred_prob[pred_label].item()  # 预测置信度

    return original_img, spike_img, pred_label, pred_confidence


# -------------------------- 6. 可视化函数：支持测试集+自定义图片 --------------------------
def visualize_prediction(model, data_source, is_custom=False, device=T, show_time_steps=4):
    """
    可视化预测结果（支持两种模式：Fashion-MNIST测试集 / 自定义图片）
    Args:
        model: SNN模型
        data_source: 数据来源（测试集样本索引列表 / 自定义图片路径列表）
        is_custom: 是否为自定义图片（True/False）
        device: 计算设备
        show_time_steps: 显示的脉冲时间步数量
    """
    # 计算显示的时间步索引（均匀分布，避免密集）
    step_indices = [int(T / show_time_steps * i) for i in range(show_time_steps)]

    # 处理子图布局
    num_samples = len(data_source)
    fig, axes = plt.subplots(
        num_samples, 1 + show_time_steps,  # 每行：原图 + N个时间步脉冲
        figsize=(3 * (1 + show_time_steps), 3 * num_samples)
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)  # 单样本时调整维度

    # 循环处理每个样本
    for idx, source in enumerate(data_source):
        if not is_custom:
            # 模式1：Fashion-MNIST测试集
            spike_img, true_label, original_img = data_source[source]
            # 模型预测
            model.eval()
            with torch.no_grad():
                input_tensor = spike_img.unsqueeze(0).to(device)
                output = model(input_tensor)
                pred_prob = torch.softmax(output, dim=1).squeeze(0)
                pred_label = torch.argmax(pred_prob).item()
                pred_confidence = pred_prob[pred_label].item()
            # 原图标题（含真实标签）
            title = f"原图\n真实: {class_names[true_label]}\n预测: {class_names[pred_label]}\n置信度: {pred_confidence:.2f}"
            # 转换原图格式（Tensor→numpy）
            original_img_np = original_img.squeeze().numpy()

        else:
            # 模式2：自定义图片
            image_path = source
            original_img, spike_img, pred_label, pred_confidence = predict_custom_image(
                model, image_path, transform, T, device
            )
            # 原图标题（无真实标签）
            title = f"自定义图片\n预测: {class_names[pred_label]}\n置信度: {pred_confidence:.2f}"
            # 转换原图格式（PIL→numpy）
            original_img_np = original_img

        # 1. 显示原图
        ax = axes[idx, 0]
        ax.imshow(original_img_np, cmap='gray')
        ax.set_title(title, fontsize=9)
        ax.axis('off')

        # 2. 显示脉冲编码过程
        for step_idx, t in enumerate(step_indices):
            spike_t = spike_img[t].squeeze().numpy()  # 第t个时间步的脉冲
            ax = axes[idx, step_idx + 1]
            ax.imshow(spike_t, cmap='gray_r')  # 反转灰度：脉冲点为白色
            ax.set_title(f"时间步 {t + 1}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# -------------------------- 7. 主函数：入口逻辑 --------------------------
if __name__ == '__main__':
    # -------------------------- 7.1 数据预处理（与训练时完全一致！） --------------------------
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 缩放到28x28（模型输入尺寸）
        transforms.Grayscale(num_output_channels=1),  # 转为1通道灰度图
        transforms.ToTensor(),  # 转为Tensor（[1,28,28]）
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST统计参数（勿修改）
    ])

    # -------------------------- 7.2 加载模型权重 --------------------------
    model = SNN(T, tau).to(device)
    model_path = 'best_snn_fashion_mnist.pth'  # 训练好的模型路径
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ 成功加载模型权重: {model_path}")
    else:
        raise FileNotFoundError(f"❌ 未找到模型文件: {model_path}（请先训练模型或检查路径）")

    # -------------------------- 7.3 选择预测模式（二选一） --------------------------
    # 模式A：预测Fashion-MNIST测试集（用于验证模型）
    # original_test_dataset = datasets.FashionMNIST(
    #     root='./data2', train=False, transform=transform, download=True
    # )
    # test_dataset = PoissonEncodedDataset(original_test_dataset, T)
    # test_indices = random.sample(range(len(test_dataset)), 3)  # 随机选3个测试样本
    # visualize_prediction(model, test_indices, is_custom=False, device=device)

    # 模式B：预测自定义图片（重点！请修改图片路径）
  #  custom_image_paths = [
        #      "./test_ankle_boot.jpg",  # 示例1：脚踝靴（对应类别9）
        #      "./test_trouser.jpg",     # 示例2：裤子（对应类别1）
    #      "./test_shirt.jpg"        # 示例3：衬衫（对应类别6）
    #  ]
    custom_image_paths = ["E:\TEST_CODE\Learn_SNNCODE\T2\imgs\cloth2.png"]
    # 检查自定义图片是否存在
    for path in custom_image_paths:
        if not os.path.exists(path):
            print(f"⚠️  自定义图片不存在：{path}（请替换为你的图片路径）")
    # 可视化预测结果 is_custom=False：预测 Fashion-MNIST 测试集（保留原功能，用于验证模型）。
    visualize_prediction(model, custom_image_paths, is_custom=True, device=device)