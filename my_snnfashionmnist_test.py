import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, layer, functional, surrogate
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# 中文字体设置
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
plt.switch_backend('TkAgg')

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数（与训练时保持一致）
T = 8  # 脉冲序列时间步长
tau = 2.0  # LIF神经元时间常数

# Fashion-MNIST 类别名称
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# 脉冲编码工具
def poisson_encode(x, T):
    """将图像转换为泊松脉冲序列"""
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # 归一化到[0,1]
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
        spike_img = poisson_encode(img.unsqueeze(0), self.T).squeeze(0)
        return spike_img, label, img  # 返回脉冲序列、标签和原始图像


# 脉冲神经网络模型
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
                x_dummy = x[:, 0].to(device)  # [N, 1, 28, 28]
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
        out_spikes = 0.0
        for t in range(self.T):
            x_t = x[:, t].to(device)  # 取第t个时间步的输入

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


# 自定义图片加载与预测
def load_custom_image(image_path, transform):
    """加载并预处理自定义图片"""
    try:
        original_img = Image.open(image_path).convert('L')  # 转为灰度图
    except FileNotFoundError:
        raise FileNotFoundError(f"自定义图片不存在：{image_path}")
    except Exception as e:
        raise RuntimeError(f"加载图片失败：{str(e)}")

    processed_img = transform(original_img).unsqueeze(0)  # 增加批量维度 [1,1,28,28]
    return original_img, processed_img


def predict_custom_image(model, image_path, transform, T, device):
    """预测单张自定义图片"""
    original_img, processed_img = load_custom_image(image_path, transform)
    spike_img = poisson_encode(processed_img, T).squeeze(0)  # [T,1,28,28]

    model.eval()
    with torch.no_grad():
        input_tensor = spike_img.unsqueeze(0).to(device)  # [1, T, 1, 28, 28]
        output = model(input_tensor)  # [1, 10]
        pred_prob = torch.softmax(output, dim=1).squeeze(0)
        pred_label = torch.argmax(pred_prob).item()
        pred_confidence = pred_prob[pred_label].item()

    return original_img, spike_img, pred_label, pred_confidence


# 可视化函数（支持测试集和自定义图片）
def visualize_prediction(model, dataset, data_indices, is_custom=False, device=device, show_time_steps=4):
    """
    可视化预测结果
    :param model: SNN模型
    :param dataset: 数据集（测试集或自定义图片路径列表）
    :param data_indices: 样本索引列表
    :param is_custom: 是否为自定义图片
    :param device: 计算设备
    :param show_time_steps: 显示的脉冲时间步数量
    """
    step_indices = [int(T / show_time_steps * i) for i in range(show_time_steps)]
    num_samples = len(data_indices)

    fig, axes = plt.subplots(
        num_samples, 1 + show_time_steps,
        figsize=(3 * (1 + show_time_steps), 3 * num_samples)
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    model.eval()
    with torch.no_grad():
        for idx, source_idx in enumerate(data_indices):
            if not is_custom:
                # 处理测试集样本
                spike_img, true_label, original_img = dataset[source_idx]
                input_tensor = spike_img.unsqueeze(0).to(device)
                output = model(input_tensor)
                pred_prob = torch.softmax(output, dim=1).squeeze(0)
                pred_label = torch.argmax(pred_prob).item()
                pred_confidence = pred_prob[pred_label].item()

                title = (f"原图\n真实: {class_names[true_label]}\n"
                         f"预测: {class_names[pred_label]}\n置信度: {pred_confidence:.2f}")
                original_img_np = original_img.squeeze().numpy()
            else:
                # 处理自定义图片
                image_path = dataset[source_idx]  # dataset此时为路径列表
                original_img, spike_img, pred_label, pred_confidence = predict_custom_image(
                    model, image_path, transform, T, device
                )
                title = (f"自定义图片\n预测: {class_names[pred_label]}\n"
                         f"置信度: {pred_confidence:.2f}")
                original_img_np = original_img

            # 显示原图
            ax = axes[idx, 0]
            ax.imshow(original_img_np, cmap='gray')
            ax.set_title(title, fontsize=9)
            ax.axis('off')

            # 显示脉冲编码过程
            for step_idx, t in enumerate(step_indices):
                spike_t = spike_img[t].squeeze().numpy()
                ax = axes[idx, step_idx + 1]
                ax.imshow(spike_t, cmap='gray_r')
                ax.set_title(f"时间步 {t + 1}", fontsize=8)
                ax.axis('off')

    plt.tight_layout()
    plt.show()


# 主函数
if __name__ == '__main__':
    # 数据预处理（与训练时完全一致）
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST专用归一化参数
    ])

    # 加载模型权重
    model = SNN(T, tau).to(device)
    model_path = 'best_snn_fashion_mnist.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ 成功加载模型权重: {model_path}")
    else:
        raise FileNotFoundError(f"❌ 未找到模型文件: {model_path}")

    # 预测模式选择（二选一）

    # 模式A：预测Fashion-MNIST测试集
    """
    original_test_dataset = datasets.FashionMNIST(
        root='./data2', train=False, transform=transform, download=True
    )
    test_dataset = PoissonEncodedDataset(original_test_dataset, T)
    test_indices = random.sample(range(len(test_dataset)), 3)  # 随机选择3个样本
    visualize_prediction(model, test_dataset, test_indices, is_custom=False, device=device)
"""
    #模式B：预测自定义图片（取消注释并修改路径）
    custom_image_paths = [
        "E:\TEST_CODE\Learn_SNNCODE\T2\imgs\cloth4.png",
        "E:\TEST_CODE\Learn_SNNCODE\T2\imgs\cloth3.png"
    ]
    # 检查图片路径是否存在
    for path in custom_image_paths:
        if not os.path.exists(path):
            print(f"⚠️ 警告：自定义图片不存在 - {path}")
    # 生成索引列表（0到图片数量-1）
    custom_indices = list(range(len(custom_image_paths)))
    visualize_prediction(model, custom_image_paths, custom_indices, is_custom=True, device=device)
