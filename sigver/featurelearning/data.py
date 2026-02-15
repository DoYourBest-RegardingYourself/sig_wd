import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms as transforms


class TransformDataset(Dataset):
    """
        Dataset that applies a transform on the data points on __get__item.
    """
    def __init__(self, dataset, transform, transform_index=0):
        self.dataset = dataset
        self.transform = transform
        self.transform_index = transform_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img = data[self.transform_index]

        return tuple((self.transform(img), *data[1:]))



def extract_features(x: np.ndarray,
                     process_function: callable,
                     batch_size: int,
                     input_size: tuple = None) -> np.ndarray:
    """
    批量提取特征的主函数

    参数:
    ----------
    x : np.ndarray
        输入数据，形状应为 (N, C, H, W) 的numpy数组:
        - N: 样本数量
        - C: 通道数 (通常为1表示灰度图)
        - H: 原始高度
        - W: 原始宽度

    process_function : callable
        特征处理函数，接收一个batch数据，返回特征张量
        示例: 训练好的CNN模型的前向传播函数

    batch_size : int
        批处理大小，影响内存使用和计算效率

    input_size : tuple, optional
        目标图像尺寸 (height, width)，若提供会执行中心裁剪

    返回:
    -------
    np.ndarray
        特征矩阵，形状为 (N, D)：
        - N: 样本数量（与输入一致）
        - D: 特征维度（由process_function决定）

    处理流程:
    ----------
    1. 将numpy数据转换为PyTorch数据集
    2. 执行可选的中心裁剪预处理
    3. 分批次进行特征提取
    4. 合并所有批次结果
    """
    # 将原始数据转换为TensorDataset
    # 输入x应为float32类型，形状(N, C, H, W)
    data = TensorDataset(torch.from_numpy(x).float())

    # 数据预处理流程
    if input_size is not None:
        # 定义预处理流水线
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),  # 将张量转换为PIL图像（需要CxHxW格式）
            transforms.CenterCrop(input_size),  # 中心裁剪至目标尺寸
            transforms.ToTensor()  # 转回张量并自动归一化到[0,1]
        ])

        # 应用预处理转换
        # 假设TransformDataset是自定义数据集类，应用转换规则
        data = TransformDataset(data, data_transforms)

    # 创建数据加载器（设置pin_memory=True可加速GPU传输）
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,  # 保持原始顺序
        pin_memory=torch.cuda.is_available()
    )

    result = []
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for batch in data_loader:
            # 处理单个batch（假设process_function返回形状为(B, D)的张量）
            # batch 是包含单个张量的列表，需解包
            features = process_function(batch)
            result.append(features)

    # 合并所有batch结果并转换为numpy数组
    return torch.cat(result, dim=0).cpu().numpy()