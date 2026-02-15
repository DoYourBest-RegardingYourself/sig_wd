# 导入必要的库和模块
import argparse  # 命令行参数解析
import functools  # 高阶函数工具
from typing import Tuple, Optional  # 类型注解支持
import numpy as np  # 数值计算库

# 导入签名验证相关数据集处理模块
from sigver.datasets import available_datasets  # 可用数据集列表
from sigver.datasets.base import IterableDataset  # 可迭代数据集基类
from sigver.datasets.util import process_dataset_images  # 数据集图像处理工具

# 导入签名预处理模块
from sigver.preprocessing.normalize import preprocess_signature  # 签名标准化函数


def process_dataset(dataset: IterableDataset,
                    save_path: str,
                    img_size: Tuple[int, int],
                    subset: Optional[slice] = None) -> None:
    """
    数据集预处理及存储函数

    参数说明：
    dataset    : 可迭代数据集对象，包含签名文件路径
    save_path  : 预处理数据存储路径(.npz格式)
    img_size   : 目标图像尺寸元组(高度, 宽度)
    subset     : 用户子集切片，默认处理全部用户

    处理流程：
    1. 创建图像预处理函数
    2. 执行数据集标准化处理
    3. 保存处理后的多维数组数据
    """
    # 创建部分应用函数，固定预处理参数
    preprocess_fn = functools.partial(preprocess_signature,
                                      canvas_size=dataset.maxsize,  # 使用数据集最大画布尺寸
                                      img_size=img_size,  # 指定目标尺寸
                                      input_size=img_size)  # 保持原始比例不裁剪

    # 设置默认处理范围（全部用户）
    if subset is None:
        subset = slice(None)

    # 执行数据集预处理
    processed = process_dataset_images(dataset, preprocess_fn, img_size, subset)
    x, y, yforg, user_mapping, used_files = processed  # 解包处理结果

    # 保存为压缩的Numpy二进制文件
    np.savez(save_path,
             x=x,  # 图像数据矩阵
             y=y,  # 用户标签数组
             yforg=yforg,  # 伪造标签标识
             user_mapping=user_mapping,  # 用户ID映射表
             filenames=used_files)  # 使用的文件路径列表


if __name__ == '__main__':
    # 配置命令行参数解析器
    parser = argparse.ArgumentParser(description='签名数据集预处理工具')

    # 定义必需参数
    parser.add_argument('--dataset',
                        choices=available_datasets.keys(),  # 从可用数据集中选择
                        required=True,
                        help='数据集类型标识符')
    parser.add_argument('--path',
                        required=True,
                        help='原始数据存储目录路径')
    parser.add_argument('--save-path',
                        required=True,
                        help='预处理数据输出路径')
    parser.add_argument('--image-size',
                        nargs=2,
                        type=int,
                        default=(170, 242),  # 默认尺寸(高170px, 宽242px)
                        help='目标图像尺寸(高度 宽度)')

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化指定数据集对象
    ds = available_datasets[args.dataset]
    dataset = ds(args.path)

    # 执行预处理流程
    print('启动数据集预处理')
    process_dataset(dataset, args.save_path, args.image_size)