import os

import numpy as np
import functools
from tqdm import tqdm
from sigver.datasets.base import IterableDataset
from sigver.preprocessing.normalize import preprocess_signature
from typing import Tuple, Callable, Dict, Union


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray]:
    """
    加载预处理的签名数据集（.npz格式）

    参数:
    ----------
    path : str
        预处理数据集的路径，必须是包含以下键值的.npz文件：
        - 'x'    : 签名图像数据，形状为(N, 1, H, W)的float32数组
        - 'y'    : 用户标签，形状为(N,)的int32数组
        - 'yforg': 伪造标签，形状为(N,)的int32数组（0=真实，1=伪造）
        - 'usermapping' : 用户ID映射字典（将数据集中原始用户ID映射为连续整数）
        - 'filenames'   : 文件名数组，形状为(N,)的字符串数组

    返回:
    -------
    x : np.ndarray
        签名图像数据，形状为(N, 1, H, W)的float32数组，数值范围通常为[0,1]或[-1,1]
        N: 样本数量，1: 灰度通道，H: 图像高度，W: 图像宽度

    y : np.ndarray
        用户标签数组，形状为(N,)，元素为连续整数（从0开始）
        例如：y[0] = 5 表示第0个样本属于用户5

    yforg : np.ndarray
        伪造标签数组，形状为(N,)，元素取值：
        0 - 真实签名
        1 - 伪造签名
        例如：yforg[1] = 1 表示第1个样本是伪造签名

    user_mapping : Dict
        用户ID映射字典，格式为 {数据集中用户索引: 原始用户ID}
        例如：{0: "user_001", 1: "user_002"} 表示：
        - 数据集中索引0对应原始用户"user_001"
        - 数据集中索引1对应原始用户"user_002"

    filenames : np.ndarray
        原始文件名数组，形状为(N,)，元素为字符串类型
        例如：["user001_001.png", "user001_002.png"...]

    异常:
    ------
    FileNotFoundError: 当指定路径不存在时抛出
    KeyError: 当.npz文件缺少必要键值时抛出
    ValueError: 当数据格式不符合预期时抛出

    示例:
    ------
    >>> x, y, yforg, mapping, files = load_dataset("data/signatures.npz")
    >>> print(f"加载 {x.shape[0]} 个样本")
    >>> print(f"用户数量: {len(mapping)}")
    """
    # 使用上下文管理器安全加载.npz文件
    # allow_pickle=True 允许加载包含Python对象的数据
    # 注意：加载不可信来源文件可能存在安全风险
    with np.load(path, allow_pickle=True) as data:
        # 提取图像数据并验证维度
        # 预期形状：(N, 1, H, W)，其中：
        # N - 样本数量，1 - 单通道，H - 高度，W - 宽度
        x = data['x']
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("图像数据维度异常，预期形状：(N, 1, H, W)")

        # 提取用户标签并验证范围
        y = data['y'].astype(np.int32)
        unique_users = np.unique(y)
        if not (unique_users == np.arange(len(unique_users))).all():
            raise ValueError("用户标签应为连续整数，从0开始")

        # 提取伪造标签并验证取值范围
        yforg = data['yforg'].astype(np.int32)
        if not np.all(np.isin(yforg, [0, 1])):
            raise ValueError("伪造标签只能包含0或1")

        # 提取用户映射字典（可能包含原始用户ID）
        user_mapping = data['user_mapping'].item()  # .item()将numpy对象转为Python字典
        if not isinstance(user_mapping, dict):
            raise TypeError("用户映射应为字典类型")

        # 提取文件名数组
        filenames = data['filenames']

    return x, y, yforg, user_mapping, filenames


def process_dataset(dataset: IterableDataset,
                    save_path: str,
                    img_size: Tuple[int, int],
                    subset: slice = slice(None)):
    """ Processes a dataset (normalizing the images) and saves the result as a
        numpy npz file (collection of np.ndarrays).

    Parameters
    ----------
    dataset : IterableDataset
        The dataset, that knows where the signature files are located
    save_path : str
        The name of the file to save the numpy arrays
    img_size : tuple (H x W)
        The final size of the images
    subset : slice
        Which users to consider. e.g. slice(None) to consider all users, or slice(first, last)


    Returns
    -------
    None

    """
    preprocess_fn = functools.partial(preprocess_signature,
                                      canvas_size=dataset.maxsize,
                                      img_size=img_size,
                                      input_size=img_size)

    processed = process_dataset_images(dataset, preprocess_fn, img_size, subset)
    x, y, yforg, user_mapping, used_files = processed

    np.savez(save_path,
             x=x,
             y=y,
             yforg=yforg,
             user_mapping=user_mapping,
             filenames=used_files)
    return x, y, yforg, user_mapping, used_files


def process_dataset_images(dataset: IterableDataset,
                           preprocess_fn: Callable[[np.ndarray], np.ndarray],
                           img_size: Tuple[int, int],
                           subset: slice) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    处理签名数据集图像，生成标准化的多维数组及标签

    参数说明：
    dataset       : 数据集对象，需实现用户列表和签名迭代接口
    preprocess_fn : 图像预处理函数（输入输出均为numpy数组）
    img_size      : 目标图像尺寸（高度，宽度）
    subset        : 用户范围切片（如slice(0,100)处理前100个用户）

    返回值：
    x          : 图像数据（N x 1 x H x W）的uint8数组
    y          : 用户ID标签数组（0~num_users-1）
    yforg      : 伪造类型标签（0=真实，1=熟练伪造，2=简单伪造）
    usermapping: 内部用户ID到原始用户标识的映射字典
    filenames  : 对应签名的文件名数组
    """

    # 初始化用户映射字典（内部ID -> 原始用户标识）
    user_mapping = {}

    # 获取并筛选目标用户列表
    users = dataset.get_user_list()
    users = users[subset]  # 应用用户切片
    print(f'处理用户数: {len(users)}')

    # 图像维度解包
    H, W = img_size

    # 计算预分配数组的最大容量（基于用户数×每用户最大签名数）
    max_signatures = len(users) * (
            dataset.genuine_per_user +
            dataset.skilled_per_user +
            dataset.simple_per_user
    )

    # 预分配内存（避免动态扩容的内存峰值）
    x = np.empty((max_signatures, H, W), dtype=np.uint8)  # 图像数据
    y = np.empty(max_signatures, dtype=np.int32)  # 用户ID标签
    yforg = np.empty(max_signatures, dtype=np.int32)  # 伪造类型标签
    used_files = []  # 文件名缓存

    print(f'预分配数组尺寸: {x.shape}')

    # 签名计数器
    N = 0

    # 遍历每个用户（显示进度条）
    for i, user in enumerate(tqdm(users, desc="处理用户")):
        # 记录用户映射关系
        user_mapping[i] = user  # 内部i对应原始user标识

        # 处理真实签名 -----------------------------------------------
        # 生成（预处理图像, 文件名）元组列表
        user_gen_data = [
            (preprocess_fn(img), filename)
            for (img, filename) in dataset.iter_genuine(user)
        ]

        # 解包为图像列表和文件名列表
        if len(user_gen_data) > 0:
            gen_imgs, gen_filenames = zip(*user_gen_data)
            new_img_count = len(gen_imgs)

            # 计算写入位置
            indexes = slice(N, N + new_img_count)

            # 填充数据
            x[indexes] = gen_imgs  # 图像数据
            y[indexes] = i  # 当前用户ID
            yforg[indexes] = 0  # 真实签名标签
            used_files.extend(gen_filenames)  # 记录文件名

            N += new_img_count  # 更新计数器

        # 处理熟练伪造签名 -------------------------------------------
        user_forg_data = [
            (preprocess_fn(img), filename)
            for (img, filename) in dataset.iter_forgery(user)
        ]

        if len(user_forg_data) > 0:
            forg_imgs, forg_filenames = zip(*user_forg_data)
            new_img_count = len(forg_imgs)

            indexes = slice(N, N + new_img_count)
            x[indexes] = forg_imgs
            y[indexes] = i
            yforg[indexes] = 1  # 熟练伪造标签
            used_files.extend(forg_filenames)
            N += new_img_count

        # 处理简单伪造签名 -------------------------------------------
        user_forg_data = [
            (preprocess_fn(img), filename)
            for (img, filename) in dataset.iter_simple_forgery(user)
        ]

        if len(user_forg_data) > 0:
            forg_imgs, forg_filenames = zip(*user_forg_data)
            new_img_count = len(forg_imgs)

            indexes = slice(N, N + new_img_count)
            x[indexes] = forg_imgs
            y[indexes] = i
            yforg[indexes] = 2  # 简单伪造标签
            used_files.extend(forg_filenames)
            N += new_img_count

    # 最终数组调整 -------------------------------------------------
    if N != max_signatures:
        # 当实际数据量小于预分配空间时，收缩数组
        # refcheck=False允许修改数组内存布局
        x.resize((N, 1, H, W), refcheck=False)
        y.resize(N, refcheck=False)
        yforg.resize(N, refcheck=False)
    else:
        # 添加通道维度（符合PyTorch等框架的输入格式）
        x = np.expand_dims(x, axis=1)

    # 转换文件名列表为numpy数组
    used_files = np.array(used_files)

    return x, y, yforg, user_mapping, used_files


def get_subset(data: Tuple[np.ndarray, ...],
               subset: Union[list, range],
               y_idx: int = 1) -> Tuple[np.ndarray, ...]:
    """ Gets a data for a subset of users (the second array in data)

    Parameters
    ----------
    data: Tuple (x, y, ...)
        The dataset
    subset: list
        The list of users to include
    y_idx: int
        The index in data that refers to the users (usually index=1)

    Returns
    -------
    Tuple (x, y , ...)
        The dataset containing only data from users in the subset

    """
    to_include = np.isin(data[y_idx], subset)

    return tuple(d[to_include] for d in data)


def remove_forgeries(data: Tuple[np.ndarray, ...],
                     forg_idx: int = 2) -> Tuple[np.ndarray, ...]:
    """ Remove the forgeries from a dataset

    Parameters
    ----------
    data: Tuple (x, y, yforg)
        The dataset
    forg_idx: int
        The index in data that refers to wheter the signature is a forgery (usually
        we pass (x, y, yforg) so forg_idx=2)

    Returns
    -------
    Tuple (x, y, yforg)
        The dataset with only genuine signatures

    """
    to_include = (data[forg_idx] == False)

    return tuple(d[to_include] for d in data)
