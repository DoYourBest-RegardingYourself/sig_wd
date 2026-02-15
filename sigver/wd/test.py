import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn

from sigver.featurelearning.data import extract_features
import sigver.featurelearning.models as models
import argparse
from sigver.datasets.util import load_dataset, get_subset
import sigver.wd.training as training  # 导入训练和测试相关函数
import numpy as np
import pickle

from sigver.featurelearning.models.signet import BaseHeadSplit, SigNet, SigNet_smaller


def main(args):
    # 定义实验用户和开发用户的范围
    exp_users = range(*args.exp_users)
    dev_users = range(*args.dev_users)

    # 加载预训练模型参数和权重
    state_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')
    print('使用设备: {}'.format(device))

    # 根据权重文件的键判断使用哪个模型类
    # SigNet 有 conv_layers.conv4 和 fc_layers.fc2，而 SigNet_smaller 没有
    has_conv4 = any('conv_layers.conv4' in key for key in state_dict.keys())
    has_fc2 = any('fc_layers.fc2' in key for key in state_dict.keys())
    
    if has_conv4 and has_fc2:
        print('检测到 SigNet 权重文件')
        model_class = SigNet
    else:
        print('检测到 SigNet_smaller 权重文件')
        model_class = SigNet_smaller
    
    # 初始化模型并加载预训练参数
    base_model = model_class().to(device).eval()
    head = nn.Linear(2048, 700).to(device)
    model = BaseHeadSplit(base_model, head)

    model_dict = model.state_dict()

    # 过滤出匹配的键
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 更新当前模型的state_dict
    model_dict.update(pretrained_dict)

    # 加载更新后的state_dict
    model.load_state_dict(model_dict)
    # model.load_state_dict(state_dict)
    base_model = model.base

    # ============================== 从这开始
    # 定义处理函数：将输入数据送入模型并返回特征
    def process_fn(batch):
        input = batch[0].to(device)
        return base_model(input)  # 提取特征

    # 加载数据集（签名图像、标签、伪造信息等）
    """
    预处理数据集的路径，必须是包含以下键值的.npz文件：
        - 'x'    : 签名图像数据，形状为(N, 1, H, W)的float32数组
        - 'y'    : 用户标签，形状为(N,)的int32数组
        - 'yforg': 伪造标签，形状为(N,)的int32数组（0=真实，1=伪造）
        - 'usermapping' : 用户ID映射字典（将数据集中原始用户ID映射为连续整数）
        - 'filenames'   : 文件名数组，形状为(N,)的字符串数组
    """
    x, y, yforg, user_mapping, filenames = load_dataset(args.data_path)

    # 提取特征：使用预训练模型处理图像数据
    """
        features : np.ndarray
            特征矩阵，形状为(N, D)，其中D是特征维度
            就是倒数第一层的输出(即分类头的前一层的合并)
            return torch.cat(result, dim=0).cpu().numpy()
    """
    features = extract_features(x, process_fn, args.batch_size, args.input_size)
    data = (features, y, yforg)  # 组合特征和标签

    # 划分实验集和开发集
    exp_set = get_subset(data, exp_users)  # 实验集数据（用于训练）
    dev_set = get_subset(data, dev_users)  # 开发集数据（用于测试）

    rng = np.random.RandomState(1234)  # 固定随机种子以确保可重复性

    # 初始化结果存储
    eer_u_list = []  # 用户特定阈值EER
    eer_list = []  # 全局阈值EER
    all_results = []  # 所有交叉验证结果

    # 进行多次交叉验证（默认为10次）
    for _ in range(args.folds):
        # 训练并测试所有用户，获取分类器和结果
        classifiers, results = training.train_test_all_users(
            exp_set, dev_set,  # 实验集和开发集数据
            svm_type=args.svm_type,  # SVM核类型（RBF或线性）
            C=args.svm_c,  # SVM正则化参数
            gamma=args.svm_gamma,  # RBF核的gamma参数
            num_gen_train=args.gen_for_train,  # 训练使用的真实签名数量
            num_forg_from_exp=args.forg_from_exp,  # 来自实验集的伪造签名数量
            num_forg_from_dev=args.forg_from_dev,  # 来自开发集的伪造签名数量
            num_gen_test=args.gen_for_test,  # 测试使用的真实签名数量
            rng=rng)

        # 记录当前交叉验证的EER结果
        this_eer_u = results['all_metrics']['EER_userthresholds']  # 用户特定阈值
        this_eer = results['all_metrics']['EER']  # 全局阈值
        all_results.append(results)
        eer_u_list.append(this_eer_u)
        eer_list.append(this_eer)

    # 输出平均EER及标准差
    print('全局阈值EER: {:.2f} (±{:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
    print('用户阈值EER: {:.2f} (±{:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))

    # 保存结果到指定路径（Pickle格式）
    if args.save_path is not None:
        print('保存结果至 {}'.format(args.save_path))
        with open(args.save_path, 'wb') as f:
            pickle.dump(all_results, f)
    return all_results


if __name__ == '__main__':
    # 参数解析器配置
    parser = argparse.ArgumentParser(description='签名验证系统性能评估脚本')

    # 模型相关参数
    parser.add_argument('-m', choices=models.available_models,
                        help='模型架构名称', dest='model', default='signet')
    parser.add_argument('--model-path', help='预训练模型路径')
    parser.add_argument('--data-path', help='数据集路径')
    parser.add_argument('--save-path', help='结果保存路径', default='../../result/results.pkl')
    parser.add_argument('--input-size', nargs=2, default=(224, 224),
                        help='输入图像尺寸（高度, 宽度）')

    # 数据集划分参数
    parser.add_argument('--exp-users', type=int, nargs=2, default=(0, 300),
                        help='实验集用户ID范围（起始, 结束）')
    parser.add_argument('--dev-users', type=int, nargs=2, default=(0, 300),
                        help='开发集用户ID范围（起始, 结束）')

    # 训练/测试数据配置
    parser.add_argument('--gen-for-train', type=int, default=12,
                        help='训练时每用户使用的真实签名数量')
    parser.add_argument('--gen-for-test', type=int, default=10,
                        help='测试时每用户使用的真实签名数量')
    parser.add_argument('--forg-from-exp', type=int, default=12,
                        help='从实验集选取的伪造签名数量')
    parser.add_argument('--forg-from-dev', type=int, default=0,
                        help='从开发集选取的伪造签名数量')

    # SVM分类器参数
    parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf',
                        help='SVM核类型')
    parser.add_argument('--svm-c', type=float, default=1,
                        help='SVM正则化参数C')
    parser.add_argument('--svm-gamma', type=float, default=2 ** -11,
                        help='RBF核的gamma参数（2^-11 by default）')

    # 硬件配置
    parser.add_argument('--gpu-idx', type=int, default=0,
                        help='使用的GPU索引')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='特征提取时的批次大小')

    # 实验设置
    parser.add_argument('--folds', type=int, default=5,
                        help='交叉验证次数')

    arguments = parser.parse_args()
    
    # 定义4个子目录和5个客户端
    subdirectories = ['DA_DH', 'all', 'non', 'only_da']
    num_clients = 5
    base_path = r'D:\FedCTF_result\消融实验\DA\5c'
    
    for subdir in subdirectories:
        for client_id in range(num_clients):
            arguments.model_path = f'{base_path}\\{subdir}\\{client_id}GPDS_FedCTF.pth'
            arguments.save_path = f'../../result/{subdir}_client{client_id}_results.pkl'
            arguments.data_path = '../../data/gpds.npz'
            print(f'\n{"="*60}')
            print(f'测试子目录: {subdir}, 客户端: {client_id}')
            print(f'模型路径: {arguments.model_path}')
            print(f'{"="*60}')
            print(arguments)
            main(arguments)
