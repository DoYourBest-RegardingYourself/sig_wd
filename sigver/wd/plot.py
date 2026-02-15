import numpy as np
import torch
from torch import nn

from sigver.featurelearning.data import extract_features
import sigver.featurelearning.models as models
import argparse
from sigver.datasets.util import load_dataset, get_subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sigver.featurelearning.models.signet import BaseHeadSplit

# 设置全局字体为新罗马字体
plt.rcParams['font.family'] = 'Times New Roman'


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# 预定义5个客户端的配色方案（浅色用于伪造，深色用于真实）
client_colors = [
    ('#9ECAE1', '#2171B5'),  # 客户端1
    ('#A1D99B', '#238B45'),  # 客户端2
    ('#FC9272', '#CB181D'),  # 客户端3
    ('#C994C7', '#807DBA'),  # 客户端4
    ('#FDBF6F', '#E6550D')  # 客户端5
]


def main(args):
    exp_users = range(*args.exp_users)

    # 加载数据集
    x, y, yforg, _, _ = load_dataset(args.data_path)
    data = (x, y, yforg)
    exp_set = get_subset(data, exp_users)
    x_exp, y_exp, yforg_exp = exp_set[0], exp_set[1], exp_set[2]

    # 客户端模型路径
    client_model_paths = [
        "../../models/1Bengali_FedCTF.pt",
        "../../models/2Bengali_FedCTF.pt",
        "../../models/3Bengali_FedCTF.pt",
        "../../models/4Bengali_FedCTF.pt",
    ]

    device = get_device()
    print(f'Using device: {device}')

    # 存储所有客户端特征
    all_client_features = []

    for model_path in client_model_paths:
        # 加载模型
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        base_model = models.available_models[args.model]().to(device).eval()
        head = nn.Linear(2048, 700).to(device)
        model = BaseHeadSplit(base_model, head)
        model.load_state_dict(state_dict)

        # 特征提取（添加归一化）
        def process_fn(batch):
            input = (batch[0].to(device).float() / 255.0)  # 归一化
            return base_model(input.detach())

        features = extract_features(x_exp, process_fn, args.batch_size, args.input_size)
        all_client_features.append(features)

    # 合并所有客户端特征 (5*N, 2048)
    all_features = np.concatenate(all_client_features, axis=0)

    # 生成标签 (5*N, )
    num_samples = len(yforg_exp)
    client_labels = np.repeat(np.arange(5), num_samples)  # 客户端标识
    forgery_labels = np.tile(yforg_exp, 5)  # 真实/伪造标识

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    # 可视化
    plt.figure(figsize=(15, 10))

    # 绘制每个客户端的特征
    for client_idx in range(5):
        light_color, dark_color = client_colors[client_idx]

        # 获取当前客户端的数据
        mask = (client_labels == client_idx)
        client_points = features_2d[mask]
        client_forgery = forgery_labels[mask]

        # 绘制真实样本（深色 + 圆形），增大特征点大小
        real_mask = (client_forgery == 0)
        plt.scatter(client_points[real_mask, 0], client_points[real_mask, 1],
                    c=dark_color, s=50, alpha=0.7, edgecolors='w', linewidth=0.5,
                    label=f'Client {client_idx + 1} Real')

        # 绘制伪造样本（浅色 + 叉号），增大特征点大小
        fake_mask = (client_forgery == 1)
        plt.scatter(client_points[fake_mask, 0], client_points[fake_mask, 1],
                    c=light_color, s=50, alpha=0.7, marker='x', linewidth=0.8,
                    label=f'Client {client_idx + 1} Fake')

    # 图例和样式
    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left', ncol=2,
               prop={'size': 20}, markerscale=2)  # 调整bbox_to_anchor给图例留出更多空间
    plt.axis('off')
    plt.tight_layout()

    plt.savefig("FedAvg_Combined.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices=models.available_models, default="signet", dest='model')
    parser.add_argument('--data-path', default="../../data/Bengali.npz")
    parser.add_argument('--input-size', nargs=2, default=(170, 242))
    parser.add_argument('--exp-users', type=int, nargs=2, default=(0, 55))
    parser.add_argument('--batch-size', type=int, default=32)
    arguments = parser.parse_args()

    main(arguments)
