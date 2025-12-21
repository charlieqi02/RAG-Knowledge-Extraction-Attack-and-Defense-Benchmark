import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ================== 基本设置 ==================
METHODS = ["randtoken", "randemb", "randtext", "dgea", "ikea", "copybreak"]
DATASETS = ["HealthCareMagic", "Enron", "HarryPotter", "Pokemon"]

# 原始实验 log 路径（用于取 RD）
BASE_LOG = "./logs/"
experiment_dict = {
    "HealthCareMagic": {
        "randtoken": "11_08/TextRAG-HealthCareMagic/RandomToken-None-03_28_03",
        "randemb":   "11_08/TextRAG-HealthCareMagic/RandomEmb-None-01_26_09",
        "randtext":  "11_06/TextRAG-HealthCareMagic/RandomText-None-12_41_17",
        "dgea":      "11_07/TextRAG-HealthCareMagic/DGEA-None-22_51_12",
        "ikea":      "11_12/TextRAG-HealthCareMagic/IKEA-None-07_33_32",
        "copybreak": "11_08/TextRAG-HealthCareMagic/CopyBreak-None-05_05_21",
    },
    "Enron": {
        "randtoken": "11_06/TextRAG-Enron/RandomToken-None-05_06_38",
        "randemb":   "11_06/TextRAG-Enron/RandomEmb-None-01_05_44",
        "randtext":  "11_05/TextRAG-Enron/RandomText-None-16_04_06",
        "dgea":      "11_05/TextRAG-Enron/DGEA-None-18_43_59",
        "ikea":      "11_11/TextRAG-Enron/IKEA-None-18_57_12",
        "copybreak": "11_08/TextRAG-Enron/CopyBreak-None-05_04_53",
    },
    "HarryPotter": {
        "randtoken": "11_08/TextRAG-HarryPotter/RandomToken-None-03_21_22",
        "randemb":   "11_08/TextRAG-HarryPotter/RandomEmb-None-01_37_23",
        "randtext":  "11_07/TextRAG-HarryPotter/RandomText-None-22_51_12",
        "dgea":      "11_07/TextRAG-HarryPotter/DGEA-None-23_08_38",
        "ikea":      "11_11/TextRAG-HarryPotter/IKEA-None-21_18_53",
        "copybreak": "11_08/TextRAG-HarryPotter/CopyBreak-None-05_06_10",
    },
    "Pokemon": {
        "randtoken": "11_08/TextRAG-Pokemon/RandomToken-None-03_06_05",
        "randemb":   "11_08/TextRAG-Pokemon/RandomEmb-None-01_26_09",
        "randtext":  "11_07/TextRAG-Pokemon/RandomText-None-22_51_56",
        "dgea":      "11_07/TextRAG-Pokemon/DGEA-None-23_01_43",
        "ikea":      "11_11/TextRAG-Pokemon/IKEA-None-19_46_58",
        "copybreak": "11_08/TextRAG-Pokemon/CopyBreak-None-05_04_04",
    },
}

# query-diversity 实验的 log 路径（用于取 RD）
BASE_QD_LOG = "../logs/query_diversity/"
experiment_qd_dict = {
    "HealthCareMagic": {
        "dgea":      "11_21/TextRAG-HealthCareMagic/DGEA-None-14_37_53",
        "ikea":      "11_20/TextRAG-HealthCareMagic/IKEA-None-06_49_51",
        "copybreak": "11_20/TextRAG-HealthCareMagic/CopyBreak-None-02_58_30"
    },
    "Enron": {
        "dgea":      "11_21/TextRAG-Enron/DGEA-None-04_04_20",
        "ikea":      "11_20/TextRAG-Enron/IKEA-None-10_39_35",
        "copybreak": "11_21/TextRAG-Enron/CopyBreak-None-09_46_41"
    },
    "HarryPotter": {
        "dgea":      "11_21/TextRAG-HarryPotter/DGEA-None-12_07_57",
        "ikea":      "11_20/TextRAG-HarryPotter/IKEA-None-14_46_36",
        "copybreak": "11_20/TextRAG-HarryPotter/CopyBreak-None-02_08_30"
    },
    "Pokemon": {
        "dgea":      "11_21/TextRAG-Pokemon/DGEA-None-17_10_29",
        "ikea":      "11_21/TextRAG-Pokemon/IKEA-None-19_33_03",
        "copybreak": "11_20/TextRAG-Pokemon/CopyBreak-None-04_37_53"
    }
}

# RD 结果
results = pd.read_csv("./logs/results.csv")
results = results.rename(columns={"RD_ROUGR_L": "EM"})  # 你原来的 typo 修复


# ================== 路径工具 ==================
def get_base_paths(dim_red: str, variant: str):
    """
    dim_red: 'tsne' / 'umap' / 'pca' / 'trimap'
    variant: 'origin' / 'query-diversity'
    """
    base_root = os.path.join("./logs", dim_red, variant)
    split_root = os.path.join(base_root, "split")
    together_root = os.path.join(base_root, "together")
    fig_root = os.path.join(base_root, "figs")
    return split_root, together_root, fig_root


# ================== 辅助：拿 RD ==================
def get_rd_value(dataset: str, method: str, variant: str) -> str:
    """
    根据 dataset, method, variant 找到对应的 RD 数值（字符串），
    若找不到就返回 'NA'，避免直接报错。
    """
    # 选择对应的 log path dict
    if variant == "origin" or method in ["randtoken", "randemb", "randtext"]:
        path_dict = experiment_dict
    else:
        # query-diversity 版本只对 dgea/ikea/copybreak 有
        path_dict = experiment_qd_dict

    try:
        rel_path = path_dict[dataset][method]
    except KeyError:
        return "NA"

    model_id = rel_path.split("/")[-1]
    row = results[(results["model"] == model_id) & (results["dataset"] == dataset)]
    if len(row) == 0:
        return "NA"

    rd_val = row["RD"].values[0]
    try:
        rd_str = f"{float(rd_val):.3f}"
    except Exception:
        rd_str = str(rd_val)
    return rd_str


# ================== 画图函数 ==================
def plot_panel(ax, trio_red, method, dataset, variant, cmap=plt.cm.viridis):
    main_red, dataset_red, attack_red = trio_red

    # 先画 KB embedding：点小、透明度高
    ax.scatter(
        dataset_red[:, 0], dataset_red[:, 1],
        s=6, alpha=0.15, linewidths=0, label=dataset
    )

    # 再画对应(200, 2)的数据：按索引做渐变（第一个点颜色最浅，最后一个最深）
    n = main_red.shape[0]
    idx = np.arange(n)
    norm = Normalize(vmin=0, vmax=n - 1)
    ax.scatter(
        main_red[:, 0], main_red[:, 1],
        c=idx, cmap=cmap, norm=norm, marker='^',
        s=20, alpha=0.9, linewidths=0
    )

    # 画 attack template（单点）
    ax.scatter(
        attack_red[0, 0], attack_red[0, 1],
        c='red', s=50, marker='*', label='Attack Template'
    )

    # 图例：放大图例点，并降低图例点透明度
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, markerscale=2)
    for attr in ("legend_handles", "legendHandles"):
        if hasattr(leg, attr):
            for h in getattr(leg, attr):
                try:
                    h.set_alpha(0.4)
                except Exception:
                    pass
            break

    # 标题里加 RD
    rd_str = get_rd_value(dataset, method, variant)
    title = f"{method}  RD: {rd_str}"
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


# ================== 可视化：split ==================
def visualize_split(dim_red: str, variant: str, dataset: str):
    """
    读取 ./logs/{dim_red}/{variant}/split/{dataset}/{method}.pkl
    每个 .pkl: (main_red, dataset_red, attack_red)
    """
    split_root, _, fig_root = get_base_paths(dim_red, variant)
    tsne_dir = os.path.join(split_root, dataset)

    os.makedirs(os.path.join(fig_root, "split"), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

    for ax, method in zip(axes.ravel(), METHODS):
        pkl_path = os.path.join(tsne_dir, f"{method}.pkl")
        with open(pkl_path, "rb") as f:
            trio_red = pickle.load(f)  # (main_red, dataset_red, attack_red)
        plot_panel(ax, trio_red, method, dataset, variant, cmap=plt.cm.viridis)

    out_path = os.path.join(fig_root, "split", f"{dataset}.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[{dim_red.upper()}-{variant} SPLIT FIG SAVED] {out_path}")


# ================== 可视化：together ==================
def visualize_together(dim_red: str, variant: str, dataset: str):
    """
    读取 ./logs/{dim_red}/{variant}/together/{dataset}.pkl
    字典结构类似：
        {
            "randtoken": np.ndarray (N1, 2),
            "randemb":   ...,
            "randtext":  ...,
            "dgea":      ...,
            "ikea":      ...,
            "copybreak": ...,
            "dataset":   np.ndarray (Nk, 2),
            "attack_template": np.ndarray (1, 2),
        }
    然后对每个 method 重新组装成 (main_red, dataset_red, attack_red)
    再调用 plot_panel。
    """
    _, together_root, fig_root = get_base_paths(dim_red, variant)
    together_path = os.path.join(together_root, f"{dataset}.pkl")
    with open(together_path, "rb") as f:
        data = pickle.load(f)  # dict

    os.makedirs(os.path.join(fig_root, "together"), exist_ok=True)

    dataset_red = data["dataset"]
    attack_red = data["attack_template"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

    for ax, method in zip(axes.ravel(), METHODS):
        main_key = method
        if main_key not in data:
            raise KeyError(f"Key '{main_key}' not found in together {dim_red} dict for dataset={dataset}")
        main_red = data[main_key]

        trio_red = (main_red, dataset_red, attack_red)
        plot_panel(ax, trio_red, method, dataset, variant, cmap=plt.cm.viridis)

    out_path = os.path.join(fig_root, "together", f"{dataset}.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[{dim_red.upper()}-{variant} TOGETHER FIG SAVED] {out_path}")


# ================== 主函数 ==================
def main():
    dim_red_methods = ["tsne", "umap", "pca", "trimap"]
    variants = ["origin", "query-diversity"]

    for dim_red in dim_red_methods:
        for variant in variants:
            print(f"\n==== VISUALIZING: {dim_red.upper()} / {variant} ====")
            for d in DATASETS:
                print(f"=== {dim_red.upper()} {variant} SPLIT: {d} ===")
                visualize_split(dim_red, variant, d)

                print(f"=== {dim_red.upper()} {variant} TOGETHER: {d} ===")
                visualize_together(dim_red, variant, d)


if __name__ == "__main__":
    main()
