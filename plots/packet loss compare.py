import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1) 数据
# =========================
packet_loss = np.array([0.0, 0.1, 0.3, 0.5, 0.7])

data = {
    "Wildtrack": {
        "F1": {
            "Feature Map": [95.2, 94.7, 93.6, 89.6, 82.9],
            "Logits": [89.6, 89.3, 87.3, 84.2, 74.1],
            "Feature Map + UCB(loss)": [83.9, 82.1, 78.9, 74.1, 66.7],
            "Local Only": [48.0, 48.0, 48.0, 48.0, 48.0],
        },
        "MODA": {
            "Feature Map": [90.1, 89.3, 87.3, 80.3, 69.7],
            "Logits": [78.5, 78.2, 75.2, 70.5, 57.1],
            "Feature Map + UCB(loss)": [71.2, 68.6, 64.3, 58.1, 49.3],
            "Local Only": [32.3, 32.3, 32.3, 32.3, 32.3],
        },
    },
    "MultiviewX": {
        "F1": {
            "Feature Map": [96.8, 95.8, 93.0, 87.5, 78.0],
            "Logits": [79.0, 78.9, 78.9, 76.4, 72.6],
            "Feature Map + UCB(loss)": [84.9, 83.1, 79.7, 74.5, 66.8],
            "Local Only": [52.8, 52.8, 52.8, 52.8, 52.8],
        },
        "MODA": {
            "Feature Map": [93.6, 91.8, 86.4, 76.6, 61.9],
            "Logits": [52.2, 52.9, 55.2, 53.5, 50.4],
            "Feature Map + UCB(loss)": [72.3, 69.4, 64.4, 57.2, 47.3],
            "Local Only": [32.9, 32.9, 32.9, 32.9, 32.9],
        },
    },
}

# =========================
# 2) 画图风格
# =========================
style_map = {
    "Feature Map": {
        "color": "#d62728",   # 红
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.6,
    },
    "Logits": {
        "color": "#1f77b4",   # 蓝
        "marker": "s",
        "linestyle": "-",
        "linewidth": 2.6,
    },
    "Feature Map + UCB(loss)": {
        "color": "#2ca02c",   # 绿
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.6,
    },
    "Local Only": {
        "color": "#1f3a93",   # 深蓝
        "marker": None,
        "linestyle": "--",
        "linewidth": 2.6,
    },
}

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# =========================
# 3) 画单个子图的函数
# =========================
def plot_panel(ax, dataset_name, metric_name):
    methods = data[dataset_name][metric_name]

    for method_name, y in methods.items():
        style = style_map[method_name]
        ax.plot(
            packet_loss,
            y,
            label=method_name,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=7,
        )

    ax.set_title(f"{dataset_name} - {metric_name}")
    ax.set_xlabel("Packet Loss Rate")
    ax.set_ylabel(f"{metric_name} (%)")
    ax.set_xticks(packet_loss)
    ax.set_xticklabels([f"{x:.1f}" for x in packet_loss])
    ax.grid(True, alpha=0.3)

    # 根据不同指标自动设置纵轴范围，让图更紧凑
    all_values = []
    for y in methods.values():
        all_values.extend(y)
    y_min = max(0, min(all_values) - 5)
    y_max = min(100, max(all_values) + 3)
    ax.set_ylim(y_min, y_max)

# =========================
# 4) 生成 2x2 子图
#    第一行：F1
#    第二行：MODA
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

plot_panel(axes[0, 0], "Wildtrack", "F1")
plot_panel(axes[0, 1], "MultiviewX", "F1")
plot_panel(axes[1, 0], "Wildtrack", "MODA")
plot_panel(axes[1, 1], "MultiviewX", "MODA")

# 统一图例
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03),
)

fig.suptitle("Robustness under Packet Loss", fontsize=18, y=1.08)

# 保存图片
plt.savefig("packet_loss_multi_method_curves.png", dpi=300, bbox_inches="tight")
plt.show()