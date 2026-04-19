import matplotlib.pyplot as plt
import numpy as np

# =========================================
# 1) 表格中的“无”条件数据
# =========================================
datasets = ["Wildtrack", "MultiviewX"]
methods = ["Local Only", "Logits", "Feature Map", "Feature Map + UCB(loss)"]

data = {
    "MODA": {
        "Local Only": [32.3, 32.9],
        "Logits": [78.5, 52.2],
        "Feature Map": [90.1, 93.6],
        "Feature Map + UCB(loss)": [71.2, 72.3],
    },
    "Precision": {
        "Local Only": [95.3, 87.9],
        "Logits": [86.6, 70.5],
        "Feature Map": [93.3, 97.9],
        "Feature Map + UCB(loss)": [94.6, 93.6],
    },
    "Recall": {
        "Local Only": [33.9, 38.4],
        "Logits": [92.9, 89.8],
        "Feature Map": [97.1, 95.6],
        "Feature Map + UCB(loss)": [75.5, 77.6],
    },
    "F1": {
        "Local Only": [48.0, 52.8],
        "Logits": [89.6, 79.0],
        "Feature Map": [95.2, 96.8],
        "Feature Map + UCB(loss)": [83.9, 84.9],
    },
}

# =========================================
# 2) 画图风格
# =========================================
colors = {
    "Local Only": "#1f3a93",
    "Logits": "#1f77b4",
    "Feature Map": "#d62728",
    "Feature Map + UCB(loss)": "#2ca02c",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

x = np.arange(len(datasets))
bar_width = 0.18

# =========================================
# 3) 通用绘图函数
# =========================================
def plot_grouped_bars(ax, metric_name):
    metric_data = data[metric_name]

    for i, method in enumerate(methods):
        offset = (i - 1.5) * bar_width
        values = metric_data[method]

        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=method,
            color=colors[method],
            alpha=0.9
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    ax.set_title(metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(f"{metric_name} (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

# =========================================
# 4) 2x2 子图：论文风格顺序
#    左上 MODA，右上 Precision
#    左下 Recall，右下 F1
# =========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

plot_grouped_bars(axes[0, 0], "MODA")
plot_grouped_bars(axes[0, 1], "Precision")
plot_grouped_bars(axes[1, 0], "Recall")
plot_grouped_bars(axes[1, 1], "F1")

# 总图例
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03)
)

fig.suptitle("Method Comparison on Wildtrack and MultiviewX (No Packet Loss)", fontsize=17)

plt.savefig("grouped_bar_2x2_metrics_paper_style.png", dpi=300, bbox_inches="tight")
plt.show()