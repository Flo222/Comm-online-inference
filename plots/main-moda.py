import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1) 数据
# =========================
datasets = ["Wildtrack", "MultiviewX"]
methods = ["Local Only", "Logits", "Feature Map", "Feature Map + UCB(loss)"]

moda_data = {
    "Local Only": [32.3, 32.9],
    "Logits": [78.5, 52.2],
    "Feature Map": [90.1, 93.6],
    "Feature Map + UCB(loss)": [71.2, 72.3],
}

f1_data = {
    "Local Only": [48.0, 52.8],
    "Logits": [89.6, 79.0],
    "Feature Map": [95.2, 96.8],
    "Feature Map + UCB(loss)": [83.9, 84.9],
}

# =========================
# 2) 画图参数
# =========================
x = np.arange(len(datasets))   # 每个数据集的位置
bar_width = 0.18

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

# =========================
# 3) 创建子图
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# -------------------------
# 左图：MODA
# -------------------------
ax = axes[0]
for i, method in enumerate(methods):
    offset = (i - 1.5) * bar_width
    bars = ax.bar(
        x + offset,
        moda_data[method],
        width=bar_width,
        label=method,
        color=colors[method],
        alpha=0.9
    )
    # 数值标注
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.0,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.set_title("MODA Comparison (No Packet Loss)")
ax.set_xlabel("Dataset")
ax.set_ylabel("MODA (%)")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)

# -------------------------
# 右图：F1
# -------------------------
ax = axes[1]
for i, method in enumerate(methods):
    offset = (i - 1.5) * bar_width
    bars = ax.bar(
        x + offset,
        f1_data[method],
        width=bar_width,
        label=method,
        color=colors[method],
        alpha=0.9
    )
    # 数值标注
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.0,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.set_title("F1 Comparison (No Packet Loss)")
ax.set_xlabel("Dataset")
ax.set_ylabel("F1 (%)")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)

# =========================
# 4) 总图例
# =========================
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))

fig.suptitle("Grouped Bar Chart of Methods on Wildtrack and MultiviewX", fontsize=16)

# 保存
plt.savefig("grouped_bar_no_packet_loss.png", dpi=300, bbox_inches="tight")
plt.show()