import matplotlib.pyplot as plt

# =========================
# Data from current results
# =========================
wildtrack = {
    "Full Comm":   {"moda": 90.1, "f1": 95.2, "comm_mb": 18579.456},
    "No Comm":     {"moda": 32.3, "f1": 48.0, "comm_mb": 0.0},
    "Late Fusion": {"moda": 78.5, "f1": 89.6, "comm_mb": 18579.456},
    "Bandit v3":   {"moda": 76.4, "f1": 87.3, "comm_mb": 7232.717},
}

multiviewx = {
    "Full Comm":   {"moda": 93.6, "f1": 96.8, "comm_mb": 12288.000},
    "No Comm":     {"moda": 32.9, "f1": 52.8, "comm_mb": 0.0},
    "Late Fusion": {"moda": 52.2, "f1": 79.0, "comm_mb": 12288.000},
    "Bandit v3":   {"moda": 77.6, "f1": 87.9, "comm_mb": 5939.200},
}

# Optional label offsets for annotation
label_offsets = {
    "Full Comm":   (8, 8),
    "No Comm":     (8, -12),
    "Late Fusion": (8, 8),
    "Bandit v3":   (8, -12),
}


def plot_tradeoff(ax, data, metric_key, title, ylabel):
    for method, vals in data.items():
        x = vals["comm_mb"]
        y = vals[metric_key]
        ax.scatter(x, y, s=80, label=method)
        dx, dy = label_offsets.get(method, (5, 5))
        ax.annotate(
            method,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Total Communication (MB)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_tradeoff(
        axes[0, 0],
        wildtrack,
        metric_key="moda",
        title="(a) Wildtrack: MODA vs Communication",
        ylabel="MODA (%)"
    )
    plot_tradeoff(
        axes[0, 1],
        multiviewx,
        metric_key="moda",
        title="(b) MultiviewX: MODA vs Communication",
        ylabel="MODA (%)"
    )
    plot_tradeoff(
        axes[1, 0],
        wildtrack,
        metric_key="f1",
        title="(c) Wildtrack: F1 vs Communication",
        ylabel="F1 (%)"
    )
    plot_tradeoff(
        axes[1, 1],
        multiviewx,
        metric_key="f1",
        title="(d) MultiviewX: F1 vs Communication",
        ylabel="F1 (%)"
    )

    # Make a shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11, frameon=True)

    fig.suptitle("Communication-Performance Trade-off of Different Collaboration Strategies", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = "main_tradeoff_figure.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()