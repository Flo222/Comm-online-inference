
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = "summary_tradeoff_complete.csv"
df = pd.read_csv(csv_path)
df["comm_ratio_percent"] = 100.0 * df["actual_total_comm_mb"] / df["full_comm_total_comm_mb"]
df["comm_saving_percent"] = 100.0 - df["comm_ratio_percent"]

name_map = {
    "full_comm": "Full Comm",
    "no_comm": "No Comm",
    "logits_comm_late_fusion": "Logits Comm",
    "c2ucb_feature": "C2UCB",
}

output_dir = Path("tradeoff_plots_complete")
output_dir.mkdir(exist_ok=True)

for dataset in sorted(df["dataset"].unique()):
    sub = df[df["dataset"] == dataset].copy().sort_values("comm_ratio_percent")

    plt.figure(figsize=(8, 6))
    for _, row in sub.iterrows():
        x = row["comm_ratio_percent"]
        y = row["avg_agent_moda"]
        label = name_map.get(row["method"], row["method"])
        plt.scatter(x, y, s=120)
        plt.text(x + 1.2, y, label, fontsize=10)
    plt.xlabel("Communication Ratio vs Feature Full Comm (%)")
    plt.ylabel("Average MODA (%)")
    plt.title(f"{dataset}: MODA vs Communication")
    plt.xlim(-2, 105)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset}_moda_vs_comm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    for _, row in sub.iterrows():
        x = row["comm_ratio_percent"]
        y = row["avg_agent_f1"]
        label = name_map.get(row["method"], row["method"])
        plt.scatter(x, y, s=120)
        plt.text(x + 1.2, y, label, fontsize=10)
    plt.xlabel("Communication Ratio vs Feature Full Comm (%)")
    plt.ylabel("Average F1 (%)")
    plt.title(f"{dataset}: F1 vs Communication")
    plt.xlim(-2, 105)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset}_f1_vs_comm.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Saved plots to", output_dir.resolve())
