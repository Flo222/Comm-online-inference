import csv
import os
from typing import Dict, List


class OnlineLogger:
    def __init__(self, logdir: str):
        self.logdir = logdir
        self.records: List[Dict] = []

    def log_slot(self, slot_id: int, metrics: Dict, decision: Dict, comm_stats: Dict):
        row = {
            "slot_id": slot_id,
            **metrics,
            "num_active_nodes": len(decision.get("active_nodes", [])),
            "message_type": decision.get("message_type", ""),
            "fusion_mode": decision.get("fusion_mode", ""),
            "comm_cost": comm_stats.get("comm_cost", 0),
            "num_messages": comm_stats.get("num_messages", 0),
        }
        self.records.append(row)

    def summarize(self) -> Dict:
        if not self.records:
            return {}

        summary = {
            "num_slots": len(self.records),
            "avg_comm_cost": sum(r["comm_cost"] for r in self.records) / len(self.records),
            "avg_num_messages": sum(r["num_messages"] for r in self.records) / len(self.records),
        }
        return summary

    def save_csv(self, filename: str = "online_log.csv"):
        if not self.records:
            return

        os.makedirs(self.logdir, exist_ok=True)
        fpath = os.path.join(self.logdir, filename)

        fieldnames = list(self.records[0].keys())
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[OnlineLogger] saved to {fpath}")