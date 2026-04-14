from typing import Dict, List


class DecisionPolicy:
    """
    v1:
    all nodes active, all send one fixed message type.
    """

    def __init__(self, mode: str = "all_active"):
        self.mode = mode

    def decide(self, slot_sample: Dict, node_ids: List[str]) -> Dict:
        if self.mode == "all_active":
            active_nodes = list(node_ids)
        else:
            active_nodes = list(node_ids)

        return {
            "active_nodes": active_nodes,
            "message_type": "feature",
            "fusion_mode": "baseline",
        }