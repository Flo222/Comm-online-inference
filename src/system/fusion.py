# src/system/fusion.py

from typing import Any, Dict, List


class FusionEngine:
    """
    v1 fusion engine:
    - does not yet perform sophisticated fusion
    - stores local outputs/messages and leaves the final task model
      integration for next step
    """

    def __init__(self, mode: str = "baseline"):
        self.mode = mode

    def fuse(self, local_outputs: List[Dict], messages: List[Any]) -> Dict:
        return {
            "fusion_mode": self.mode,
            "num_local_outputs": len(local_outputs),
            "num_messages": len(messages),
            "local_outputs": local_outputs,
            "messages": messages,
        }

    def predict(self, fused_state: Dict) -> Dict:
        """
        Placeholder global prediction.
        In the next step, this can call MVDet/MVCNN-based inference.
        """
        return {
            "prediction_ready": True,
            "fused_state": fused_state,
        }