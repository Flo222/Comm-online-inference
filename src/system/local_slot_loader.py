# src/system/local_slot_loader.py

class LocalSlotLoader:
    """
    单节点 slot loader。
    它不应该返回全相机 imgs，只返回 cam_idx 对应的一路数据。
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def load(self, slot_id: int, cam_idx: int):
        if not hasattr(self.dataset, "get_single_cam_slot"):
            raise RuntimeError(
                "dataset must implement get_single_cam_slot(index, cam_idx) "
                "for decentralized local loading."
            )

        return self.dataset.get_single_cam_slot(slot_id, cam_idx)