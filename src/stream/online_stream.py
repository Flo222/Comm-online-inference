from typing import Dict, Iterator


class OnlineStream:
    """
    Wrap an existing dataset into a time-ordered online stream.
    v1:
      - assume dataset order is already time-consistent
      - one dataset sample = one slot
    """

    def __init__(self, dataset, sort_by_time: bool = True):
        self.dataset = dataset
        self.sort_by_time = sort_by_time
        self.indices = self._build_indices()
        self.ptr = 0

    def _build_indices(self):
        return list(range(len(self.dataset)))

    def reset(self):
        self.ptr = 0

    def __len__(self):
        return len(self.indices)

    def __iter__(self) -> Iterator[Dict]:
        self.reset()
        return self

    def __next__(self) -> Dict:
        if self.ptr >= len(self.indices):
            raise StopIteration

        idx = self.indices[self.ptr]
        raw = self.dataset[idx]
        slot = self._to_slot(raw, slot_id=self.ptr, dataset_idx=idx)

        self.ptr += 1
        return slot

    def _to_slot(self, raw_sample, slot_id: int, dataset_idx: int) -> Dict:
        if isinstance(raw_sample, (list, tuple)) and len(raw_sample) == 6:
            imgs, world_gt, imgs_gt, affine_mats, frame, keep_cams = raw_sample
            return {
                "slot_id": slot_id,
                "dataset_idx": dataset_idx,
                "frame_id": frame,
                "images": imgs,
                "world_gt": world_gt,
                "imgs_gt": imgs_gt,
                "affine_mats": affine_mats,
                "keep_cams": keep_cams,
                "meta": {},
            }

        return {
            "slot_id": slot_id,
            "dataset_idx": dataset_idx,
            "raw": raw_sample,
            "meta": {},
        }