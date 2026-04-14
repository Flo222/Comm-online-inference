from torch import nn
from src.models.aggregation import aggregate_feat


class MultiviewBase(nn.Module):
    def __init__(self, dataset, aggregation='max'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.aggregation = aggregation

    def forward(self, imgs, M=None, down=1, keep_cams=None, visualize=False):
        """
        Baseline forward:
            1) extract per-view features
            2) aggregate all available views
            3) predict task output
        """
        feat, aux_res = self.get_feat(imgs, M, down, visualize)
        overall_feat = aggregate_feat(feat, keep_cams, aggregation=self.aggregation)
        overall_res = self.get_output(overall_feat, visualize)
        return overall_res, aux_res

    def get_feat(self, imgs, M, down=1, visualize=False):
        raise NotImplementedError

    def get_output(self, overall_feat, visualize=False):
        raise NotImplementedError