import torch


def aggregate_feat(feat, selection=None, aggregation='mean'):
    """
    feat: [B, N, C, H, W]
    selection:
        - None: use all views
        - [B, N] bool/0-1 mask
    """
    if selection is None:
        if aggregation == 'mean':
            overall_feat = feat.mean(dim=1)
        elif aggregation == 'max':
            overall_feat = feat.max(dim=1)[0]
        else:
            raise ValueError(f'Unknown aggregation: {aggregation}')
    else:
        selection = selection.bool().to(feat.device)
        overall_feat = feat * selection[:, :, None, None, None]
        if aggregation == 'mean':
            denom = selection.sum(dim=1).view(-1, 1, 1, 1).clamp_min(1)
            overall_feat = overall_feat.sum(dim=1) / denom
        elif aggregation == 'max':
            overall_feat = overall_feat.max(dim=1)[0]
        else:
            raise ValueError(f'Unknown aggregation: {aggregation}')

    return overall_feat