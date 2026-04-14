import numpy as np
from src.evaluation.pyeval.evaluateDetection import evaluateDetection_py


def evaluate(res_fpath, gt_fpath, dataset='wildtrack', frames=None):
    """
    Unified return order:
        moda, modp, precision, recall
    """
    moda, modp, precision, recall, stats = evaluateDetection_py(res_fpath, gt_fpath, frames)
    return moda, modp, precision, recall


if __name__ == "__main__":
    import os

    res_fpath = os.path.abspath('test-demo.txt')
    gt_fpath = os.path.abspath('gt-demo.txt')
    os.chdir('../..')
    print(os.path.abspath('.'))

    moda, modp, precision, recall = evaluate(res_fpath, gt_fpath, dataset='Wildtrack')
    print(f'eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')