import os
import yaml
import torch


class EvalReport:
    def __init__(self, root_path, report_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        self.root_path = root_path
        self.report_name = report_name
        self.path = os.path.join(root_path, report_name + ".yaml")
        self.report = {
            'CLIP_evaluation': {
                '2_imgs_vs_1_txt': {'score_G1': None, 'score_G2': None}
            }
        }

    def save(self, g1_score, g2_score):
        self.report['CLIP_evaluation']['2_imgs_vs_1_txt']['score_G1'] = g1_score
        self.report['CLIP_evaluation']['2_imgs_vs_1_txt']['score_G2'] = g2_score
        with open(self.path, 'w', encoding='utf8') as outfile:
            yaml.dump(self.report, outfile, allow_unicode=True)


def label_like(label, x):
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v


def zeros_like(x):
    return label_like(0, x)


def ones_like(x):
    return label_like(1, x)


