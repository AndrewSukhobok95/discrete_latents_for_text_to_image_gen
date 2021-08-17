import os
import yaml
import numpy as np
import torch


def init_report():
    ranking_1 = dict(zip(range(0, 13), [0] * 13))
    ranking_2 = dict(zip(range(0, 13), [0] * 13))
    report = {
        'CLIP_evaluation': {
            'CICS': {'score_G1': 0, 'score_G2': 0},
            'CTRS': {'dist_G1': ranking_1, 'dist_G2': ranking_2}
        }
    }
    return report


def update_counter_dict(d_main, d_update):
    for k,v in d_update.items():
        d_main[k] = d_main[k] + v
    return d_main


class EvalReport:
    def __init__(self, root_path, report_name):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        self.root_path = root_path
        self.report_name = report_name
        self.path = os.path.join(root_path, report_name + ".yaml")
        self.report = init_report()

    def get_cics(self):
        current_s1 = self.report['CLIP_evaluation']['CICS']['score_G1']
        current_s2 = self.report['CLIP_evaluation']['CICS']['score_G2']
        return current_s1, current_s2

    def set_cics(self, score_1, score_2):
        '''
        CICS = CLIP Image Comparison Score
        '''
        self.report['CLIP_evaluation']['CICS']['score_G1'] = score_1
        self.report['CLIP_evaluation']['CICS']['score_G2'] = score_2

    def update_cics(self, score_1, score_2):
        '''
        CICS = CLIP Image Comparison Score
        '''
        current_s1 = self.report['CLIP_evaluation']['CICS']['score_G1']
        current_s2 = self.report['CLIP_evaluation']['CICS']['score_G2']
        current_s1 += score_1
        current_s2 += score_2
        self.report['CLIP_evaluation']['CICS']['score_G1'] = current_s1
        self.report['CLIP_evaluation']['CICS']['score_G2'] = current_s2

    def get_ctrs_peaks(self):
        current_d1 = self.report['CLIP_evaluation']['CTRS']['dist_G1']
        current_d2 = self.report['CLIP_evaluation']['CTRS']['dist_G2']
        md1 = np.argmax([current_d1[v] for v in range(13)])
        md2 = np.argmax([current_d2[v] for v in range(13)])
        return md1, md2

    def update_ctrs(self, d1, d2):
        '''
        CTRS = CLIP Text Ranking Score
        '''
        current_d1 = self.report['CLIP_evaluation']['CTRS']['dist_G1']
        current_d2 = self.report['CLIP_evaluation']['CTRS']['dist_G2']
        self.report['CLIP_evaluation']['CTRS']['dist_G1'] = update_counter_dict(current_d1, d1)
        self.report['CLIP_evaluation']['CTRS']['dist_G2'] = update_counter_dict(current_d2, d2)

    def save(self):
        with open(self.path, 'w') as outfile:
            yaml.dump(self.report, outfile, allow_unicode=False)


def label_like(label, x):
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v


def zeros_like(x):
    return label_like(0, x)


def ones_like(x):
    return label_like(1, x)


