# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import re
import argparse
import numpy as np
from tqdm import tqdm
from kaldiio import ReadHelper, WriteHelper
from sklearn.metrics.pairwise import cosine_similarity

from speakerlab.utils.utils import get_logger
from speakerlab.utils.score_metrics import (
    compute_pmiss_pfa_rbst, compute_eer, compute_c_norm, plot_det_curve)

parser = argparse.ArgumentParser(description='Compute score and metrics')
parser.add_argument('--enrol_data', default='', type=str, help='Enroll data dir')
parser.add_argument('--test_data', default='', type=str, help='Test data dir')
parser.add_argument('--scores_dir', default='', type=str, help='Scores dir')
parser.add_argument('--trials', nargs='+', help='Trial')
parser.add_argument('--p_target', default=0.01, type=float, help='p_target in DCF')
parser.add_argument('--c_miss', default=1, type=float, help='c_miss in DCF')
parser.add_argument('--c_fa', default=1, type=float, help='c_fa in DCF')
parser.add_argument('--enroll_utt2spk', default='', type=str, help='the path of enroll utt2spk, use to average enroll embedding into per spk')


def main():
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.scores_dir, exist_ok=True)

    result_path = os.path.join(args.scores_dir, 'result.metrics')
    logger = get_logger(fpath=result_path, fmt = "%(message)s")
    det_curve_path = os.path.join(args.scores_dir, 'result.det.png')

    def collect(data_dir):
        data_dict = {}
        emb_arks = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if re.search('.ark$',i)]
        if len(emb_arks) == 0:
            raise Exception(f'No embedding ark files found in {data_dir}')

        # load embedding data
        for ark in emb_arks:
            with ReadHelper(f'ark:{ark}') as reader:
                for key, array in reader:
                    data_dict[key] = array

        return data_dict

    enrol_dict = collect(args.enrol_data)

    enrol_spkemb_dict = {}
    enrol_spk_uttnum_dict = {}  # 用于在线更新平均值
    if args.enroll_utt2spk != '':
        enrol_spks = {}
        with open(args.enroll_utt2spk, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip().split()
                utt, spk = line[0], line[1]
                enrol_spks[utt] = spk

        for utt in enrol_dict.keys():
            if utt in enrol_spks:
                vec = enrol_dict[utt]
                spk = enrol_spks[utt]
                if spk in enrol_spk_uttnum_dict:
                    enrol_spk_uttnum_dict[spk] += 1
                else:
                    enrol_spk_uttnum_dict[spk] = 1
                if spk in enrol_spkemb_dict:
                    avg = enrol_spkemb_dict[spk]
                    enrol_spkemb_dict[spk] = avg + (vec - avg)/enrol_spk_uttnum_dict[spk]
                else:
                    enrol_spkemb_dict[spk] = vec

        # 存储所有注册说话人向量
        logger.info(f"Total enroll speakers: {len(enrol_spkemb_dict)}")
        emb_ark = os.path.join(args.enrol_data, 'enroll_spkemb.ark')
        emb_scp = os.path.join(args.enrol_data, 'enroll_spkemb.scp')
        with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer:
            for k in enrol_spkemb_dict:
                emb = enrol_spkemb_dict[k]
                writer(k, emb)

    test_dict = collect(args.test_data)

    for trial in args.trials:
        scores = []
        labels = []

        trial_name = os.path.basename(trial)
        score_path = os.path.join(args.scores_dir, f'{trial_name}.score')

        if os.path.exists(score_path) and args.enroll_utt2spk == '':
            with open(score_path, 'r') as score_f:
                lines = score_f.readlines()
                for line in tqdm(lines, desc=f'scoring trial {trial_name}'):
                    pair = line.strip().split()
                    cosine_score = float(pair[-1])
                    scores.append(cosine_score)
                    if pair[2] == '1' or pair[2] == 'target':
                        labels.append(1)
                    elif pair[2] == '0' or pair[2] == 'nontarget':
                        labels.append(0)
                    else:
                        raise Exception(f'Unrecognized label in {line}.')
        else:
            with open(trial, 'r') as trial_f, open(score_path, 'w') as score_f:
                lines = trial_f.readlines()
                for line in tqdm(lines, desc=f'scoring trial {trial_name}'):
                    pair = line.strip().split()
                    enrol_emb, test_emb = enrol_dict[pair[0]], test_dict[pair[1]]
                    if args.enroll_utt2spk != '':
                        enrol_emb = enrol_spkemb_dict[enrol_spks[pair[0]]]

                    cosine_score = cosine_similarity(enrol_emb.reshape(1, -1),
                                                  test_emb.reshape(1, -1))[0][0]
                    # write the score
                    score_f.write(' '.join(pair)+' %.5f\n'%cosine_score)
                    scores.append(cosine_score)
                    if pair[2] == '1' or pair[2] == 'target':
                        labels.append(1)
                    elif pair[2] == '0' or pair[2] == 'nontarget':
                        labels.append(0)
                    else:
                        raise Exception(f'Unrecognized label in {line}.')

        # compute metrics
        scores = np.array(scores)
        labels = np.array(labels)

        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        eer, thres = compute_eer(fnr, fpr, scores)
        min_dcf = compute_c_norm(fnr,
                                fpr,
                                p_target=args.p_target,
                                c_miss=args.c_miss,
                                c_fa=args.c_fa)
        # plot det curve
        plot_det_curve(fnr, fpr, save_path=det_curve_path)

        # write the metrics
        logger.info("Results of {} is:".format(trial_name))
        logger.info(f"EER = {100 * eer:.4f}, Threshold = {thres}")
        logger.info("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, min_dcf))


if __name__ == "__main__":
    main()
