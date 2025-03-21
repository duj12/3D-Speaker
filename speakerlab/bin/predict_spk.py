import os
import sys
import re
import torch
import torchaudio
import argparse
import numpy as np
from tqdm import tqdm
from kaldiio import ReadHelper
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from speakerlab.utils.builder import build
from speakerlab.utils.config import build_config
from speakerlab.utils.utils import get_logger
from speakerlab.utils.fileio import load_wav_scp
from speakerlab.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer)

parser = argparse.ArgumentParser(
    description='Detect wheether a speaker is enroll, if enroll give the speaker name')
parser.add_argument('--model_dir', default='', type=str, help='Model dir')
parser.add_argument('--enrol_emb', default='', type=str,
                    help='Enroll data, include enroll speaker embeddings.')
parser.add_argument('--test_scp', default='', type=str, help='Test data wav.scp')
parser.add_argument('--threshold', default=0.54735, type=float,
                    help='the threshold, if the similarity exceed this value, mean the speaker is enroll')
parser.add_argument('--test_label', default='', type=str, help='Test data label path')
parser.add_argument('--test_emb', default='', type=str,
                    help='Test data dir, include test speaker embeddings.')

def measure(y_pred, y_true):
    # 计算混淆矩阵，返回顺序为：[[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算 FNR = FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # 计算 FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # 利用 sklearn 计算 precision 和 recall
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    return fnr, fpr, p, r

def main():
    logger = get_logger()
    args = parser.parse_args(sys.argv[1:])

    def build_model():
        config_file = os.path.join(args.model_dir, 'config.yaml')
        config = build_config(config_file)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            msg = 'No cuda device is detected. Using the cpu device.'
            logger.warning(msg)
            device = torch.device('cpu')

        # Build the embedding model
        embedding_model = build('embedding_model', config)
        # Recover the embedding params of last epoch
        config.checkpointer['args']['checkpoints_dir'] = os.path.join(args.model_dir, 'models')
        config.checkpointer['args']['recoverables'] = {'embedding_model':embedding_model}
        checkpointer = build('checkpointer', config)
        checkpointer.recover_if_possible(epoch=config.num_epoch, device=device)

        embedding_model.to(device)
        embedding_model.eval()
        feature_extractor = build('feature_extractor', config)

        return config, embedding_model, feature_extractor, device

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

    result_path = os.path.join(args.model_dir, 'result.txt')
    # result_path = "/data/megastore/Datasets/ASR/SensitiveSpk/data/test/result_0.54735.txt"
    # args.test_label = "/data/megastore/Datasets/ASR/SensitiveSpk/data/test/label.txt"
    # args.threshold = 0.6

    scores = []     # 测试音频与注册说话人向量的最高相似度得分s
    predicts = []   # 预测结果，0标示未注册，1表示注册
    labels = []     # 如果有标注正负例标签，则加到这里面计算准确率信息

    if args.test_label != "":
        test_labels = load_wav_scp(args.test_label)

    if os.path.exists(result_path):
        with open(result_path, 'r') as fin:
            for line in fin:
                line = line.strip().split('\t')
                test_utt = line[0]
                score = float(line[2])
                # predict = int(line[3])
                if score > args.threshold:
                    predict = 1
                else:
                    predict = 0
                label = int(test_labels[test_utt])
                scores.append(score)
                predicts.append(predict)
                labels.append(label)
    else:
        enrol_dict = {}
        with ReadHelper(f'ark:{args.enrol_emb}') as reader:
            for key, array in reader:
                enrol_dict[key] = array
        test_dict = load_wav_scp(args.test_scp)

        if args.test_emb != "":
            test_emb_dict = collect(args.test_emb)
            for utt in test_dict:
                test_dict[utt] = test_emb_dict[utt]

        with open(result_path, 'w') as score_f:
            for test_utt in tqdm(test_dict, desc=f'Test...'):
                wav_path = test_dict[test_utt]
                label = int(test_labels[test_utt])

                if args.test_emb == "":
                    with torch.no_grad():
                        config, embedding_model, feature_extractor, device = build_model()
                        wav, fs = torchaudio.load(wav_path)
                        target_sample_rate = config.sample_rate
                        if fs != target_sample_rate:
                            import torchaudio.transforms as T
                            resampler = T.Resample(orig_freq=fs,
                                                   new_freq=target_sample_rate)
                            wav = resampler(wav)
                            fs = target_sample_rate

                        assert fs == config.sample_rate, f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."

                        feat = feature_extractor(wav)
                        feat = feat.unsqueeze(0)
                        feat = feat.to(device)
                        test_emb = embedding_model(feat).detach().cpu().numpy()
                else:
                    test_emb = test_dict[test_utt]

                is_enroll = 0
                max_simility = -10000.0
                detected_spk = ""
                for enrol_spk in enrol_dict:
                    enrol_emb = enrol_dict[enrol_spk]
                    cosine_score = cosine_similarity(enrol_emb.reshape(1, -1),
                                                     test_emb.reshape(1, -1))[0][0]
                    if cosine_score > max_simility:
                        max_simility = cosine_score
                        detected_spk = enrol_spk

                if max_simility > args.threshold:
                    is_enroll = 1
                    logger.info(f"{test_utt} is predicted as enroll speaker {detected_spk}, score: {max_simility}")
                else:
                    logger.info(f"{test_utt} is not enroll, the most similar speaker is {detected_spk}, score: {max_simility}")

                # write the score
                score_f.write(f"{test_utt}\t{detected_spk}\t{max_simility}\t{is_enroll}\n")
                scores.append(max_simility)
                predicts.append(is_enroll)
                labels.append(label)
            # compute metrics

    scores = np.array(scores)
    predicts = np.array(predicts)
    labels = np.array(labels)

    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, thres = compute_eer(fnr, fpr, scores)
    # write the metrics
    logger.info("Results of {} is:".format(args.test_scp))
    logger.info(f"EER = {100 * eer:.4f}, Threshold = {thres}")

    fnr_th, fpr_th, precision, recall = measure(predicts, labels)
    logger.info(f"Set threshold = {args.threshold}")
    logger.info(f"False Negative Rate (FNR): {fnr_th}")
    logger.info(f"False Positive Rate (FPR): {fpr_th}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")

if __name__ == "__main__":
    main()
