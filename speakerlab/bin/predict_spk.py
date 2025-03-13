import os
import sys
import re
import torch
import torchaudio
import argparse
import numpy as np
from tqdm import tqdm
from kaldiio import ReadHelper, WriteHelper
from sklearn.metrics.pairwise import cosine_similarity

from speakerlab.utils.builder import build
from speakerlab.utils.config import build_config
from speakerlab.utils.utils import get_logger
from speakerlab.utils.fileio import load_wav_scp
from speakerlab.utils.score_metrics import (
    compute_pmiss_pfa_rbst, compute_eer, compute_c_norm, plot_det_curve)

parser = argparse.ArgumentParser(
    description='Detect wheether a speaker is enroll, if enroll give the speaker name')
parser.add_argument('--model_dir', default='', type=str, help='Model dir')
parser.add_argument('--enrol_data', default='', type=str,
                    help='Enroll data dir, include enroll speaker embeddings.')
parser.add_argument('--test_data', default='', type=str, help='Test data wav.scp')
parser.add_argument('--test_label', default='', type=str, help='Test data label path')
parser.add_argument('--threshold', default=0.37327, type=float,
                    help='the threshold, if the similarity exceed this value, mean the speaker is enroll')


def main():
    logger = get_logger()
    args = parser.parse_args(sys.argv[1:])
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

    def collect(data_dir):
        data_dict = {}
        emb_arks = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if re.search('spkemb.ark$',i)]
        if len(emb_arks) == 0:
            raise Exception(f'No embedding ark files found in {data_dir}')

        # load embedding data
        for ark in emb_arks:
            with ReadHelper(f'ark:{ark}') as reader:
                for key, array in reader:
                    data_dict[key] = array

        return data_dict

    enrol_dict = collect(args.enrol_data)
    test_dict = load_wav_scp(args.test_data)
    result_path = os.path.join(args.model_dir, 'result.txt')

    scores = []     # 测试音频与注册说话人向量的最高相似度得分
    predicts = []   # 预测结果，0标示未注册，1表示注册
    labels = []     # 如果有标注正负例标签，则加到这里面计算准确率信息

    if args.test_label != "":
        test_labels = load_wav_scp(args.test_label)


    with open(result_path, 'w') as score_f:
        for test_utt in tqdm(test_dict, desc=f'Test...'):
            wav_path = test_dict[test_utt]
            label = test_labels[test_utt]

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
            score_f.write(f"{test_utt}\t{detected_spk}\t{max_simility}\t{is_enroll}")
            scores.append(max_simility)
            predicts.append(is_enroll)
            labels.append(label)


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

        # write the metrics
        logger.info("Results of {} is:".format(args.test_data))
        logger.info(f"EER = {100 * eer:.4f}, Threshold = {thres}")
        logger.info("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, min_dcf))


if __name__ == "__main__":
    main()
