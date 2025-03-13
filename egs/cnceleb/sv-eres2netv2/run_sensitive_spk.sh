#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
. ./path.sh || exit 1

stage=7
stop_stage=7

data=/data/megastore/Datasets/ASR/SensitiveSpk/data
exp=exp
exp_lm_dir=$exp/eres2netv2w24s4ep4_0312

gpus="0 1 2 3 4 5 6 7"

. utils/parse_options.sh || exit 1

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Extract embeddings of test datasets.
  echo "Stage5: Extracting speaker embeddings..."
  torchrun --nproc_per_node=4 speakerlab/bin/extract.py --exp_dir $exp_lm_dir \
           --data $data/test/wav.scp --use_gpu --gpu 5
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Output score metrics.
  echo "Stage6: Computing score metrics..."
  trials="$data/test/trials.txt"
  python speakerlab/bin/compute_score_metrics.py --enrol_data $exp_lm_dir/embeddings --test_data $exp_lm_dir/embeddings \
                                                 --scores_dir $exp_lm_dir/scores --trials $trials  \
                                                 --enroll_utt2spk $data/enroll/utt2spk
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  # Output score metrics.
  echo "Stage7: Predict the test wav"
  python speakerlab/bin/predict_spk.py \
    --model_dir $exp_lm_dir             \
    --enrol_data $exp_lm_dir/embeddings \
    --test_data "$data/test/test.scp" \
    --test_label "$data/test/label.txt"
fi