#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script trains an LDA transform, applies it to the enroll and
# test i-vectors and does cosine scoring.

use_existing_models=false
simple_length_norm=false # If true, replace the default length normalization
                         # performed in PLDA  by an alternative that
                         # normalizes the length of the iVectors to be equal
                         # to the square root of the iVector dimension.
lda_dim=150
# covar_factor=0.1
covar_factor=0.0
beta=1.0
nj=16

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 8 ]; then
  echo "Usage: $0 <lda-data-dir> <enroll-data-dir> <test-data-dir> <lda-ivec-dir> <enroll-ivec-dir> <test-ivec-dir> <trials-file> <scores-dir>"
fi

plda_data_dir=$1 #data/sre
enroll_data_dir=$2 #data/sre10_train
test_data_dir=$3 # nouse: data/sre10_test 
plda_ivec_dir=$4 #sub_sre_path_cs
enroll_ivec_dir=$5 #sub_sre_train_path_cs
test_ivec_dir=$6 #sub_sre_test_path_cs
trials=$7
scores_dir=$8 # exp/cs_score

if [ "$use_existing_models" == "true" ]; then
  for f in ${test_ivec_dir}/gmm_pdf.mat ${plda_ivec_dir}/transform_lcp.ark  ${plda_ivec_dir}/spk_mean.ark ${plda_ivec_dir}/lcplda_plda_set.ark ; do
    [ ! -f $f ] && echo "No such file $f" && exit 1;
  done
else
  
mkdir -p ${plda_ivec_dir}/log
run.pl ${plda_ivec_dir}/log/lcplda_plda1.log \
    ivector-compute-lcplda-pdf --nj=$nj \
      --pdf=${test_ivec_dir}/gmm_pdf.mat \
      --dim=$lda_dim --total-covariance-factor=$covar_factor \
    "ark:ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp ark:- |" \
      ark:${plda_data_dir}/utt2spk \
      ark:${plda_ivec_dir}/transform_lcp.ark \
      ark,t:${plda_ivec_dir}/spk_mean.ark || exit 1;

  run.pl ${plda_ivec_dir}/log/spk_mean_ln.log \
    ivector-normalize-length ark:${plda_ivec_dir}/spk_mean.ark ark:${plda_ivec_dir}/spk_mean_ln.ark || exit 1;

  # # Train a PLDA model. using transform-vec-lclda function
  # # apply spk_mean_ln here, use cosine distance to find nearest spker
  # $train_cmd $plda_ivec_dir/log/lcplda_plda2.log \
  #   ivector-compute-plda ark:$plda_data_dir/spk2utt \
  #   "ark:ivector-normalize-length scp:$plda_ivec_dir/ivector.scp ark:- | 
  #   transform-vec-lclda --development=true ark:${plda_ivec_dir}/transform_lcp.ark ark:${plda_ivec_dir}/spk_mean_ln.ark  ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
  #    $plda_ivec_dir/lcplda_plda || exit 1;

  # Train a PLDA set model. using transform-vec-lclda function
  # apply spk_mean_ln here, use cosine distance to find nearest spker
  $train_cmd $plda_ivec_dir/log/lcplda_plda2.log \
    ivector-compute-plda-set --nj=$nj \
      --pdf=${test_ivec_dir}/gmm_pdf.mat \
      --spk-mean=ark:${plda_ivec_dir}/spk_mean_ln.ark \
      ark:$plda_data_dir/spk2utt \
      ark:${plda_ivec_dir}/transform_lcp.ark  \
      "ark:ivector-normalize-length scp:$plda_ivec_dir/ivector.scp ark:- |" \
      ark:$plda_ivec_dir/lcplda_plda_set.ark || exit 1;

fi

mkdir -p $scores_dir/log
echo "do lcplda_plda_scoring!"
# # Guess: ivector-subtract-global-mean is due to transform-vec add mean 
# $train_cmd $scores_dir/log/eval_scoring.log \
#   ivector-lclda-plda-scoring2 --normalize-length=true \
#   --simple-length-normalization=$simple_length_norm \
#   --num-utts=ark:$enroll_ivec_dir/num_utts.ark \
#   --transform=ark:${plda_ivec_dir}/transform_lcp.ark \
#   --spk-mean=ark:${plda_ivec_dir}/spk_mean_ln.ark \
#   "ivector-copy-plda --smoothing=0.0  $plda_ivec_dir/lcplda_plda - |" \
#   ark:"ivector-mean ark:$enroll_data_dir/spk2utt scp:$enroll_ivec_dir/ivector.scp ark:- | ivector-subtract-global-mean --subtract-mean=false $plda_ivec_dir/mean.vec ark:- ark:- |"  \
#   ark:"ivector-subtract-global-mean --subtract-mean=false $plda_ivec_dir/mean.vec scp:$test_ivec_dir/ivector.scp ark:- |" \
#   "cat '$trials' | cut -d\  --fields=1,2 |" \
#   $scores_dir/eval_scores  || exit 1;

$train_cmd $scores_dir/log/eval_scoring.log \
  ivector-lclda-pldaset-scoring --normalize-length=true \
  --simple-length-normalization=$simple_length_norm \
  --num-utts=ark:$enroll_ivec_dir/num_utts.ark \
  --transform=ark:${plda_ivec_dir}/transform_lcp.ark \
  --spk-mean=ark:${plda_ivec_dir}/spk_mean_ln.ark \
  ark:"$plda_ivec_dir/lcplda_plda_set.ark" \
  ark:"ivector-mean ark:$enroll_data_dir/spk2utt scp:$enroll_ivec_dir/ivector.scp ark:- | ivector-subtract-global-mean --subtract-mean=false $plda_ivec_dir/mean.vec ark:- ark:- |"  \
  ark:"ivector-subtract-global-mean --subtract-mean=false $plda_ivec_dir/mean.vec scp:$test_ivec_dir/ivector.scp ark:- |" \
  "cat '$trials' | cut -d\  --fields=1,2 |" \
  $scores_dir/eval_scores  || exit 1;