#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
# modified by z3c
# This script trains LDA and PLDA models and does scoring.

use_existing_models=false
simple_length_norm=false # If true, replace the default length normalization
                         # performed in PLDA  by an alternative that
                         # normalizes the length of the iVectors to be equal
                         # to the square root of the iVector dimension.

echo "$0 $@"  # Print the command line for logging

lda_dim=200
use_existing_models=false
covar_factor=0.0

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 8 ]; then
  echo "Usage: $0 --lda-dim 200 <plda-data-dir> <enroll-data-dir> <test-data-dir> <plda-ivec-dir> <enroll-ivec-dir> <test-ivec-dir> <trials-file> <scores-dir>"
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
  for f in ${plda_ivec_dir}/mean.vec ${plda_ivec_dir}/transform.mat ${plda_ivec_dir}/plda ; do
    [ ! -f $f ] && echo "No such file $f" && exit 1;
  done
else
	# Compute the mean vector for centering the evaluation ivector. without length normalization
	$train_cmd $plda_ivec_dir/log/compute_mean.log \
		ivector-mean scp:$plda_ivec_dir/ivector.scp \
		$plda_ivec_dir/mean.vec || exit 1;

	# This script uses LDA to decrease the dimensionality prior to PLDA.
	$train_cmd $plda_ivec_dir/log/lda.log \
		ivector-compute-lda --total-covariance-factor=$covar_factor --dim=$lda_dim \
		"ark:ivector-normalize-length scp:$plda_ivec_dir/ivector.scp ark:- |" \
		ark:$plda_data_dir/utt2spk $plda_ivec_dir/transform.mat || exit 1;

	# Train a PLDA model. may need a sub global mean here
	$train_cmd $plda_ivec_dir/log/plda.log \
		ivector-compute-plda ark:$plda_data_dir/spk2utt \
		"ark:transform-vec  $plda_ivec_dir/transform.mat scp:$plda_ivec_dir/ivector.scp ark:- | ivector-normalize-length ark:-  ark:- |" \
		 $plda_ivec_dir/plda || exit 1;

fi

# Get results using the out-of-domain PLDA model. 
# Guess: ivector-subtract-global-mean is due to transform-vec add mean 
$train_cmd $scores_dir/log/eval_scoring.log \
	ivector-plda-scoring --normalize-length=true \
	--simple-length-normalization=$simple_length_norm \
	--num-utts=ark:$enroll_ivec_dir/num_utts.ark \
	"ivector-copy-plda --smoothing=0.0  $plda_ivec_dir/plda - |" \
	"ark:ivector-mean ark:$enroll_data_dir/spk2utt scp:$enroll_ivec_dir/ivector.scp ark:- | ivector-subtract-global-mean $plda_ivec_dir/mean.vec ark:- ark:- | transform-vec $plda_ivec_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
	"ark:ivector-subtract-global-mean $plda_ivec_dir/mean.vec scp:$test_ivec_dir/ivector.scp ark:- | transform-vec $plda_ivec_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
	"cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/eval_scores || exit 1;

