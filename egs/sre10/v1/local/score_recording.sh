#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
# modified by z3c
# This script divide the scores for males and females and record in the file


lda_dim=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# == 4 ]; then
	trials=$1
	trials_female=$2
	trials_male=$3
	scores_dir=$4 # exp/cs_score

	utils/filter_scp.pl $trials_female $scores_dir/eval_scores > $scores_dir/eval_scores_female
	utils/filter_scp.pl $trials_male $scores_dir/eval_scores > $scores_dir/eval_scores_male

	# eer_female=`compute-eer <(python3 local/prepare_for_eer.py $trials_female $scores_dir/eval_scores_female) 2> /dev/null`

	# eer_male=`compute-eer <(python3 local/prepare_for_eer.py $trials_male $scores_dir/eval_scores_male) 2> /dev/null`
	
	eer_female=`compute-eer-mdcf <(python3 local/prepare_for_eer.py $trials_female $scores_dir/eval_scores_female) 2> /dev/null`

	eer_male=`compute-eer-mdcf <(python3 local/prepare_for_eer.py $trials_male $scores_dir/eval_scores_male) 2> /dev/null`	

	echo "female: $eer_female; male: $eer_male" 
	echo "female: $eer_female; male: $eer_male"  >> $scores_dir/tot_scores


elif [ $# == 2 ]; then
	trials=$1
	scores_dir=$2 # exp/cs_score

else
  echo "Usage: $0 <trials> <trials-female> <trials-male> <scores-dir> or "
  echo "Usage: $0 <trials> <scores-dir>  "
  
fi


echo " DNN-UBM EER" 
# eer=`compute-eer <(python3 local/prepare_for_eer.py $trials $scores_dir/eval_scores) 2> /dev/null`
# s$train_cmd exp/ivector_score/core_lda_lclda_plda600/log/score.log \
eer=`compute-eer-mdcf <(python3 local/prepare_for_eer.py $trials $scores_dir/eval_scores) 2> /dev/null`

wait
echo "total: $eer.  --lda :$lda_dim" 
echo "total: $eer.  --lda :$lda_dim"  >> $scores_dir/tot_scores


# echo "The bad target key:" 
# get-bad-key <(python3 local/prepare_for_eer.py $trials $scores_dir/eval_scores) $trials 