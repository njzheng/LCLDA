#!/bin/bash
# Copyright 2015-2017   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.
#
# This example script shows how to replace the GMM-UBM
# with a DNN trained for ASR. It also demonstrates the
# using the DNN to create a supervised-GMM.


# LDA and PLDA for ivector

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
trials_female=data/sre10_test_female/trials
trials_male=data/sre10_test_male/trials
trials=data/sre10_test/trials
trials_10sec=data/sre_10sec_test/trials
trials_core_10sec=data/sre_core_10sec_test/trials
trials_8conv_10sec=data/sre_8conv_10sec_test/trials
trials_cc=data/sre_cc_test/trials # core - core
num_components=2048 # Larger than this doesn't make much of a difference.


ivector_dim=600

stage=6


if [ $stage -le 1 ]; then

  # Prepare the SRE 2010 evaluation data.
  SRE2010_path=/scratch/njzheng/DATA/sre2010/LDC2012E09/SRE10
  # local/make_sre_2010_test.pl $SRE2010_path/eval/ data/
  # local/make_sre_2010_train.pl $SRE2010_path/eval/ data/

  # # 10sec-10sec
  # local/make_sre_10sec_test.pl $SRE2010_path/eval/ data/
  # local/make_sre_10sec_train.pl $SRE2010_path/eval/ data/


  # # core-10sec
  # local/make_sre_core_10sec_test.pl $SRE2010_path/eval/ data/
  # local/make_sre_core_10sec_train.pl $SRE2010_path/eval/ data/

  # # 8conv-10sec
  # local/make_sre_8conv_10sec_test.pl $SRE2010_path/eval/ data/
  # local/make_sre_8conv_10sec_train.pl $SRE2010_path/eval/ data/

  # # core-core
  local/make_sre_cc_test.pl $SRE2010_path/eval/ data/
  local/make_sre_cc_train.pl $SRE2010_path/eval/ data/

  # # Prepare a collection of NIST SRE data prior to 2010. This is
  # # used to train the PLDA model and is also combined with SWB
  # # for UBM and i-vector extractor training data.
  # pre_sre_path=/scratch/njzheng/DATA/sre/LDC2009E100
  # local/make_sre.sh $pre_sre_path data

  # # Prepare SWB for UBM and i-vector extractor training.
  # SWBD_path=/scratch/njzheng/DATA/swbd/LDC2016E81
  # local/make_swbd2_phase2.pl $SWBD_path/LDC99S79 \
  #   data/swbd2_phase2_train
  # local/make_swbd2_phase3.pl $SWBD_path/LDC2002S06 \
  #   data/swbd2_phase3_train
  
  # local/make_swbd_cellular1.pl $SWBD_path/LDC2001S13 \
  #   data/swbd_cellular1_train
  # local/make_swbd_cellular2.pl $SWBD_path/LDC2004S07 \
  #   data/swbd_cellular2_train


  # utils/combine_data.sh data/train \
  #     data/swbd_cellular1_train data/swbd_cellular2_train \
  #     data/swbd2_phase2_train data/swbd2_phase3_train data/sre

  # utils/copy_data_dir.sh data/train data/train_dnn
  # utils/copy_data_dir.sh data/sre data/sre_dnn
  # utils/copy_data_dir.sh data/sre10_train data/sre10_train_dnn
  # utils/copy_data_dir.sh data/sre10_test data/sre10_test_dnn
  # # 10sec-10sec
  # utils/copy_data_dir.sh data/sre_10sec_train data/sre_10sec_train_dnn
  # utils/copy_data_dir.sh data/sre_10sec_test data/sre_10sec_test_dnn
  # # core-10sec
  # utils/copy_data_dir.sh data/sre_core_10sec_train data/sre_core_10sec_train_dnn
  # utils/copy_data_dir.sh data/sre_core_10sec_test data/sre_core_10sec_test_dnn
  # # 8conv-10sec
  # utils/copy_data_dir.sh data/sre_8conv_10sec_train data/sre_8conv_10sec_train_dnn
  # utils/copy_data_dir.sh data/sre_8conv_10sec_test data/sre_8conv_10sec_test_dnn
  
  # # core-core
  # utils/copy_data_dir.sh data/sre_cc_train data/sre_cc_train_dnn
  # utils/copy_data_dir.sh data/sre_cc_test data/sre_cc_test_dnn
fi


if [ $stage -le 2 ]; then
  # # Extract speaker recogntion features.
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/train exp/make_mfcc $mfccdir #line 56
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/sre exp/make_mfcc $mfccdir #sre08
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/sre10_train exp/make_mfcc $mfccdir
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/sre10_test exp/make_mfcc $mfccdir

  # # 10sec-10sec
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/sre_10sec_train exp/make_mfcc $mfccdir
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/sre_10sec_test exp/make_mfcc $mfccdir

  # # core-10sec
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/sre_core_10sec_train exp/make_mfcc $mfccdir
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/sre_core_10sec_test exp/make_mfcc $mfccdir

  # # 8conv-10sec
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/sre_8conv_10sec_train exp/make_mfcc $mfccdir
  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
  #   data/sre_8conv_10sec_test exp/make_mfcc $mfccdir

  # core-core
  steps/make_mfcc.sh --mfcc-config conf/mfcc13.conf --nj 16 --cmd "$train_cmd" \
    data/sre_cc_train exp/make_mfcc $mfccdir
  steps/make_mfcc.sh --mfcc-config conf/mfcc13.conf --nj 16 --cmd "$train_cmd" \
    data/sre_cc_test exp/make_mfcc $mfccdir


  # for name in \
    # sre_dnn sre10_train_dnn sre10_test_dnn train_dnn sre \
    # sre10_train sre10_test train \
    # sre_10sec_train sre_10sec_test sre_core_10sec_train sre_core_10sec_test \
    # sre_8conv_10sec_train sre_8conv_10sec_test \
  for name in sre_cc_train sre_cc_test ; do
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 3 ]; then
  # # Compute VAD decisions. These will be shared across both sets of features.
  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/train exp/make_vad $vaddir
  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/sre exp/make_vad $vaddir
  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/sre10_train exp/make_vad $vaddir
  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/sre10_test exp/make_vad $vaddir


  # # rename
  # cp -R data/sre10_train data/sre_train
  # cp -R data/sre10_test data/sre_test

  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/sre_10sec_train exp/make_vad $vaddir
  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/sre_10sec_test exp/make_vad $vaddir

  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/sre_core_10sec_train exp/make_vad $vaddir
  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/sre_core_10sec_test exp/make_vad $vaddir

  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/sre_8conv_10sec_train exp/make_vad $vaddir
  # sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
  #   data/sre_8conv_10sec_test exp/make_vad $

  sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
    data/sre_cc_train exp/make_vad $vaddir
  sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
    data/sre_cc_test exp/make_vad $vaddir

  # for name in \
    # sre sre10_train sre10_test train sre_test sre_train\
    # sre_10sec_train sre_10sec_test sre_core_10sec_train sre_core_10sec_test\
    # sre_8conv_10sec_train sre_8conv_10sec_test \
  for name in sre_cc_train sre_cc_test ; do
    # may lose vad in train here
    utils/fix_data_dir.sh data/${name}
  done

fi


if [ $stage -le -1 ]; then
	# Reduce the amount of training data for the UBM.
	utils/subset_data_dir.sh data/train 16000 data/train_16k
	utils/subset_data_dir.sh data/train 32000 data/train_32k

	# Train UBM and i-vector extractor.
	# using apply-cmvn-sliding here
	sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
		--nj 20 --num-threads 8 \
		data/train_16k $num_components \
		exp/diag_ubm_$num_components 

	sid/train_full_ubm.sh --nj 20 --remove-low-count-gaussians false \
	--cmd "$train_cmd --mem 25G" data/train_32k \
  	exp/diag_ubm_$num_components exp/full_ubm_$num_components
fi

sre_path=exp/ivectors_sre$ivector_dim
sre_train_path=exp/ivectors_sre10_train$ivector_dim
sre_test_path=exp/ivectors_sre10_test$ivector_dim

sre_10sec_test_path=exp/ivectors_10sec_test$ivector_dim
sre_10sec_train_path=exp/ivectors_10sec_train$ivector_dim
sre_core_10sec_test_path=exp/ivectors_core_10sec_test$ivector_dim
sre_core_10sec_train_path=exp/ivectors_core_10sec_train$ivector_dim
sre_8conv_10sec_test_path=exp/ivectors_8conv_10sec_test$ivector_dim
sre_8conv_10sec_train_path=exp/ivectors_8conv_10sec_train$ivector_dim
sre_8conv_core_test_path=exp/ivectors_8conv_core_test$ivector_dim
sre_8conv_core_train_path=exp/ivectors_8conv_core_train$ivector_dim
sre_cc_test_path=exp/ivectors_cc_test$ivector_dim
sre_cc_train_path=exp/ivectors_cc_train$ivector_dim


if [ $stage -le 5 ]; then

  # # Train an i-vector extractor based on the DNN-UBM.
  # sid/train_ivector_extractor_dnn.sh \
  #   --cmd "$train_cmd --mem 100G" --nnet-job-opt "--mem 4G" \
  #   --min-post 0.015 --ivector-dim $ivector_dim --num-iters 5 \
  #   exp/full_ubm/final.ubm $nnet \
  #   data/train \
  #   data/train_dnn \
  #   exp/extractor_dnn$ivector_dim

  nj_num=8

  # 10sec core-10sec 8conv-10sec
  # for f in sre_test sre_train sre \
  #     sre_10sec_test sre_10sec_train \
  #     sre_core_10sec_test sre_core_10sec_train \
  #     sre_8conv_10sec_test sre_8conv_10sec_train 
  for f in sre_cc_test sre_cc_train ; do 

	eval iv_dir=$(echo '$'$"${f}_path")
	
	mkdir -p $iv_dir
	

	# Extract i-vectors. and cal the spk_mean with length norm
	sid/extract_ivectors.sh \
		--cmd "$train_cmd --mem 6G" --nj $nj_num \
		exp/extractor data/$f \
	  	$iv_dir

  done

fi



if [ $stage -le 6 ]; then

  lda_dim=$([ 200 -le $ivector_dim ] && echo "200" || echo "$ivector_dim")

  # # This script uses LDA to decrease the dimensionality 
  # local/lda_scoring.sh --lda-dim $lda_dim --use-existing-models true \
  #            --covar_factor 0.0 \
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_lda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lda$ivector_dim

  # # This script uses NDA to decrease the dimensionality 
  # local/nda_scoring.sh --lda-dim $lda_dim --use-existing-models true \
  #            --covar_factor 0.0 \
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_nda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_nda$ivector_dim

  # # This script uses LpLDA to decrease the dimensionality 
  # local/lplda_scoring.sh --lda-dim $lda_dim --use-existing-models true \
  #            --covar_factor 0.0 \
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_lplda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lplda$ivector_dim


  # This script uses LCLDA to decrease the dimnsionality 
  local/lclda_scoring.sh --lda-dim $lda_dim --beta 0.0 --use-existing-models true \
        --covar_factor 0.0 \
      data/sre data/sre_cc_train data/sre_cc_test \
      $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
      exp/ivector_score/cc_lclda$ivector_dim

  local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lclda$ivector_dim


  # # This script uses LDA and LCLDA to decrease the dimnsionality 
  # local/lda_lclda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models true \
  #         --covar_factor 0.0 \
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_lda_lclda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lda_lclda$ivector_dim



  # This script uses LcpLDA to decrease the dimensionality 
  local/lcplda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models true \
             --covar_factor 0.0 \
      data/sre data/sre_cc_train data/sre_cc_test \
      $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
      exp/ivector_score/cc_lcplda$ivector_dim

  local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lcplda$ivector_dim


  # # This script uses LDA and LcpLDA to decrease the dimensionality 
  # local/lda_lcplda_scoring.sh --lda-dim $lda_dim --beta 0.0 --use-existing-models true \
  #            --covar_factor 0.0 \
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_lda_lcplda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lda_lcplda$ivector_dim

fi

# exit;
# add plda here 
if [ $stage -le 7 ]; then

  lda_dim=$([ 200 -le $ivector_dim ] && echo "200" || echo "$ivector_dim")

  # # This script uses LDA and PLDA to decrease the dimensionality 
  # local/lda_plda_scoring.sh --lda-dim $lda_dim --use-existing-models true \
  #            --covar_factor 0.0 \
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_lda_plda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lda_plda$ivector_dim


  # # This script uses NDA and PLDA to decrease the dimensionality 
  # local/nda_plda_scoring.sh --lda-dim $lda_dim --use-existing-models true \
  #            --covar_factor 0.0 \
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_nda_plda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_nda_plda$ivector_dim

  # # This script uses LpLDA and PLDA to decrease the dimensionality 
  # local/lplda_plda_scoring.sh --lda-dim $lda_dim --use-existing-models true \
  #            --covar_factor 0.0 \
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_lplda_plda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lplda_plda$ivector_dim


  # This script uses LCLDA and PLDA to decrease the dimensionality 
  local/lclda_plda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models true \
        --covar_factor 0.0 \
      data/sre data/sre_cc_train data/sre_cc_test \
      $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
      exp/ivector_score/cc_lclda_plda$ivector_dim

  local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lclda_plda$ivector_dim


  # # # This script uses LCLDA and PLDA to decrease the dimensionality 
  # local/lda_lclda_plda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models true \
  #       --covar_factor 0.0 --rest_dim 50\
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_lda_lclda_plda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lda_lclda_plda$ivector_dim


  # This script uses LCpLDA and PLDA to decrease the dimensionality 
  local/lcplda_plda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models true \
        --covar_factor 0.0 \
      data/sre data/sre_cc_train data/sre_cc_test \
      $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
      exp/ivector_score/cc_lcplda_plda$ivector_dim

  local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lcplda_plda$ivector_dim


  # # # This script uses LCpLDA and PLDA to decrease the dimensionality 
  # local/lda_lcplda_plda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models true \
  #       --covar_factor 0.0 --rest_dim 50\
  #     data/sre data/sre_cc_train data/sre_cc_test \
  #     $sre_path $sre_cc_train_path $sre_cc_test_path $trials_cc \
  #     exp/ivector_score/cc_lda_lcplda_plda$ivector_dim

  # local/score_recording.sh --lda-dim $lda_dim $trials_cc exp/ivector_score/cc_lda_lcplda_plda$ivector_dim


fi



# local/lda_scoring.sh 
# total: 3.814 , 0.0005038.  --lda :200
# local/nda_scoring.sh 
# total: 2.966 , 0.0004329.  --lda :200
# local/lplda_scoring.sh 
# total: 3.107 , 0.0004193.  --lda :200
# local/lclda_scoring.sh 
# total: 3.39 , 0.0004729.  --lda :200
# local/lcplda_scoring.sh 
# total: 2.825 , 0.0004289.  --lda :200


# total: 3.39 , 0.0004645.  --lda :200
# total: 3.249 , 0.0004106.  --lda :200


# local/lda_plda_scoring.sh 
# total: 1.836 , 0.000367.  --lda :200
# local/nda_plda_scoring.sh -
# total: 1.836 , 0.0003061.  --lda :200
# local/lplda_plda_scoring.sh -
# total: 1.554 , 0.0002837.  --lda :200
# local/lclda_plda_scoring.sh 
# total: 1.554 , 0.0003442.  --lda :200
# local/lcplda_plda_scoring.sh 
# total: 1.412 , 0.0002667.  --lda :200

# local/lclda_plda_scoring.sh 
# total: 1.412 , 0.0003327.  --lda :200
# local/lcplda_plda_scoring.sh 
# total: 1.554 , 0.0002794.  --lda :200