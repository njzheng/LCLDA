#!/bin/bash
# Copyright 2015-2017   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

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
trials_8conv_core=data/sre_8conv_core_test/trials
num_components=2048 # Larger than this doesn't make much of a difference.

stage=7
ivector_dim=600


if [ $stage -le 0 ]; then
	echo "--read sre data!"
	# Prepare the SRE 2010 evaluation data.
	sre_path=/scratch/njzheng/DATA/sre2010/LDC2012E09/SRE10
	local/make_sre_2010_test.pl $sre_path/eval data
	local/make_sre_2010_train.pl $sre_path/eval data

	echo "--read sre data prior to 2010!"
	# Prepare a collection of NIST SRE data prior to 2010. This is
	# used to train the PLDA model and is also combined with SWB
	# for UBM and i-vector extractor training data.
	pre_sre_path=/scratch/njzheng/DATA/sre/LDC2009E100
	local/make_sre.sh $pre_sre_path data

	echo "--read swbd data!"
	# Prepare SWB for UBM and i-vector extractor training.
	swbd_path=/scratch/njzheng/DATA/swbd/LDC2016E81
	# add a new part training set: swbd2 phase 1
	local/make_swbd2_phase1.pl $swbd_path/LDC98S75 \
		data/swbd2_phase1_train
	local/make_swbd2_phase2.pl $swbd_path/LDC99S79 \
		data/swbd2_phase2_train
	local/make_swbd2_phase3.pl $swbd_path/LDC2002S06 \
		data/swbd2_phase3_train
	local/make_swbd_cellular1.pl $swbd_path/LDC2001S13 \
		data/swbd_cellular1_train
	local/make_swbd_cellular2.pl $swbd_path/LDC2004S07 \
		data/swbd_cellular2_train

	echo "--combine the swbd and sre in data/train!"
	# put them into data/train directory
	utils/combine_data.sh data/train \
		data/swbd_cellular1_train data/swbd_cellular2_train \
		data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train \
		data/sre
fi
	


if [ $stage -le 1 ]; then
	# extract the MFCC features

	steps/make_mfcc.sh --mfcc-config conf/mfcc13.conf --nj 20 --cmd "$train_cmd" \
		data/train exp/make_mfcc $mfccdir
	
	# actually train has include sre

	steps/make_mfcc.sh --mfcc-config conf/mfcc13.conf --nj 20 --cmd "$train_cmd" \
		data/sre exp/make_mfcc $mfccdir
	
	steps/make_mfcc.sh --mfcc-config conf/mfcc13.conf --nj 20 --cmd "$train_cmd" \
		data/sre10_train exp/make_mfcc $mfccdir
	steps/make_mfcc.sh --mfcc-config conf/mfcc13.conf --nj 20 --cmd "$train_cmd" \
		data/sre10_test exp/make_mfcc $mfccdir

	for name in sre sre10_train sre10_test train; 
	do 
		utils/fix_data_dir.sh data/${name}
	done

	mkdir -p data/sre_test
	mkdir -p data/sre_train
	cp -r data/sre10_test/* data/sre_test
	cp -r data/sre10_train/* data/sre_train

fi


if [ $stage -le 3 ]; then

	sid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" \
		data/train exp/make_vad $vaddir
	sid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" \
		data/sre exp/make_vad $vaddir
	sid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" \
		data/sre10_train exp/make_vad $vaddir
	sid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" \
		data/sre10_test exp/make_vad $vaddir

	for name in sre sre10_train sre10_test train; do
		utils/fix_data_dir.sh data/${name}
	done
fi


if [ $stage -le 4 ]; then
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
	# sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 15G" \
	# --ivector-dim $ivector_dim \
	# --num-iters 5  \
	# --stage -4 exp/full_ubm_$num_components/final.ubm data/train \
	# exp/extractor

	nj_num=20
	# 10sec core-10sec 8conv-10sec
	
	for f in  sre sre_test  sre_train ; do 
			# sre_10sec_test sre_10sec_train \
			# sre_core_10sec_test sre_core_10sec_train \
			# sre_8conv_10sec_test sre_8conv_10sec_train \
			# sre_8conv_core_test sre_8conv_core_train  ; do

		eval iv_dir=$(echo '$'$"${f}_path")
		
		mkdir -p $iv_dir
		

		# Extract i-vectors. and cal the spk_mean with length norm
		sid/extract_ivectors.sh \
			--cmd "$train_cmd --mem 6G" --nj $nj_num \
			exp/extractor data/$f \
		  	$iv_dir

	done

  	# Compute the mean vector for centering the evaluation i-vectors.
  	$train_cmd $sre_path/log/compute_mean.log \
    	ivector-mean scp:$sre_path/ivector_org.scp $sre_path/mean.vec || exit 1;

fi

# do not do length normalization and mean subscribe before LDA


if [ $stage -le 6 ]; then

	lda_dim=$([ 200 -le $ivector_dim ] && echo "200" || echo "$ivector_dim")

	# # This script uses LDA to decrease the dimensionality 
	# local/lda_scoring.sh --lda-dim $lda_dim --use-existing-models true \
	# 					 --covar_factor 0.0 \
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_lda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lda$ivector_dim

	# # This script uses NDA to decrease the dimensionality 
	# local/nda_scoring.sh --lda-dim $lda_dim --use-existing-models true \
	# 					 --covar_factor 0.0 \
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_nda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_nda$ivector_dim

	# # This script uses LpLDA to decrease the dimensionality 
	# local/lplda_scoring.sh --lda-dim $lda_dim --use-existing-models true \
	# 					 --covar_factor 0.0 \
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_lplda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lplda$ivector_dim


	# This script uses LCLDA to decrease the dimnsionality 
	local/lclda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models true \
				--covar_factor 0.0 \
			data/sre data/sre_train data/sre_test \
			$sre_path $sre_train_path $sre_test_path $trials \
			exp/ivector_score/core_lclda$ivector_dim

	local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lclda$ivector_dim


	# # This script uses LDA and LCLDA to decrease the dimnsionality 
	# local/lda_lclda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models true \
	# 				--covar_factor 0.0 \
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_lda_lclda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lda_lclda$ivector_dim


	# This script uses LcpLDA to decrease the dimensionality 
	local/lcplda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models true \
						 --covar_factor 0.0 \
			data/sre data/sre_train data/sre_test \
			$sre_path $sre_train_path $sre_test_path $trials \
			exp/ivector_score/core_lcplda$ivector_dim

	local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lcplda$ivector_dim


	# # This script uses LDA and LcpLDA to decrease the dimensionality 
	# local/lda_lcplda_scoring.sh --lda-dim $lda_dim --beta 0.0 --use-existing-models true \
	# 					 --covar_factor 0.0 \
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_lda_lcplda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lda_lcplda$ivector_dim
exit;
fi




# add plda here 
if [ $stage -le 7 ]; then

	lda_dim=$([ 200 -le $ivector_dim ] && echo "200" || echo "$ivector_dim")

	# # This script uses LDA and PLDA to decrease the dimensionality 
	# local/lda_plda_scoring.sh --lda-dim $lda_dim --use-existing-models false \
	# 					 --covar_factor 0.0 \
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_lda_plda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lda_plda$ivector_dim


	# # This script uses NDA and PLDA to decrease the dimensionality 
	# local/nda_plda_scoring.sh --lda-dim $lda_dim --use-existing-models false \
	# 					 --covar_factor 0.0 \
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_nda_plda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_nda_plda$ivector_dim



	# # This script uses LpLDA and PLDA to decrease the dimensionality 
	# local/lplda_plda_scoring.sh --lda-dim $lda_dim --use-existing-models false \
	# 					 --covar_factor 0.0 \
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_lplda_plda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lplda_plda$ivector_dim


	# This script uses LCLDA and PLDA to decrease the dimensionality 
	local/lclda_plda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models false \
				--covar_factor 0.0 \
			data/sre data/sre_train data/sre_test \
			$sre_path $sre_train_path $sre_test_path $trials \
			exp/ivector_score/core_lclda_plda$ivector_dim

	local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lclda_plda$ivector_dim


	# # # This script uses LCLDA and PLDA to decrease the dimensionality 
	# local/lda_lclda_plda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models false \
	# 			--covar_factor 0.0 --rest_dim 50\
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_lda_lclda_plda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lda_lclda_plda$ivector_dim


	# This script uses LCpLDA and PLDA to decrease the dimensionality 
	local/lcplda_plda_scoring.sh --lda-dim $lda_dim --beta 6.0 --use-existing-models false \
				--covar_factor 0.0 \
			data/sre data/sre_train data/sre_test \
			$sre_path $sre_train_path $sre_test_path $trials \
			exp/ivector_score/core_lcplda_plda$ivector_dim

	local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lcplda_plda$ivector_dim


	# # # This script uses LCpLDA and PLDA to decrease the dimensionality 
	# local/lda_lcplda_plda_scoring.sh --lda-dim $lda_dim --beta 0.0 --use-existing-models false \
	# 			--covar_factor 0.0 --rest_dim 50\
	# 		data/sre data/sre_train data/sre_test \
	# 		$sre_path $sre_train_path $sre_test_path $trials \
	# 		exp/ivector_score/core_lda_lcplda_plda$ivector_dim

	# local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_lda_lcplda_plda$ivector_dim


fi

# --lda :200
# local/lda_scoring.sh 
# female: 3.186 , 0.0004677; male: 3.752 , 0.0005219
# total: 3.473 , 0.0005116.  --lda :200
# local/nda_scoring.sh --lda-dim 200 
# female: 2.889 , 0.0004005; male: 3.348 , 0.0004474
# total: 3.111 , 0.0004304.  --lda :200
# local/lplda_scoring.sh --lda-dim 200 
# female: 2.808 , 0.0003805; male: 2.944 , 0.0004311
# total: 2.86 , 0.0004239.  --lda :200
# local/lclda_scoring.sh --lda-dim 200 
# female: 2.862 , 0.0004436; male: 3.492 , 0.0004765
# total: 3.083 , 0.0004545.  --lda :200
# local/lcplda_scoring.sh --lda-dim 200
# female: 2.646 , 0.000349; male: 2.915 , 0.0004024
# total: 2.748 , 0.0003934.  --lda :200

# local/lclda_scoring.sh --lda-dim 200 
# female: 2.97 , 0.0004471; male: 3.521 , 0.0004721
# total: 3.25 , 0.0004735.  --lda :200
# local/lcplda_scoring.sh --lda-dim 200
# female: 2.673 , 0.0003562; male: 2.944 , 0.0003877
# total: 2.748 , 0.0003948.  --lda :200


# local/lda_plda_scoring.sh 
# female: 1.701 , 0.0003474; male: 2.078 , 0.0003842
# total: 1.855 , 0.0003854.  --lda :200
# local/nda_plda_scoring.sh 
# female: 1.674 , 0.0002649; male: 1.847 , 0.0002989
# total: 1.758 , 0.0002867.  --lda :200
# local/lplda_plda_scoring.sh 
# female: 1.593 , 0.0002758; male: 1.674 , 0.00029
# total: 1.646 , 0.0002908.  --lda :200
# local/lclda_plda_scoring.sh
# female: 1.593 , 0.000307; male: 1.789 , 0.0003201
# total: 1.674 , 0.0003122.  --lda :200
# local/lcplda_plda_scoring.sh -
# female: 1.485 , 0.0002417; male: 1.443 , 0.0002595
# total: 1.451 , 0.0002639.  --lda :200

# local/lclda_plda_scoring.sh
# female: 1.539 , 0.0003089; male: 1.76 , 0.0003185
# total: 1.604 , 0.0003215.  --lda :200
# local/lcplda_plda_scoring.sh -
# female: 1.485 , 0.0002505; male: 1.443 , 0.0002644
# total: 1.451 , 0.0002704.  --lda :200

exit;


if [ $stage -le 9 ]; then

	lda_dim=$([ 200 -le $ivector_dim ] && echo "150" || echo "$ivector_dim")
	 

	# Separate the i-vectors into male and female partitions and calculate
	# i-vector means used by the scoring scripts.
	local/scoring_common.sh data/sre data/sre_train data/sre_test \
	  exp/ivectors_sre${ivector_dim} exp/ivectors_sre10_train${ivector_dim} \
	  exp/ivectors_sre10_test${ivector_dim}


	# Create a gender independent PLDA model and do scoring.
	local/plda_scoring.sh data/sre data/sre10_train data/sre10_test \
	  exp/ivectors_sre$ivector_dim exp/ivectors_sre10_train$ivector_dim \
	  $sre_test_path $trials exp/scores_gmm_2048_ind_pooled
	local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_female data/sre10_test_female \
	  exp/ivectors_sre$ivector_dim exp/ivectors_sre10_train${ivector_dim}_female exp/ivectors_sre10_test${ivector_dim}_female $trials_female exp/scores_gmm_2048_ind_female
	local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_male data/sre10_test_male \
	  exp/ivectors_sre$ivector_dim exp/ivectors_sre10_train${ivector_dim}_male exp/ivectors_sre10_test${ivector_dim}_male $trials_male exp/scores_gmm_2048_ind_male

	# Create gender dependent PLDA models and do scoring.
	local/plda_scoring.sh data/sre_female data/sre10_train_female data/sre10_test_female \
	  exp/ivectors_sre$ivector_dim exp/ivectors_sre10_train${ivector_dim}_female exp/ivectors_sre10_test${ivector_dim}_female $trials_female exp/scores_gmm_2048_dep_female
	local/plda_scoring.sh data/sre_male data/sre10_train_male data/sre10_test_male \
	  exp/ivectors_sre$ivector_dim exp/ivectors_sre10_train${ivector_dim}_male exp/ivectors_sre10_test${ivector_dim}_male $trials_male exp/scores_gmm_2048_dep_male

	# Pool the gender dependent results.
	mkdir -p exp/scores_gmm_2048_dep_pooled
	cat exp/scores_gmm_2048_dep_male/plda_scores exp/scores_gmm_2048_dep_female/plda_scores \
	  > exp/scores_gmm_2048_dep_pooled/plda_scores


fi


# GMM-2048 PLDA EER
# ind pooled: 2.26
# ind female: 2.33
# ind male:   2.05
# dep female: 2.30
# dep male:   1.59
# dep pooled: 2.00
echo "GMM-$num_components EER"
for x in ind dep; do
  for y in female male pooled; do
    eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores_gmm_${num_components}_${x}_${y}/plda_scores) 2> /dev/null`
    echo "${x} ${y}: $eer"
  done
done



# GMM-2048 PLDA EER 20MFCC 
# ind pooled: 2.26
# ind female: 2.33
# ind male:   2.05
# dep female: 2.30
# dep male:   1.59
# dep pooled: 2.00

# GMM-2048 EER 13MFCC 150
# ind female: 1.917
# ind male: 1.962
# ind pooled: 1.925
# dep female: 2.025
# dep male: 1.674
# dep pooled: 1.855


# echo "GMM-$num_components EER"
# for x in ind dep; do
#   for y in female male pooled; do
#     eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores_gmm_${num_components}_${x}_${y}/plda_scores) 2> /dev/null`
#     echo "${x} ${y}: $eer"
#   done
# done


# wish to use single linear space but failed!
# if [ $stage -le 7 ]; then

# 	lda_dim=$([ 200 -le $ivector_dim ] && echo "150" || echo "$ivector_dim")

# 	# This script uses CLDA to decrease the dimnsionality 
# 	local/clda_scoring.sh --lda-dim $lda_dim --beta 0.0 --use-existing-models false --covar_factor 0.1 \
# 			data/sre data/sre_train data/sre_test \
# 			$sre_path $sre_train_path $sre_test_path $trials \
# 			exp/ivector_score/core_clda$ivector_dim

# 	local/score_recording.sh --lda-dim $lda_dim $trials $trials_female $trials_male exp/ivector_score/core_clda$ivector_dim


# fi