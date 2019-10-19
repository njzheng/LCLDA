#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# See README.txt for more info on data required.

#set -e

audio_dir=$1
data_dir=$2

if [ ! -f data/local/speaker_list ]; then
	wget -P data/local/ http://www.openslr.org/resources/15/speaker_list.tgz
	tar -C data/local/ -xvf data/local/speaker_list.tgz
fi

cp data/local/speaker_list data/local/speaker_list_orgin

speaker_list_new=data/local/speaker_list

# remove the lines including sre2004 sre2005 sre2006 
sed '/sre2004/d' $speaker_list_new
sed '/sre2005/d' $speaker_list_new
sed '/sre2006/d' $speaker_list_new

sre_ref=data/local/speaker_list

local/make_sre.pl $audio_dir/SRE04 sre2004 $sre_ref $data_dir/sre2004

local/make_sre.pl $audio_dir/SRE05 sre2005 $sre_ref $data_dir/sre2005_train

local/make_sre.pl $audio_dir/SRE05 sre2005 $sre_ref $data_dir/sre2005_test

local/make_sre.pl $audio_dir/SRE06 sre2006 $sre_ref $data_dir/sre2006_train

local/make_sre.pl $audio_dir/SRE06 sre2006 $sre_ref $data_dir/sre2006_test_1

local/make_sre.pl $audio_dir/SRE06 sre2006 $sre_ref $data_dir/sre2006_test_2

local/make_sre.pl $audio_dir/SRE08 sre2008 $sre_ref $data_dir/sre2008_train

local/make_sre.pl $audio_dir/SRE08 sre2008 $sre_ref $data_dir/sre2008_test

utils/combine_data.sh $data_dir/sre \
  $data_dir/sre2004 $data_dir/sre2005_train \
  $data_dir/sre2005_test $data_dir/sre2006_train \
  $data_dir/sre2006_test_1 $data_dir/sre2006_test_2 \
  $data_dir/sre2008_train $data_dir/sre2008_test 

utils/validate_data_dir.sh --no-text --no-feats $data_dir/sre
utils/fix_data_dir.sh $data_dir/sre

[ -e data/local/speaker_list.* ] && rm data/local/speaker_list.*

echo "sre_pre read ok!"
echo "read sre2004~2008!"
