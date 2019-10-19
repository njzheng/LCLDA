# Copyright 2015   David Snyder
# Apache 2.0.
#
# Given a trials and scores file, this script 
# prepares input for the binary compute-eer. 
import sys

trials_file = open(sys.argv[1], 'r')
scores_file = open(sys.argv[2], 'r')
trials = trials_file.readlines()
scores = scores_file.readlines()
spkrutt2target = {}
for line in trials:
  spkr, utt, target = line.strip().split()
  spkrutt2target[spkr+utt]=target

trials_file.close()

for line in scores:
  spkr, utt, score = line.strip().split()
  print(score, spkrutt2target[spkr+utt])

scores_file.close()
