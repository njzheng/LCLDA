// subivectorbin/subivector-mean.cc

// Copyright 2013-2014  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
// modified by z3c
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "With 3  arguments, averages sub iVectors over all the\n"
        "utterances of each speaker using the spk2utt file.\n"
        "Input the spk2utt file and a set of subiVectors indexed by\n"
        "utterance; output is subiVectors indexed by speaker.  If 4\n"
        "arguments are given, extra argument is a table for the number\n"
        "of utterances per speaker (can be useful for PLDA).\n"
        "\n"
        "Usage: ivector-mean-var <spk2utt-rspecifier> <ivector-rspecifier> "
        "<ivector-wspecifier>\n"
        "e.g.: ivector-mean-var data/spk2utt exp/ivectors.ark exp/spk_ivectors.ark \n"
        "See also: ivector-mean\n";

    ParseOptions po(usage);


    po.Read(argc, argv);

    if (po.NumArgs() != 3 ) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(1),
        ivector_rspecifier = po.GetArg(2),
        ivector_wspecifier = po.GetArg(3);


    // double spk_sumsq = 0.0;
    Vector<BaseFloat> spk_sum;

    int64 num_spk_done = 0, num_spk_err = 0,
        num_utt_done = 0, num_utt_err = 0;

    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    BaseFloatMatrixWriter ivector_mean_var_writer(ivector_wspecifier);

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      // different utterances
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      if (uttlist.empty()) {
        KALDI_ERR << "Speaker with no utterances.";
      }

      Vector<BaseFloat> spk_mean;
      Vector<BaseFloat> spk_mean2;
      Vector<BaseFloat> spk_var;

      // the number of utt for spk
      // int32 utt_num = uttlist.size();

      int32 utt_count = 0;
      // discard the norm == 0 ones
      int32 utt_count_valid = 0;

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No subiVector present in input for utterance " << utt;
          num_utt_err++;
        } else {
          if (utt_count == 0) {
            spk_mean = ivector_reader.Value(utt);
            spk_mean2.Resize(spk_mean.Dim());
            spk_mean2.AddVec2(1.0, ivector_reader.Value(utt));
            spk_var.Resize(spk_mean.Dim());
          } else {
            spk_mean.AddVec(1.0, ivector_reader.Value(utt));
            spk_mean2.AddVec2(1.0, ivector_reader.Value(utt)); 
          }
          num_utt_done++;
          utt_count++;

          // since vector has already normalized to sqrt(dim)
          if (ivector_reader.Value(utt).Norm(2.0) > 0.01)
            utt_count_valid ++;
          else
            KALDI_LOG << utt << "has zero norm value. ";
        }
      }

      if (utt_count_valid == 0) {
        KALDI_WARN << "Not producing valid output for speaker " << spk
                   << " with utt_count" << utt_count
                   << " since some utterances had iVectors with zero norm";
        num_spk_err++;
      } else {
        spk_mean.Scale(1.0 / utt_count_valid);
        if (utt_count_valid == 1){
          spk_mean2.Resize(spk_mean.Dim());
          spk_var.Set(1.0);
        }
        else{
          spk_mean2.Scale(1.0 / (utt_count_valid-1));
          // cov = e(x^2)-e(x)^2
          spk_var.AddVec(1.0, spk_mean2);
          spk_var.AddVec2(-1.0, spk_mean);

        }


        Matrix<BaseFloat> mean_var_mat(2,spk_var.Dim());
        mean_var_mat.CopyRowFromVec(spk_mean,0);
        mean_var_mat.CopyRowFromVec(spk_var,1);
        
        ivector_mean_var_writer.Write(spk, mean_var_mat);

        num_spk_done++;
      }
    }

    KALDI_LOG << "Computed mean of " << num_spk_done << " speakers ("
              << num_spk_err << " with no utterances), consisting of "
              << num_utt_done << " utterances (" << num_utt_err
              << " absent from input).";

    // if (num_spk_done != 0) {
    //   spk_sumsq /= num_spk_done;
    //   spk_sum.Scale(1.0 / num_spk_done);
    //   double mean_length = spk_sum.Norm(2.0),
    //       spk_length = sqrt(spk_sumsq),
    //       norm_spk_length = spk_length / sqrt(spk_sum.Dim());
    //   KALDI_LOG << "Norm of mean of speakers is " << mean_length
    //             << ", root-mean-square speaker-iVector length divided by "
    //             << "sqrt(dim) is " << norm_spk_length;
    // }

    return (num_spk_done != 0 ? 0 : 1);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
