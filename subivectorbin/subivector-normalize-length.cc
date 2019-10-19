// subivectorbin/subivector-normalize-length.cc

// Copyright 2013  Daniel Povey

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
#include "gmm/am-diag-gmm.h"
#include "subivector/subivector-extractor.h"
#include "util/kaldi-thread.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Normalize length of sub iVectors to equal sqrt(feature-dimension)\n"
        "\n"
        "Usage:  subivector-normalize-length [options] <subivector-rspecifier> "
        "<subivector-wspecifier>\n"
        "e.g.: \n"
        " subivector-normalize-length ark:subivectors.ark ark:normalized_subivectors.ark\n";

    ParseOptions po(usage);
    bool normalize = true;

    po.Register("normalize", &normalize,
                "Set this to false to disable normalization");

    bool scaleup = true;
    po.Register("scaleup", &scaleup,
                "If 'true', the normalized iVector is scaled-up by 'sqrt(dim)'");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string subivector_rspecifier = po.GetArg(1),
        subivector_wspecifier = po.GetArg(2);


    int32 num_done = 0;

    // double tot_ratio = 0.0, tot_ratio2 = 0.0;
    Vector<double> tot_ratio_vec;
    // Vector<double> tot_ratio2_vec(ivector_dim);
    
    // SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    // BaseFloatVectorWriter ivector_writer(ivector_wspecifier);
    SequentialBaseFloatMatrixReader subivector_reader(subivector_rspecifier);
    BaseFloatMatrixWriter subivector_writer(subivector_wspecifier);


    for (; !subivector_reader.Done(); subivector_reader.Next()) {
      std::string key = subivector_reader.Key();
      Matrix<BaseFloat> subivector = subivector_reader.Value();

      int32 row_num = subivector.NumRows();
      int32 col_num = subivector.NumCols();
      Vector<double> ratio_vec(row_num);

      for(int32 i=0; i<row_num; i++){
        SubVector<BaseFloat> subivector_row(subivector, i);


        BaseFloat norm = subivector_row.Norm(2.0);
        BaseFloat ratio = norm / sqrt(col_num); // how much larger it is
                                                      // than it would be, in
                                                      // expectation, if normally
        if (!scaleup) ratio = norm;

        if (ratio == 0.0) {
          KALDI_WARN << "Zero subiVector for row " << i << " in " << key;
        } else {
          if (normalize) subivector_row.Scale(1.0 / ratio);
        }
        ratio_vec(i) = ratio;
        // tot_ratio += ratio;
        // tot_ratio2 += ratio * ratio;
        
      }
      KALDI_VLOG(2) << "Ratio for key " << key << " is " << ratio_vec;
      if(tot_ratio_vec.Dim()==0)
        tot_ratio_vec.Resize(row_num);
      tot_ratio_vec.AddVec(1.0,ratio_vec);
      num_done++;

      subivector_writer.Write(key, subivector);
    }

    KALDI_LOG << "Processed " << num_done << " subiVectors.";
    if (num_done != 0) {

      // BaseFloat avg_ratio = tot_ratio / num_done,
      //     ratio_stddev = sqrt(tot_ratio2 / num_done - avg_ratio * avg_ratio);
      // KALDI_LOG << "Average ratio of iVector to expected length was "
      //           << avg_ratio << ", standard deviation was " << ratio_stddev;

      // tot becomes avg
      tot_ratio_vec.Scale(1.0/num_done);
      KALDI_LOG << "Average ratio of sub iVector to expected length was "
                << tot_ratio_vec ;

    }
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
