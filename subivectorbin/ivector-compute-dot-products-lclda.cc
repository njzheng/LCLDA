// ivectorbin/ivector-compute-dot-products.cc

// Copyright 2013  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
//
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
#include "ivector/ivector-extractor.h"
#include "util/kaldi-thread.h"


namespace kaldi {

BaseFloat FindSpk(
  const Matrix<BaseFloat> & spk_mean,
  const Vector<BaseFloat> &ivector,
  int32 &spk){

  Vector<BaseFloat> cosine_dis(spk_mean.NumRows());

  // KALDI_LOG<< "size of spk_mean is " <<spk_mean.NumRows()<<" "<<spk_mean.NumCols();
  // KALDI_LOG<< "size of ivector is " <<ivector.Dim();

  cosine_dis.AddMatVec(1.0, spk_mean, kNoTrans, ivector, 0.0);
  cosine_dis(spk_mean.NumRows()-1) = 0.0; //since the last is tot_spk_mean, avoid to select the last one

  int32 indx = -1;
  BaseFloat cos_dis =  cosine_dis.Max(&indx);
  cos_dis /= spk_mean.Row(indx).Norm(2.0) * ivector.Norm(2.0);
  spk = indx;

  return cos_dis;
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Computes dot-products between iVectors; useful in application of an\n"
        "iVector-based system.  The 'trials-file' has lines of the form\n"
        "<key1> <key2>\n"
        "and the output will have the form\n"
        "<key1> <key2> [<dot-product>]\n"
        "(if either key could not be found, the dot-product field in the output\n"
        "will be absent, and this program will print a warning)\n"
        "\n"
        "Usage:  ivector-compute-dot-products [options] <trials-in> "
        "<ivector1-rspecifier> <ivector2-rspecifier> \n"
        "<lclda-transform> <spk-mean> <scores-out>\n"
        "e.g.: \n"
        " ivector-compute-dot-products trials scp:train_ivectors.scp scp:test_ivectors.scp \n"
        "ark:transform.ark, ark:ivector-normalize-length ark:spk_mean.ark ark: -| \n"
        "trials.scored\n"
        
        "See also: ivector-plda-scoring\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string trials_rxfilename = po.GetArg(1),
        ivector1_rspecifier = po.GetArg(2),
        ivector2_rspecifier = po.GetArg(3),
        transform_rspecifier = po.GetArg(4),
        spk_mean_rspecifier = po.GetArg(5),
        scores_wxfilename = po.GetArg(6);


    int64 num_done = 0, num_err = 0;

    RandomAccessBaseFloatVectorReader ivector1_reader(ivector1_rspecifier);
    RandomAccessBaseFloatVectorReader ivector2_reader(ivector2_rspecifier);
    RandomAccessBaseFloatMatrixReader transform_reader(transform_rspecifier);
    SequentialBaseFloatVectorReader spk_mean_reader(spk_mean_rspecifier);

    // read the spk mean information
    std::vector< std::string > spk_list;
    std::vector< Vector<BaseFloat> > spk_mean_temp;
    for (; !spk_mean_reader.Done(); spk_mean_reader.Next()) {
      spk_list.push_back(spk_mean_reader.Key());
      spk_mean_temp.push_back(spk_mean_reader.Value());
    }
    Matrix<BaseFloat> spk_mean(spk_mean_temp.size(), spk_mean_temp[0].Dim());
    for(int32 i=0; i<spk_mean_temp.size(); i++){
      spk_mean.CopyRowFromVec(spk_mean_temp[i], i);
    }



    Input ki(trials_rxfilename);

    bool binary = false;
    Output ko(scores_wxfilename, binary);
    double sum = 0.0, sumsq = 0.0;

    std::string line;
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 2) {
        KALDI_ERR << "Bad line " << (num_done + num_err) << " in input "
                  << "(expected two fields: key1 key2): " << line;
      }
      std::string key1 = fields[0], key2 = fields[1];
      if (!ivector1_reader.HasKey(key1)) {
        KALDI_WARN << "Key " << key1 << " not present in 1st table of ivectors.";
        num_err++;
        continue;
      }
      if (!ivector2_reader.HasKey(key2)) {
        KALDI_WARN << "Key " << key2 << " not present in 2nd table of ivectors.";
        num_err++;
        continue;
      }
      const Vector<BaseFloat> &ivector1 = ivector1_reader.Value(key1),
          &ivector2 = ivector2_reader.Value(key2);

      if(spk_mean.NumCols()!=ivector1.Dim()){
        KALDI_LOG<<spk_mean.NumCols()<<";"<<ivector1.Dim();
      }  

      // find the nearest spker
      int32 spk_indx1;
      int32 spk_indx2;
      BaseFloat dis1 = FindSpk(spk_mean, ivector1, spk_indx1);
      BaseFloat dis2 = FindSpk(spk_mean, ivector2, spk_indx2);

      // Vector<BaseFloat> ivector_mid(ivector1);
      // ivector_mid.AddVec(1.0, ivector2);
      // ivector_mid.Scale(0.5);
      // BaseFloat dis_mid = FindSpk(spk_mean, ivector_mid, spk_indx1); // replace T1


      // std::string spk_list[spk_indx1]
      Matrix<BaseFloat> T1(transform_reader.Value(spk_list[spk_indx1]));
      Matrix<BaseFloat> T2(transform_reader.Value(spk_list[spk_indx2]));
      // Matrix<BaseFloat> T1(T2);
      
      // Matrix<BaseFloat> T1(transform_reader.Value("LDA"));
      // Matrix<BaseFloat> T2(transform_reader.Value("LDA"));      

      // KALDI_LOG << "(key1, spk1) is " << key1 << " "<< spk_list[spk_indx1]
      //           << " with cos_distance " << dis1 
      //           << ", (key2 spk2) is " << key2 << " "<< spk_list[spk_indx2]
      //           << " with cos_distance " << dis2 ;

      KALDI_LOG << "(key1, spk1) is " << key1 << " "<< spk_list[spk_indx1]
                << ", (key2 spk2) is " << key2 << " "<< spk_list[spk_indx2]
                << " with cos_distance " 
                << VecVec(ivector1, ivector2)/(ivector1.Norm(2.0)*ivector2.Norm(2.0)) ;


      int32 lda_dim = T1.NumRows();

      Vector<BaseFloat> proj_ivector11(lda_dim), proj_ivector12(lda_dim),
                        proj_ivector21(lda_dim), proj_ivector22(lda_dim);
      
      proj_ivector11.AddMatVec(1.0, T1, kNoTrans, ivector1, 0.0 );
      proj_ivector12.AddMatVec(1.0, T2, kNoTrans, ivector1, 0.0 );
      proj_ivector21.AddMatVec(1.0, T1, kNoTrans, ivector2, 0.0 );
      proj_ivector22.AddMatVec(1.0, T2, kNoTrans, ivector2, 0.0 );


      BaseFloat dot_prod1 = VecVec(proj_ivector11, proj_ivector21)/
        ( proj_ivector11.Norm(2.0) * proj_ivector21.Norm(2.0) );


      BaseFloat dot_prod2 = VecVec(proj_ivector12, proj_ivector22)/
         ( proj_ivector12.Norm(2.0) * proj_ivector22.Norm(2.0) );


      // The following will crash if the dimensions differ, but
      // they would likely also differ for all the ivectors so it's probably
      // best to just crash.
      // BaseFloat dot_prod = VecVec(ivector1, ivector2);
      BaseFloat dot_prod = 0.5*(dot_prod1 + dot_prod2) * lda_dim; //length norm factor
      // BaseFloat dot_prod = dot_prod1  * lda_dim; //length norm factor


      sum += dot_prod;
      sumsq += dot_prod * dot_prod;
      num_done++;
      ko.Stream() << key1 << ' ' << key2 << ' ' << dot_prod << std::endl;
    }

    if (num_done != 0) {
      BaseFloat mean = sum / num_done, scatter = sumsq / num_done,
          variance = scatter - mean * mean, stddev = sqrt(variance);
      KALDI_LOG << "Mean dot-product was " << mean << ", standard deviation was "
                << stddev;
    }
    KALDI_LOG << "Processed " << num_done << " trials " << num_err
              << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
