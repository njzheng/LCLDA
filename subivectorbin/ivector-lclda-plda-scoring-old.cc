// ivectorbin/ivector-plda-scoring.cc

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
#include "ivector/plda.h"

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
  typedef std::string string;
  try {
    const char *usage =
        "Computes log-likelihood ratios for trials using PLDA model\n"
        "Note: the 'trials-file' has lines of the form\n"
        "<key1> <key2>\n"
        "and the output will have the form\n"
        "<key1> <key2> [<dot-product>]\n"
        "(if either key could not be found, the dot-product field in the output\n"
        "will be absent, and this program will print a warning)\n"
        "For training examples, the input is the iVectors averaged over speakers;\n"
        "a separate archive containing the number of utterances per speaker may be\n"
        "optionally supplied using the --num-utts option; this affects the PLDA\n"
        "scoring (if not supplied, it defaults to 1 per speaker).\n"
        "\n"
        "Usage: ivector-lclda-plda-scoring <plda> <train-ivector-rspecifier> <test-ivector-rspecifier>\n"
        " <trials-rxfilename> \n"
        " <lclda-transform> <spk-mean> <scores-out>\n"
        "\n"
        "e.g.: ivector-lclda-plda-scoring --num-utts=ark:exp/train/num_utts.ark plda "
        "ark:exp/train/spk_ivectors.ark trials ark:exp/test/ivectors.ark ark:transform.ark ark:spk_mean.ark scores\n"
        "See also: ivector-compute-dot-products, ivector-compute-plda\n";

    ParseOptions po(usage);

    std::string num_utts_rspecifier;

    PldaConfig plda_config;
    plda_config.Register(&po);
    po.Register("num-utts", &num_utts_rspecifier, "Table to read the number of "
                "utterances per speaker, e.g. ark:num_utts.ark\n");

    po.Read(argc, argv);

    if (po.NumArgs() != 8) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
        train_ivector_rspecifier = po.GetArg(2),
        test_ivector_rspecifier = po.GetArg(3),
        trials_rxfilename = po.GetArg(4),
        transform_rspecifier = po.GetArg(5),
        spk_mean_rspecifier = po.GetArg(6),
        scores_wxfilename = po.GetArg(7),
        transform_rxfilename = po.GetArg(8);


    Matrix<BaseFloat> transform;
    ReadKaldiObject(transform_rxfilename, &transform);

    int32 transform_rows = transform.NumRows(),
        transform_cols = transform.NumCols();

    //  diagnostics:
    // double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    int64 num_train_ivectors = 0, num_train_errs = 0, num_test_ivectors = 0;

    int64 num_trials_done = 0, num_trials_err = 0;

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    int32 dim = plda.Dim();

    // SequentialBaseFloatVectorReader train_ivector_reader(train_ivector_rspecifier);
    // SequentialBaseFloatVectorReader test_ivector_reader(test_ivector_rspecifier);
    RandomAccessInt32Reader num_utts_reader(num_utts_rspecifier);

    RandomAccessBaseFloatVectorReader ivector1_reader(train_ivector_rspecifier);
    RandomAccessBaseFloatVectorReader ivector2_reader(test_ivector_rspecifier);
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


    // typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;

    // These hashes will contain the iVectors in the PLDA subspace
    // (that makes the within-class variance unit and diagonalizes the
    // between-class covariance).  They will also possibly be length-normalized,
    // depending on the config.
    // HashType train_ivectors, test_ivectors;



    // start here, combine the afored in the while circle
    Input ki(trials_rxfilename);
    bool binary = false;
    Output ko(scores_wxfilename, binary);

    double sum = 0.0, sumsq = 0.0;
    std::string line;

    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 2) {
        KALDI_ERR << "Bad line " << (num_trials_done + num_trials_err)
                  << "in input (expected two fields: key1 key2): " << line;
      }
      std::string key1 = fields[0], key2 = fields[1];
      // if (train_ivectors.count(key1) == 0) {
      //   KALDI_WARN << "Key " << key1 << " not present in training iVectors.";
      //   num_trials_err++;
      //   continue;
      // }
      if (!ivector1_reader.HasKey(key1)) {
        KALDI_WARN << "Key " << key1 << " not present in training iVectors.";
        num_trials_err++;
        continue;
      }
      if (!ivector2_reader.HasKey(key2)) {
        KALDI_WARN << "Key " << key2 << " not present in test iVectors.";
        num_trials_err++;
        continue;
      }

      const Vector<BaseFloat> &ivector1 = ivector1_reader.Value(key1),
          &ivector2 = ivector2_reader.Value(key2);

      // find the nearest spker
      int32 spk_indx1;
      int32 spk_indx2;
      BaseFloat dis1 = FindSpk(spk_mean, ivector1, spk_indx1);
      BaseFloat dis2 = FindSpk(spk_mean, ivector2, spk_indx2);


      // std::string spk_list[spk_indx1]
      Matrix<BaseFloat> T1(transform_reader.Value(spk_list[spk_indx1]));
      Matrix<BaseFloat> T2(transform_reader.Value(spk_list[spk_indx2]));          
      
      // ---------------------------------------------
      int32 vec_dim = ivector1.Dim();
      Vector<BaseFloat> vec_out(transform_rows);
      vec_out.CopyColFromMat(transform, vec_dim);
      // vec_out.AddMatVec(1.0, transform.Range(0, transform.NumRows(),0, vec_dim), kNoTrans, vec, 1.0);

      T1.CopyFromMat(transform.Range(0, transform.NumRows(),0, vec_dim)) ;
      T2.CopyFromMat(transform.Range(0, transform.NumRows(),0, vec_dim)) ;
      // ---------------------------------------------


      int32 lda_dim = T1.NumRows();
      if(lda_dim!=dim)
        KALDI_ERR<<"lda_dim is different with dim "<< lda_dim <<" "<<dim ;

      Vector<BaseFloat> proj_ivector11(lda_dim), proj_ivector12(lda_dim),
                        proj_ivector21(lda_dim), proj_ivector22(lda_dim);
      
      proj_ivector11.AddMatVec(1.0, T1, kNoTrans, ivector1, 0.0 ); //T1
      proj_ivector12.AddMatVec(1.0, T2, kNoTrans, ivector1, 0.0 );
      proj_ivector21.AddMatVec(1.0, T1, kNoTrans, ivector2, 0.0 ); //T1
      proj_ivector22.AddMatVec(1.0, T2, kNoTrans, ivector2, 0.0 );   
         
      proj_ivector11.AddVec(1.0, vec_out); //T1
      proj_ivector12.AddVec(1.0, vec_out );
      proj_ivector21.AddVec(1.0, vec_out); //T1
      proj_ivector22.AddVec(1.0, vec_out);  


      // for training ivector:
      int32 num_examples;
      if (!num_utts_rspecifier.empty()) {
        if (!num_utts_reader.HasKey(key1)) {
          KALDI_WARN << "Number of utterances not given for speaker " << key1;
          num_train_errs++;
          continue;
        }
        num_examples = num_utts_reader.Value(key1);
      } else {
        num_examples = 1;
      }
      Vector<BaseFloat> transformed_ivector11(dim);
      Vector<BaseFloat> transformed_ivector12(dim);
      plda.TransformIvector(plda_config, proj_ivector11, num_examples,
                                                      &transformed_ivector11);
      plda.TransformIvector(plda_config, proj_ivector12, num_examples,
                                                      &transformed_ivector12);
      // for testing ivector:
      num_examples = 1; // this value is always used for test (affects the
                              // length normalization in the TransformIvector
                              // function).
      // Vector<BaseFloat> *transformed_ivector2 = new Vector<BaseFloat>(dim);
      Vector<BaseFloat> transformed_ivector21(dim);
      Vector<BaseFloat> transformed_ivector22(dim);

      plda.TransformIvector(plda_config, proj_ivector21, num_examples,
                                                     &transformed_ivector21);
      plda.TransformIvector(plda_config, proj_ivector22, num_examples,
                                                     &transformed_ivector22);
      // const Vector<BaseFloat> *train_ivector = train_ivectors[key1],
      //     *test_ivector = test_ivectors[key2];

      //float to double and length normalization
      Vector<double> train_ivector_dbl1(transformed_ivector11),
          test_ivector_dbl1(transformed_ivector21);

      train_ivector_dbl1.Scale(sqrt(dim+0.0)/transformed_ivector11.Norm(2.0));
      test_ivector_dbl1.Scale(sqrt(dim+0.0)/transformed_ivector21.Norm(2.0));

      Vector<double> train_ivector_dbl2(transformed_ivector12),
          test_ivector_dbl2(transformed_ivector22);
      train_ivector_dbl2.Scale(sqrt(dim+0.0)/transformed_ivector12.Norm(2.0));
      test_ivector_dbl2.Scale(sqrt(dim+0.0)/transformed_ivector22.Norm(2.0));


      int32 num_train_examples;
      if (!num_utts_rspecifier.empty()) {
        // we already checked that it has this key.
        num_train_examples = num_utts_reader.Value(key1);
      } else {
        num_train_examples = 1;
      }


      BaseFloat score = plda.LogLikelihoodRatio(train_ivector_dbl1,
                                                num_train_examples,
                                                test_ivector_dbl1);

      score += plda.LogLikelihoodRatio(train_ivector_dbl2,
                                                num_train_examples,
                                                test_ivector_dbl2);
      score = score/2.0;

      sum += score;
      sumsq += score * score;
      num_trials_done++;
      ko.Stream() << key1 << ' ' << key2 << ' ' << score << std::endl;
    }

    // for (HashType::iterator iter = train_ivectors.begin();
    //      iter != train_ivectors.end(); ++iter)
    //   delete iter->second;
    // for (HashType::iterator iter = test_ivectors.begin();
    //      iter != test_ivectors.end(); ++iter)
    //   delete iter->second;


    if (num_trials_done != 0) {
      BaseFloat mean = sum / num_trials_done, scatter = sumsq / num_trials_done,
          variance = scatter - mean * mean, stddev = sqrt(variance);
      KALDI_LOG << "Mean score was " << mean << ", standard deviation was "
                << stddev;
    }
    KALDI_LOG << "Processed " << num_trials_done << " trials, " << num_trials_err
              << " had errors.";
    return (num_trials_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
