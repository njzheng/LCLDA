// ivectorbin/ivector-lclda-pldaset-scoring.cc

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
#include "subivector/pldam.h"



namespace kaldi {

BaseFloat FindSpk(
  const Matrix<BaseFloat> & spk_mean,
  const Vector<BaseFloat> &ivector,
  int32 &spk){

  Vector<BaseFloat> cosine_dis(spk_mean.NumRows());

  // KALDI_LOG<< "size of spk_mean is " <<spk_mean.NumRows()<<" "<<spk_mean.NumCols();
  // KALDI_LOG<< "size of ivector is " <<ivector.Dim();

  cosine_dis.AddMatVec(1.0, spk_mean, kNoTrans, ivector, 0.0);
  cosine_dis(spk_mean.NumRows()-1) = -10.0; //since the last is tot_spk_mean, avoid to select the last one

  int32 indx = -1;
  BaseFloat cos_dis =  cosine_dis.Max(&indx);
  cos_dis /= spk_mean.Row(indx).Norm(2.0) * ivector.Norm(2.0);
  spk = indx;

  return cos_dis;
}

void length_normalization(
  Vector<BaseFloat> &ivector){

  bool normalize = true;
  bool scaleup = true;

  BaseFloat norm = ivector.Norm(2.0);
  BaseFloat ratio = norm / sqrt(ivector.Dim()); // how much larger it is
                                               // than it would be, in                                             // expectation, if normally
  if (!scaleup) ratio = norm;

  if (ratio == 0.0) {
   KALDI_WARN << "Zero iVector";
  } else {
   if (normalize) ivector.Scale(1.0 / ratio);
  }

}

void transform_vec(
  const Matrix<BaseFloat> &transform,
  const Vector<BaseFloat> &vec,
  Vector<BaseFloat> &vec_out){


  int32 transform_rows = transform.NumRows(),
       transform_cols = transform.NumCols(),
       vec_dim = vec.Dim();
   
   // Vector<BaseFloat> vec_out(transform_rows);

   if (transform_cols == vec_dim) {
     vec_out.AddMatVec(1.0, transform, kNoTrans, vec, 0.0);
   } else {
     if (transform_cols != vec_dim + 1) {
       KALDI_ERR << "Dimension mismatch: input vector has dimension "
                 << vec.Dim() << " and transform has " << transform_cols
                 << " columns.";
     }
     vec_out.CopyColFromMat(transform, vec_dim);
     vec_out.AddMatVec(1.0, transform.Range(0, transform.NumRows(),
                                            0, vec_dim), kNoTrans, vec, 1.0);
   }
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
        "Usage: ivector-lclda-plda-scoring <transform> <plda> <train-ivector-rspecifier> <test-ivector-rspecifier>\n"
        " <trials-rxfilename> <scores-wxfilename>\n"
        "\n"
        "e.g.: ivector-lclda-plda-scoring --num-utts=ark:exp/train/num_utts.ark transform.mat plda "
        "ark:exp/train/spk_ivectors.ark ark:exp/test/ivectors.ark trials scores\n"
        "See also: ivector-compute-dot-products, ivector-compute-plda\n";

    ParseOptions po(usage);

    std::string num_utts_rspecifier, transform_rxfilename, spk_mean_rspecifier;

    PldaConfig plda_config;
    plda_config.Register(&po);
    po.Register("num-utts", &num_utts_rspecifier, "Table to read the number of "
                "utterances per speaker, e.g. ark:num_utts.ark\n");
    po.Register("transform", &transform_rxfilename, "Table to read the transform mat\n");
    po.Register("spk-mean", &spk_mean_rspecifier, "ark to read spk mean\n");
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string pldaset_rxfilename = po.GetArg(1),
        train_ivector_rspecifier = po.GetArg(2),
        test_ivector_rspecifier = po.GetArg(3),
        trials_rxfilename = po.GetArg(4),
        scores_wxfilename = po.GetArg(5);


    // read the spk mean information
    SequentialBaseFloatVectorReader spk_mean_reader(spk_mean_rspecifier);
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

    // transform matrix
    // Matrix<BaseFloat> transform;
    // ReadKaldiObject(transform_rxfilename, &transform);
    // int32 transform_rows = transform.NumRows();
    RandomAccessBaseFloatMatrixReader transform_reader(transform_rxfilename);
    int32 transform_rows = transform_reader.Value(spk_list[0]).NumRows();

    //  diagnostics:
    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    int64 num_train_ivectors = 0, num_train_errs = 0, num_test_ivectors = 0;

    int64 num_trials_done = 0, num_trials_err = 0;


    RandomAccessBaseFloatMatrixReader pldaset_reader(pldaset_rxfilename);



    SequentialBaseFloatVectorReader train_ivector_reader(train_ivector_rspecifier);
    SequentialBaseFloatVectorReader test_ivector_reader(test_ivector_rspecifier);
    RandomAccessInt32Reader num_utts_reader(num_utts_rspecifier);

    typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;

    // These hashes will contain the iVectors in the PLDA subspace
    // (that makes the within-class variance unit and diagonalizes the
    // between-class covariance).  They will also possibly be length-normalized,
    // depending on the config.
    HashType train_ivectors1, test_ivectors1;
    HashType train_ivectors_org, test_ivectors_org; // before transform
    HashType train_ivectors2, test_ivectors2; // save the after

    KALDI_LOG << "Reading train iVectors";
    for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) {
      std::string spk = train_ivector_reader.Key();
      if (train_ivectors_org.count(spk) != 0) {
        KALDI_ERR << "Duplicate training iVector found for speaker " << spk;
      }
      Vector<BaseFloat> ivector1 = train_ivector_reader.Value();
      // store the original ivector
      Vector<BaseFloat> *ivector_org1 = new Vector<BaseFloat>(ivector1.Dim());
      ivector_org1->CopyFromVec(ivector1);
      train_ivectors_org[spk] = ivector_org1;

      num_train_ivectors++;
    }
    KALDI_LOG << "Read " << num_train_ivectors << " training iVectors, "
              << "errors on " << num_train_errs;
    if (num_train_ivectors == 0)
      KALDI_ERR << "No training iVectors present.";
    KALDI_LOG << "Average renormalization scale on training iVectors was "
              << (tot_train_renorm_scale / num_train_ivectors);

    KALDI_LOG << "Reading test iVectors";
    for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {
      std::string utt = test_ivector_reader.Key();
      if (test_ivectors_org.count(utt) != 0) {
        KALDI_ERR << "Duplicate test iVector found for utterance " << utt;
      }
      
      Vector<BaseFloat> ivector2 = test_ivector_reader.Value();

      // store the original test ivector
      Vector<BaseFloat> *ivector_org2 = new Vector<BaseFloat>(ivector2.Dim());
      ivector_org2->CopyFromVec(ivector2);
      test_ivectors_org[utt] = ivector_org2;

      num_test_ivectors++;
    }
    KALDI_LOG << "Read " << num_test_ivectors << " test iVectors.";
    if (num_test_ivectors == 0)
      KALDI_ERR << "No test iVectors present.";
    KALDI_LOG << "Average renormalization scale on test iVectors was "
              << (tot_test_renorm_scale / num_test_ivectors);


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
      if (train_ivectors_org.count(key1) == 0) {
        KALDI_WARN << "Key " << key1 << " not present in training iVectors.";
        num_trials_err++;
        continue;
      }
      if (test_ivectors_org.count(key2) == 0) {
        KALDI_WARN << "Key " << key2 << " not present in test iVectors.";
        num_trials_err++;
        continue;
      }

      // find the nearest spker
      Vector<BaseFloat> ivector1 = *(train_ivectors_org[key1]);
      Vector<BaseFloat> ivector2 = *(test_ivectors_org[key2]);
      // Vector<BaseFloat> ivector_mid(ivector1);
      // ivector_mid.AddVec(1.0, ivector2);
      // ivector_mid.Scale(0.5);
      KALDI_LOG<<key1<<" ; "<<key2<<": cos_dis: "
          << VecVec(ivector1, ivector2)/(ivector1.Norm(2.0)*ivector2.Norm(2.0));

      int32 spk_indx1;
      int32 spk_indx2;
      BaseFloat dis1 = FindSpk(spk_mean, ivector1, spk_indx1);
      KALDI_ASSERT(dis1>=0);
      BaseFloat dis2 = FindSpk(spk_mean, ivector2, spk_indx2);
      KALDI_ASSERT(dis2>=0);
      // replace T1
      // BaseFloat dis1 = FindSpk(spk_mean, ivector_mid, spk_indx1);


      //load LDA
      Matrix<BaseFloat> T1(transform_reader.Value(spk_list[spk_indx1]));
      Matrix<BaseFloat> T2(transform_reader.Value(spk_list[spk_indx2])); 

      Vector<BaseFloat> ivector11t(transform_rows),
                        ivector12t(transform_rows);
      transform_vec(T1, ivector1, ivector11t);
      transform_vec(T2, ivector1, ivector12t);

      Vector<BaseFloat> ivector21t(transform_rows),
                        ivector22t(transform_rows);      
      transform_vec(T1, ivector2, ivector21t);
      transform_vec(T2, ivector2, ivector22t);

      length_normalization(ivector11t);
      length_normalization(ivector12t);
      const Vector<BaseFloat> &ivector_ln11 = ivector11t;
      const Vector<BaseFloat> &ivector_ln12 = ivector12t;

      length_normalization(ivector21t);
      length_normalization(ivector22t);
      const Vector<BaseFloat> &ivector_ln21 = ivector21t;
      const Vector<BaseFloat> &ivector_ln22 = ivector22t;


      //load PLDA
      Matrix<BaseFloat> Plda_mat1(pldaset_reader.Value(spk_list[spk_indx1]));
      Matrix<BaseFloat> Plda_mat2(pldaset_reader.Value(spk_list[spk_indx2])); 
      
      Pldam pldam1(Plda_mat1);
      Pldam pldam2(Plda_mat2);
      
      // ReadKaldiObject(plda_rxfilename, &plda);
      int32 dim = pldam1.Dim();
      KALDI_ASSERT(dim==pldam2.Dim());      

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

      Vector<BaseFloat> *train_ivector1 = new Vector<BaseFloat>(dim);
      Vector<BaseFloat> *train_ivector2 = new Vector<BaseFloat>(dim);
      Vector<BaseFloat> *test_ivector1 = new Vector<BaseFloat>(dim);
      Vector<BaseFloat> *test_ivector2 = new Vector<BaseFloat>(dim);

      pldam1.TransformIvector(plda_config,  ivector_ln11,  num_examples,
                             train_ivector1);
      pldam2.TransformIvector(plda_config,  ivector_ln12,  num_examples,
                             train_ivector2);
      num_examples = 1;
      pldam1.TransformIvector(plda_config, ivector_ln21, num_examples,
                                                     test_ivector1);  
      pldam2.TransformIvector(plda_config, ivector_ln22, num_examples,
                                                     test_ivector2);
    

      train_ivectors1[key1] = train_ivector1;
      Vector<double> train_ivector_dbl1(*train_ivector1);
      train_ivectors2[key1] = train_ivector2;
      Vector<double> train_ivector_dbl2(*train_ivector2);

      test_ivectors1[key2] = test_ivector1;
      Vector<double> test_ivector_dbl1(*test_ivector1);
      test_ivectors2[key2] = test_ivector2;
      Vector<double> test_ivector_dbl2(*test_ivector2);

      int32 num_train_examples;
      if (!num_utts_rspecifier.empty()) {
        // we already checked that it has this key.
        num_train_examples = num_utts_reader.Value(key1);
      } else {
        num_train_examples = 1;
      }
      BaseFloat score = pldam1.LogLikelihoodRatio(train_ivector_dbl1,
                                                num_train_examples,
                                                test_ivector_dbl1) ;
      score += pldam2.LogLikelihoodRatio(train_ivector_dbl2,
                                                num_train_examples,
                                                test_ivector_dbl2) ;
      score = score/2.0;
      sum += score;
      sumsq += score * score;
      num_trials_done++;
      ko.Stream() << key1 << ' ' << key2 << ' ' << score << std::endl;
    }

    for (HashType::iterator iter = train_ivectors1.begin();
         iter != train_ivectors1.end(); ++iter)
      delete iter->second;
    for (HashType::iterator iter = test_ivectors1.begin();
         iter != test_ivectors1.end(); ++iter)
      delete iter->second;


    // delete the new function
    for (HashType::iterator iter = train_ivectors_org.begin();
         iter != train_ivectors_org.end(); ++iter)
      delete iter->second;
    for (HashType::iterator iter = test_ivectors_org.begin();
         iter != test_ivectors_org.end(); ++iter)
      delete iter->second;
    for (HashType::iterator iter = train_ivectors2.begin();
         iter != train_ivectors2.end(); ++iter)
      delete iter->second;
    for (HashType::iterator iter = test_ivectors2.begin();
         iter != test_ivectors2.end(); ++iter)
      delete iter->second;


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
