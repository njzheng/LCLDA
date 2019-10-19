// ivectorbin/ivector-compute-plda.cc

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
#include "util/kaldi-thread.h"

namespace kaldi {

void length_normalization(
  Vector<BaseFloat> &ivector){

  bool normalize = true;
  bool scaleup = true;

  BaseFloat norm = ivector.Norm(2.0);
  BaseFloat ratio = norm / sqrt(ivector.Dim()); // how much larger it is
                                               // than it would be, in                                             
                                               // expectation, if normally
  if (!scaleup) ratio = norm;

  if (ratio == 0.0) {
   KALDI_WARN << "Zero iVector";
  } else {
   if (normalize) ivector.Scale(1.0 / ratio);
  }

}

void length_normalization(
  Matrix<BaseFloat> &ivector){

  bool normalize = true;
  bool scaleup = true;

  int32 num_utt = ivector.NumRows();
  int32 dim = ivector.NumCols();
  Vector<BaseFloat> ratio_vec(num_utt);

  for(int32 i=0; i<num_utt; i++){
    BaseFloat norm = (ivector.Row(i)).Norm(2.0);
    ratio_vec(i) = norm / sqrt(dim); // how much larger it is
                                     // than it would be, in                                             
                                     // expectation, if normally
    if (ratio_vec(i) == 0.0) {
      KALDI_ERR << "Zero iVector";
    } 
  }

  if(normalize) {
    ratio_vec.InvertElements();
    ivector.MulRowsVec(ratio_vec);
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


void transform_mat(
  const Matrix<BaseFloat> &transform,
  const Matrix<BaseFloat> &mat,
  Matrix<BaseFloat> &mat_out){


  int32 transform_rows = transform.NumRows(),
       transform_cols = transform.NumCols(),
       vec_dim = mat.NumCols();
   
   // Vector<BaseFloat> vec_out(transform_rows);

   if (transform_cols == vec_dim) {
     mat_out.AddMatMat(1.0, mat, kNoTrans, transform, kTrans,  0.0);
   } else {
     if (transform_cols != vec_dim + 1) {
       KALDI_ERR << "Dimension mismatch: input vector has dimension "
                 << vec_dim << " and transform has " << transform_cols
                 << " columns.";
     }
     Vector<BaseFloat> vec_temp(transform_rows);
     vec_temp.CopyColFromMat(transform, vec_dim);
     mat_out.CopyRowsFromVec(vec_temp);
     mat_out.AddMatMat(1.0, mat, kNoTrans, transform.Range(0, transform.NumRows(),
                                            0, vec_dim), kTrans,  1.0);
   }
}


class LcpLDAClass {
 public:
  LcpLDAClass(BaseFloat total_covariance_factor,
            int32 i,
            std::vector<Matrix<BaseFloat> > *lda_out_vec,
            Matrix<BaseFloat> *mean_spk,
            Matrix<BaseFloat> *conf_spk_mean_mat,
            Matrix<BaseFloat> *weight_spk,
            Vector<BaseFloat> *Nc,
            Matrix<BaseFloat> *utts_of_all_spk,
            std::vector<SpMatrix<BaseFloat>> *within_covar_vec):
      tcf_(total_covariance_factor),i_(i), lda_out_vec_(lda_out_vec), 
      mean_spk_(mean_spk),
      conf_spk_mean_mat_(conf_spk_mean_mat),
      weight_spk_(weight_spk),Nc_(Nc),
      utts_of_all_spk_(utts_of_all_spk),
      within_covar_vec_(within_covar_vec){ }
  void operator () () {
    ComputeLcpLdaTransform(i_, tcf_, lda_out_vec_,mean_spk_,conf_spk_mean_mat_, weight_spk_,Nc_, utts_of_all_spk_, within_covar_vec_);
  }
  ~LcpLDAClass() { }
 private:
  BaseFloat tcf_;
  int32 i_;
  std::vector<Matrix<BaseFloat> > *lda_out_vec_;
  Matrix<BaseFloat> *mean_spk_;
  Matrix<BaseFloat> *conf_spk_mean_mat_;
  Matrix<BaseFloat> *weight_spk_;
  Vector<BaseFloat> *Nc_;
  Matrix<BaseFloat> *utts_of_all_spk_;
  std::vector<SpMatrix<BaseFloat>> *within_covar_vec_;
};






}


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Computes a set of Plda objects (for Probabilistic Linear Discriminant Analysis)\n"
        "from several sets of iVectors.  Uses speaker information from a spk2utt file\n"
        "to compute within and between class variances.\n"
        "\n"
        "Usage:  ivector-compute-plda [options] <spk2utt-rspecifier> <transform.ark> <ivector-rspecifier> "
        "<plda-out>\n"
        "e.g.: \n"
        " ivector-compute-plda ark:spk2utt ark:transform.ark ark,s,cs:ivectors.ark plda\n";

    ParseOptions po(usage);

    bool binary = true;
    PldaEstimationConfig plda_config;

    plda_config.Register(&po);

    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(1),
        transform_rxfilename = po.GetArg(2),
        ivector_rspecifier = po.GetArg(3),
        plda_wxfilename = po.GetArg(4);

    int64 num_spk_done = 0, num_spk_err = 0,
        num_utt_done = 0, num_utt_err = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatMatrixReader transform_reader(transform_rxfilename);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    BaseFloatMatrixWriter plda_set_writer(plda_wxfilename);



    std::vector<std::string> spk_list;
    std::vector<Matrix<BaseFloat> > ivector_all; // store all ivectors
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      if (uttlist.empty()) {
        KALDI_ERR << "Speaker with no utterances.";
      }

      std::vector<Vector<BaseFloat> > ivectors;
      ivectors.reserve(uttlist.size());

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No iVector present in input for utterance " << utt;
          num_utt_err++;
        } else {
          ivectors.resize(ivectors.size() + 1);
          ivectors.back() = ivector_reader.Value(utt);
          num_utt_done++;
        }
      }

      spk_list.push_back(spk);
      
      if (ivectors.size() == 0) {
        KALDI_WARN << "Not producing output for speaker " << spk
                   << " since no utterances had iVectors";
        num_spk_err++;
      } else {
        Matrix<BaseFloat> ivector_mat(ivectors.size(), ivectors[0].Dim());
        for (size_t i = 0; i < ivectors.size(); i++)
          ivector_mat.Row(i).CopyFromVec(ivectors[i]);
        
        num_spk_done++;
        ivector_all.push_back(ivector_mat);
      }
    }
    spk_list.push_back("LDA"); 
    // if (num_utt_done <= plda_stats.Dim())
    //   KALDI_ERR << "Number of training iVectors is not greater than their "
    //             << "dimension, unable to estimate PLDA.";

    KALDI_LOG << "Accumulated stats from " << num_spk_done << " speakers ("
              << num_spk_err << " with no utterances), consisting of "
              << num_utt_done << " utterances (" << num_utt_err
              << " absent from input).";

    if (num_spk_done == 0)
      KALDI_ERR << "No stats accumulated, unable to estimate PLDA.";
    if (num_spk_done == num_utt_done)
      KALDI_ERR << "No speakers with multiple utterances, "
                << "unable to estimate PLDA.";


    // spker/class number
    int32 num_spk_ = spk_list.size(); // have LDA
    // last one is store the original lda or lplda
    // Vector<BaseFloat> Nc(num_spk_); // store the number of utt for each spk


    Matrix<BaseFloat> T(transform_reader.Value(spk_list[0]));
    int32 plda_dim = T.NumRows();

    // It is easy to combine the parameters of Plda as a matrix: mean, psi, transform
    // and construct an other class to reconstruct Plda from the matrix
    std::vector<Matrix<BaseFloat> > plda_set(num_spk_);


    for(int32 i=0; i<num_spk_; i++)
      plda_set[i].Resize(plda_dim+2, plda_dim); // PLDA matrix without the offset term.

    // parallel compute pldam for each spk
    // for each spk call withinCovar and between Covariance
    // Note, we could have used RunMultiThreaded for this and similar tasks we have here,
    {
      TaskSequencerConfig sequencer_opts;
      sequencer_opts.num_threads = 16; // g_num_threads;
      TaskSequencer<PLDAMClass> sequencer(sequencer_opts);
      for(int32 spk_indx=0; spk_indx< num_spk_; spk_indx++)
        sequencer.Run(new LcpLDAClass(
          total_covariance_factor,
          spk_indx, 
          &lda_out_vec,
          &mean_spk,
          &conf_spk_mean_mat,
          &weight_spk,
          &Nc,
          &utts_of_all_spk,
          &within_covar_vec));
    }




    // orignal pldam
    int32 num_spk_skip=0;
    for (int32 spk_indx=0; spk_indx< num_spk_; spk_indx++) 
    {
      KALDI_LOG << spk_indx <<" : "<<spk_list[spk_indx];
      PldaStats plda_stats;

      if(!transform_reader.HasKey(spk_list[spk_indx])){
        KALDI_WARN << "transform has no entry for spk " << spk_list[spk_indx]
             << ", skipping it.";
        num_spk_skip++;
        continue;
      }
      Matrix<BaseFloat> T(transform_reader.Value(spk_list[spk_indx]));
      KALDI_LOG << T.Sum() ;

      for(int32 j=0; j<ivector_all.size(); j++){
        Matrix<BaseFloat> ivector_t(ivector_all[j].NumRows(), plda_dim);
        transform_mat(T, ivector_all[j], ivector_t);
        length_normalization(ivector_t);
        // KALDI_LOG << ivector_t.Sum() ;
        double weight = 1.0; 
        plda_stats.AddSamples(weight, (Matrix<double>) ivector_t);
      }

      plda_stats.Sort();
      PldaEstimator plda_estimator(plda_stats);
      Pldam pldam; //plda with mat convert
      plda_estimator.Estimate(plda_config, &pldam);

      pldam.WriteToMatrix(plda_set[spk_indx]);

      plda_set_writer.Write(spk_list[spk_indx],plda_set[spk_indx]);
    }

    // WriteKaldiObject(plda, plda_wxfilename, binary);
    KALDI_WARN << "There are "<< num_spk_skip <<" speakers with no entry ";

    return (num_spk_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
