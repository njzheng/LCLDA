// ivectorbin/ivector-compute-lda.cc parallel with multi-thread

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
#include <math.h>       /* sqrt */
#include <algorithm>     /* sort */
#include "omp.h"

namespace kaldi {


class CovarianceStats {
 public:
  CovarianceStats(int32 dim): tot_covar_(dim),
                              between_covar_(dim),
                              num_spk_(0),
                              num_utt_(0) { }

  /// get total covariance, normalized per number of frames.
  void GetTotalCovar(SpMatrix<double> *tot_covar) const {
    KALDI_ASSERT(num_utt_ > 0);
    *tot_covar = tot_covar_;
    tot_covar->Scale(1.0 / num_utt_);
  }
  void GetWithinCovar(SpMatrix<double> *within_covar) {
    KALDI_ASSERT(num_utt_ - num_spk_ > 0);
    *within_covar = tot_covar_;
    within_covar->AddSp(-1.0, between_covar_);
    within_covar->Scale(1.0 / num_utt_);
  }
  void AccStats(const Matrix<double> &utts_of_this_spk) {
    int32 num_utts = utts_of_this_spk.NumRows();
    tot_covar_.AddMat2(1.0, utts_of_this_spk, kTrans, 1.0);
    Vector<double> spk_average(Dim());
    spk_average.AddRowSumMat(1.0 / num_utts, utts_of_this_spk);
    between_covar_.AddVec2(num_utts, spk_average); //N_c*(\mu*\mu^T)
    num_utt_ += num_utts;
    num_spk_ += 1;
  }
  /// Will return Empty() if the within-class covariance matrix would be zero.
  bool SingularTotCovar() { return (num_utt_ < Dim()); }
  bool Empty() { return (num_utt_ - num_spk_ == 0); }
  std::string Info() {
    std::ostringstream ostr;
    ostr << num_spk_ << " speakers, " << num_utt_ << " utterances. ";
    return ostr.str();
  }
  int32 Dim() { return tot_covar_.NumRows(); }
  // Use default constructor and assignment operator.
  void AddStats(const CovarianceStats &other) {
    tot_covar_.AddSp(1.0, other.tot_covar_);
    between_covar_.AddSp(1.0, other.between_covar_);
    num_spk_ += other.num_spk_;
    num_utt_ += other.num_utt_;
  }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(CovarianceStats);
  SpMatrix<double> tot_covar_;
  SpMatrix<double> between_covar_;
  int32 num_spk_;
  int32 num_utt_;
};


template<class Real>
void ComputeNormalizingTransform(const SpMatrix<Real> &covar,
                                 Real floor,
                                 MatrixBase<Real> *proj) {
  int32 dim = covar.NumRows();
  Matrix<Real> U(dim, dim);
  Vector<Real> s(dim);
  covar.Eig(&s, &U);
  // Sort eigvenvalues from largest to smallest.
  SortSvd(&s, &U);
  // Floor eigenvalues to a small positive value.
  int32 num_floored;
  floor *= s(0); // Floor relative to the largest eigenvalue
  s.ApplyFloor(floor, &num_floored);
  if (num_floored > 0) {
    KALDI_WARN << "Floored " << num_floored << " eigenvalues of covariance "
               << "to " << floor;
  }
  // Next two lines computes projection proj, such that
  // proj * covar * proj^T = I.
  s.ApplyPow(-0.5);
  proj->AddDiagVecMat(1.0, s, U, kTrans, 0.0);
}

void ComputeDistanceMat(
  const Matrix<double> &mean_spk,
  SpMatrix<double> &dist_mat){

  int32 num_spk_ = mean_spk.NumRows();
  // int32 dim = mean_spk.NumCols();
  KALDI_LOG << "num_spk is "<< num_spk_;

  SpMatrix<double> mean2(num_spk_);
  mean2.AddMat2(1.0, mean_spk, kNoTrans, 0.0); //dist = mean*mean^T

  for(int32 i=0; i< num_spk_; i++){
    for(int32 j=0; j<num_spk_; j++){
      dist_mat(i,j) = mean2(i,i) + mean2(j,j) - 2*mean2(i,j);
    }
  }

  KALDI_LOG << "min of dist is " << dist_mat.Min()
            << ", max of dist is " << dist_mat.Max();

}

void ComputeCosineDistanceMat(
  const Matrix<double> &mean_spk,
  SpMatrix<double> &dist_mat){

  int32 num_spk_ = mean_spk.NumRows();
  // int32 dim = mean_spk.NumCols();
  KALDI_LOG << "num_spk is "<< num_spk_;

  SpMatrix<double> mean2(num_spk_);
  mean2.AddMat2(1.0, mean_spk, kNoTrans, 0.0); //dist = mean*mean^T

  for(int32 i=0; i< num_spk_; i++){
    for(int32 j=0; j<num_spk_; j++){
      dist_mat(i,j) = mean2(i,j)/std::sqrt(mean2(i,i) * mean2(j,j));
    }
  }

  KALDI_LOG << "min of dist is " << dist_mat.Min()
            << ", max of dist is " << dist_mat.Max();

}


void ComputeWeightMat(
  const SpMatrix<double> &dist_mat,
  Matrix<double> &weight_spk,
  double beta = 0.0){

  int32 num_spk_ = dist_mat.NumRows();
  weight_spk.CopyFromSp(dist_mat);

  weight_spk.ApplyPow(0.5); // get ||x-y||_2

  for(int32 i=0; i< num_spk_; i++){
    double max_weight = 0.0;
    for(int32 j=0; j<num_spk_; j++){
      if(i==j){continue;}
      weight_spk(i,j) = 1.0/weight_spk(i,j);

      // get the largest weight
      if(max_weight < weight_spk(i,j)){
        max_weight = weight_spk(i,j);
      }
    }
    // value the diagonal element
    weight_spk(i,i) = max_weight;
  }


  KALDI_LOG << "beta is " << beta;

  if(beta == 0.0){ 
    // all element to one, all equal, return to LDA
    weight_spk.Set(1.0/num_spk_);
  }
  else
    weight_spk.ApplyPow(beta);


  // normalization the weights
  Vector<double> col_sum(num_spk_);
  col_sum.AddColSumMat(1.0, weight_spk, 0.0);
  col_sum.InvertElements();
  col_sum.Scale(num_spk_);
  weight_spk.MulRowsVec(col_sum);

}

void ComputeCosineWeightMat(
  const SpMatrix<double> &dist_mat,
  Matrix<double> &weight_spk,
  double beta = 0.0){

  int32 num_spk_ = dist_mat.NumRows();
  weight_spk.CopyFromSp(dist_mat);

  // replace the diag with the max 
  for(int32 i=0; i< num_spk_; i++){
    double max_weight = 0.0;
    for(int32 j=0; j<num_spk_; j++){
      if(i==j){continue;}
      // get the largest weight
      if(max_weight < weight_spk(i,j)){
        max_weight = weight_spk(i,j);
      }
    }
    // value the diagonal element
    weight_spk(i,i) = max_weight;
  }

  weight_spk.Scale(4.0);
  weight_spk.Sigmoid(weight_spk);

  KALDI_LOG << "beta is " << beta;

  if(beta == 0.0)
    weight_spk.Set(1.0/num_spk_);
  else
    weight_spk.ApplyPow(beta);


  // normalization the weights
  Vector<double> col_sum(num_spk_);
  col_sum.AddColSumMat(1.0, weight_spk, 0.0);
  col_sum.InvertElements();
  col_sum.Scale(num_spk_);
  weight_spk.MulRowsVec(col_sum);

}

// given the k nearest 1 weigth, while 1-beta for others
void ComputekNNWeightMat(
  const SpMatrix<double> &dist_mat,
  Matrix<double> &weight_spk,
  double beta , int32 k){

  int32 num_spk_ = dist_mat.NumRows();
  // Vector<double> mid_value(num_spk_);
  KALDI_ASSERT(k <= num_spk_);

  if(beta>=1.0)
    KALDI_ERR<<"beta is larger than 1!";

  KALDI_LOG<<"Here";
  // replace the diag with the max 
  for(int32 i=0; i< num_spk_; i++){
    // Vector<double> dist_Vrow(dist_mat.NumRows());
    // dist_Vrow.CopyRowFromSp(dist_mat, i);
    std::vector<double> dist_row;

    for(int32 j=0; j< num_spk_; j++)
    {
      dist_row.push_back(dist_mat(i,j)); 
    }
    std::sort(dist_row.begin(), dist_row.end());

    double mid_value = dist_row[num_spk_- k];

    for(int32 j=0; j< num_spk_; j++){
      weight_spk(i,j) = (dist_mat(i,j)<mid_value) ? 1.0- beta : 1.0 ;

    }
  }


}


void ComputeTotMeanSpk(
  const Matrix<double> &mean_spk,
  const Vector<double> &Nc,
  const Matrix<double> &weight_spk,
  Matrix<double> &tot_mean_spk){

  int32 num_spk_ = mean_spk.NumRows();
  int32 dim = mean_spk.NumCols();

  for(int32 spk_indx=0; spk_indx<num_spk_; spk_indx++){
    // Nc*w
    Vector<double> Nc_weight(num_spk_);
    Nc_weight.AddVecVec(1.0, Nc, weight_spk.Row(spk_indx), 0.0);

    //\sum{Nc*w}
    double sum = Nc_weight.Sum();
    
    //\frac{\sum(Nc*w(c)*\mu_c)}{\sum(Nc*w(c))}
    Vector<double> Nc_weight_spk(dim);
    Nc_weight_spk.AddMatVec(1.0, mean_spk, kTrans, Nc_weight, 0.0);
    Nc_weight_spk.Scale(1.0/sum);

    tot_mean_spk.CopyRowFromVec(Nc_weight_spk,spk_indx);
  }

  // original tot mean without weight
  Vector<double> Nc_spk(dim);
  Nc_spk.AddMatVec(1.0, mean_spk, kTrans, Nc, 0.0);
  Nc_spk.Scale(1.0/Nc.Sum());
  tot_mean_spk.CopyRowFromVec(Nc_spk,num_spk_); //the last row

}


// 1-level
void ComputeCLdaTransform(
    const std::map<std::string, Vector<BaseFloat> *> &utt2ivector,
    const std::map<std::string, std::vector<std::string> > &spk2utt,
    BaseFloat total_covariance_factor,
    BaseFloat covariance_floor,
    Matrix<BaseFloat>  &lda_out, //lda_out_vec
    Matrix<BaseFloat>  &spk_mean,
    double beta = 0.0) {
  
  KALDI_ASSERT(!utt2ivector.empty());
  KALDI_ASSERT(beta >=  0.0);

  int32 lda_dim = lda_out.NumRows(), dim = lda_out.NumCols();
  KALDI_ASSERT(dim == utt2ivector.begin()->second->Dim());
  KALDI_ASSERT(lda_dim > 0 && lda_dim <= dim);

  CovarianceStats stats(dim); //no use here

  int32 num_spk_ = spk2utt.size() ; 
  Matrix<double> mean_spk(num_spk_, dim); // tp store the spk_mean
  // Matrix<double> delt(num_spk_, dim); // mean_spk - mean_spk_all

  Vector<double> mean_spk_all(dim); //mean of all ivectors
  Vector<double> Nc(num_spk_); // store the number of utt for each spk

  SpMatrix<double> total_covar(dim); 

  // covariance for each spk
  std::vector<SpMatrix<double> > within_covar_vec(num_spk_); 
  for(int32 i=0; i< num_spk_; i++){
    within_covar_vec[i].Resize(dim);
  }

  std::map<std::string, std::vector<std::string> >::const_iterator iter;
  int32 spk_indx=0;
  for (iter = spk2utt.begin(); iter != spk2utt.end(); ++iter) {
    const std::vector<std::string> &uttlist = iter->second;
    KALDI_ASSERT(!uttlist.empty());

    int32 N = uttlist.size(); // number of utterances.
    Matrix<double> utts_of_this_spk(N, dim);
    for (int32 n = 0; n < N; n++) {
      std::string utt = uttlist[n];
      KALDI_ASSERT(utt2ivector.count(utt) != 0);
      utts_of_this_spk.Row(n).CopyFromVec(
          *(utt2ivector.find(utt)->second));
    }

    // cal spk mean
    int32 num_utts = utts_of_this_spk.NumRows();
    Vector<double> spk_average(dim);
    spk_average.AddRowSumMat(1.0 / num_utts, utts_of_this_spk); 
    mean_spk.Row(spk_indx).CopyFromVec(spk_average);
    Nc(spk_indx) = num_utts + 0.0; //convert to double

    // get the within covariance for each class
    within_covar_vec[spk_indx].AddMat2(1.0, utts_of_this_spk, kTrans, 0.0);
    total_covar.AddSp(1.0, within_covar_vec[spk_indx]); // E(X^2)
    SpMatrix<double> spk_average2(dim);
    spk_average2.AddVec2(num_utts, spk_average); //N_c*(\mu*\mu^T)
    within_covar_vec[spk_indx].AddSp(-1.0, spk_average2);

    // stats.AccStats(utts_of_this_spk);
    spk_indx++;
  }
  spk_mean.CopyFromMat(mean_spk);  //double to float
  mean_spk_all.AddMatVec(1.0/Nc.Sum(), mean_spk, kTrans, Nc, 0.0);  // get the total mean
  KALDI_LOG << "max element of mean_spk_all is " << mean_spk_all.Max(); // it is not zero

  // cal distance matrix
  // store the distance between spk mean
  SpMatrix<double> distance_spk(num_spk_); 
  ComputeDistanceMat(mean_spk, distance_spk);
  // ComputeCosineDistanceMat(mean_spk, distance_spk);


  // store the weight based on distance mat
  Matrix<double> weight_spk(num_spk_,num_spk_); 
  // ComputeWeightMat(distance_spk, weight_spk, beta);
  // ComputeCosineWeightMat(distance_spk, weight_spk, 6.0);
  int32 k = 10 + 1; // used for knn
  ComputekNNWeightMat(distance_spk, weight_spk, beta, k);

  KALDI_LOG<< "weight_spk.Row(0): "<<weight_spk.Row(0);
  // BaseFloatMatrixWriter weight_writer("ark,t:/scratch/njzheng/kaldi/egs/sre10/v1/exp/ivectors_sre_dnn600/weight.ark");
  // Matrix<BaseFloat> weight_spk_s(weight_spk);
  // weight_writer.Write("knn-weight",weight_spk_s);



  // store the total spker mean with weight and Nc
  // the last row is for original tot mean of spk
  Matrix<double> tot_mean_spk(num_spk_ + 1 ,dim); // \mu_c_bar
  ComputeTotMeanSpk(mean_spk,Nc,weight_spk,tot_mean_spk);





  //Sw = \sum{(xx-mm)} within covariance 
  Vector<double> weight_spk_row_sum(num_spk_);
  weight_spk_row_sum.AddRowSumMat(1.0, weight_spk, 0.0);
  SpMatrix<double> within_covar(dim);
  for(int32 i=0; i<num_spk_; i++){
    within_covar.AddSp(weight_spk_row_sum(i), within_covar_vec[i]);
  }

  if(total_covariance_factor>0.0){
    // add some E(X^2)=(Sw+Sb) to Sw to make Sw more robust
    within_covar.Scale(1.0 - total_covariance_factor);
    within_covar.AddSp(total_covariance_factor*weight_spk_row_sum.Sum()/num_spk_, total_covar);
  }
  // within_covar.Scale(1.0/weight_spk_row_sum.Sum());

  KALDI_LOG<< " Sw = sum{(xx-mm)} within covariance  Finished!";

  // Sb = \sum{MDiag(Nc_w)M^T} within covariance 
  SpMatrix<double> between_covar(dim);
  // for(int32 i=0; i< num_spk_; i++)
  // {
  //   Matrix<double> mean_spk_sub(mean_spk); //allocate new memory
  //   mean_spk_sub.AddVecToRows(-1.0, tot_mean_spk.Row(i)); 

  //   Vector<double> Nc_weight(num_spk_);
  //   Nc_weight.AddVecVec(1.0, Nc, weight_spk.Row(i), 0.0);

  //   // Sb=\sum{Nc*wcc*(m-\mu(c))(m-\mu(c))^T}
  //   SpMatrix<double> between_covar_temp(dim);
  //   between_covar_temp.AddMat2Vec(1.0, mean_spk_sub, kTrans, Nc_weight, 0.0);
  //   // between_covar_temp.Scale(1.0/Nc_weight.Sum());
    
  //   between_covar.AddSp(1.0, between_covar_temp);

  //   KALDI_LOG<< "Sb: "<< i;
  // }

  Vector<double> Nc_weight_spk_row_sum(weight_spk_row_sum);
  Nc_weight_spk_row_sum.MulElements(Nc);
  between_covar.AddMat2Vec(1.0, mean_spk, kTrans, Nc_weight_spk_row_sum, 0.0);

  Matrix<double> Nc_weigth_Mu(num_spk_, dim);
  Nc_weigth_Mu.AddDiagVecMat(1.0, Nc, mean_spk, kNoTrans, 0.0); //D_Nc*Mu
  Matrix<double> part2(dim, dim);
  SubMatrix<double> tot_mean_spk_sub(tot_mean_spk, 0, num_spk_ , 0 , dim);
  part2.AddMatMatMat(1.0, tot_mean_spk_sub, kTrans, weight_spk, kNoTrans, Nc_weigth_Mu, kNoTrans, 0.0 );

  SpMatrix<double> part3(dim);
  part3.CopyFromMat(part2, kTakeMean);
  between_covar.AddSp(-2.0, part3);

  Vector<double> w_Nc(num_spk_); 
  w_Nc.AddMatVec(1.0, weight_spk, kNoTrans, Nc, 0.0 );
  between_covar.AddMat2Vec(1.0, tot_mean_spk_sub, kTrans, w_Nc, 1.0);

  KALDI_LOG<< " Sb = sum{MDiag(Nc_w)M^T} within covariance  Finished!";


//   Matrix<double> mean_spk_sub(mean_spk); //allocate new memory
//   mean_spk_sub.AddVecToRows(-1.0, mean_spk_all);


// // Sb(c)=\sum{Nc*w(c)*(m-\mu(c))(m-\mu(c))^T}
//   SpMatrix<double> between_covar(dim);
//   between_covar.AddMat2Vec(1.0, mean_spk_sub, kTrans, Nc, 0.0);
//   between_covar.Scale(1.0/Nc.Sum());

//   //\sum{(xx-mm)}
//   SpMatrix<double> within_covar(dim);
//   for(int32 i=0; i<num_spk_; i++){
//     within_covar.AddSp(1.0, within_covar_vec[i]);
//   }

//   if(total_covariance_factor>0.0){
//     // add some E(X^2)=(Sw+Sb) to Sw to make Sw more robust
//     within_covar.Scale(1.0 - total_covariance_factor);
//     within_covar.AddSp(total_covariance_factor, total_covar);
//   }
//   within_covar.Scale(1.0/Nc.Sum());





  // cal lda_out(lda_dim, dim);
  {
    Matrix<double> T(dim, dim);
    ComputeNormalizingTransform(within_covar,
      static_cast<double>(covariance_floor), &T);   

    SpMatrix<double> between_covar_org_proj(dim);
    between_covar_org_proj.AddMat2Sp(1.0, T, kNoTrans, between_covar, 0.0);

    Matrix<double> U(dim, dim);
    Vector<double> s(dim);
    between_covar_org_proj.Eig(&s, &U);
    bool sort_on_absolute_value = false; 
    SortSvd(&s, &U, static_cast<Matrix<double>*>(NULL),
            sort_on_absolute_value);
    KALDI_LOG << "Singular values of between-class covariance after projecting "
              << "with interpolated [total/within] covariance are: " 
              << s << " For orignal LDA";

    SubMatrix<double> U_part(U, 0, dim, 0, lda_dim);

    // We first transform by T and then by U_part^T.  This means T
    // goes on the right.
    Matrix<double> temp(lda_dim, dim);
    temp.AddMatMat(1.0, U_part, kTrans, T, kNoTrans, 0.0);
    lda_out.CopyFromMat(temp);
  }

}

void ComputeAndSubtractMean(
    std::map<std::string, Vector<BaseFloat> *> utt2ivector,
    Vector<BaseFloat> *mean_out) {
  int32 dim = utt2ivector.begin()->second->Dim();
  size_t num_ivectors = utt2ivector.size();
  Vector<double> mean(dim);
  std::map<std::string, Vector<BaseFloat> *>::iterator iter;
  for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
    mean.AddVec(1.0 / num_ivectors, *(iter->second));
  mean_out->Resize(dim);
  mean_out->CopyFromVec(mean);
  for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
    iter->second->AddVec(-1.0, *mean_out);
}



}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute classed LDA matrices for each class in iVector system.  Reads in iVectors per utterance,\n"
        "and an utt2spk file which it uses to help work out the within-speaker and\n"
        "between-speaker covariance matrices.  Outputs an set of LDA projection  to a\n"
        "specified dimension and mean of spker.  By default it will normalize so that the projected\n"
        "within-class covariance is unit\n"
        "\n"
        "Usage:  ivector-compute-clda [options] <ivector-rspecifier> \n"
        "<utt2spk-rspecifier> <lda-matrix-out> <spk-mean>\n"
        "e.g.: \n"
        " ivector-compute-clda --beta=0.1 ark:ivectors.ark ark:utt2spk lclda.mat ark:spk_mean.ark \n";

    ParseOptions po(usage);

    int32 lda_dim = 200; // Dimension we reduce to
    BaseFloat total_covariance_factor = 0.0,
              covariance_floor = 1.0e-06;
    bool binary = true;
    double beta = 0.0; 

    po.Register("dim", &lda_dim, "Dimension we keep with the LDA transform");
    po.Register("total-covariance-factor", &total_covariance_factor,
                "If this is 0.0 we normalize to make the within-class covariance "
                "unit; if 1.0, the total covariance; if between, we normalize "
                "an interpolated matrix.");
    po.Register("covariance-floor", &covariance_floor, "Floor the eigenvalues "
                "of the interpolated covariance matrix to the product of its "
                "largest eigenvalue and this number.");
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("beta", &beta, "To adjust the weight for neighbor, beta=0, far or near has no difference");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_rspecifier = po.GetArg(1),
        utt2spk_rspecifier = po.GetArg(2),
        lda_wxfilename = po.GetArg(3),
        spk_mean_wxfilename = po.GetArg(4);

    KALDI_ASSERT(covariance_floor >= 0.0);

    int32 num_done = 0, num_err = 0, dim = 0;

    SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);
    // BaseFloatMatrixWriter lclda_writer(lda_wxfilename);
    BaseFloatVectorWriter spk_mean_writer(spk_mean_wxfilename);


    std::map<std::string, Vector<BaseFloat> *> utt2ivector;
    std::map<std::string, std::vector<std::string> > spk2utt;

    for (; !ivector_reader.Done(); ivector_reader.Next()) {
      std::string utt = ivector_reader.Key();
      const Vector<BaseFloat> &ivector = ivector_reader.Value();
      if (utt2ivector.count(utt) != 0) {
        KALDI_WARN << "Duplicate iVector found for utterance " << utt
                   << ", ignoring it.";
        num_err++;
        continue;
      }
      if (!utt2spk_reader.HasKey(utt)) {
        KALDI_WARN << "utt2spk has no entry for utterance " << utt
                   << ", skipping it.";
        num_err++;
        continue;
      }
      std::string spk = utt2spk_reader.Value(utt);
      utt2ivector[utt] = new Vector<BaseFloat>(ivector);
      if (dim == 0) {
        dim = ivector.Dim();
      } else {
        KALDI_ASSERT(dim == ivector.Dim() && "iVector dimension mismatch");
      }
      spk2utt[spk].push_back(utt);
      num_done++;
    }

    KALDI_LOG << "Read " << num_done << " utterances, "
              << num_err << " with errors.";

    if (num_done == 0) {
      KALDI_ERR << "Did not read any utterances.";
    } else {
      KALDI_LOG << "Computing within-class covariance.";
    }

    // Vector<BaseFloat> mean;
    // ComputeAndSubtractMean(utt2ivector, &mean);
    // KALDI_LOG << "2-norm of iVector mean is " << mean.Norm(2.0);


    // spker/class number
    int32 num_spk_ = spk2utt.size();


    Matrix<BaseFloat> lda_mat(lda_dim, dim); // LDA matrix without the offset term.

    Matrix<BaseFloat> spk_mean(num_spk_, dim); //5000*600


    ComputeCLdaTransform(utt2ivector,
                        spk2utt,
                        total_covariance_factor,
                        covariance_floor,
                        lda_mat,
                        spk_mean,
                        beta);

    // Vector<BaseFloat> offset(lda_dim);
    // offset.AddMatVec(-1.0, linear_part, kNoTrans, mean, 0.0);
    // lda_mat.CopyColFromVec(offset, dim); // add mean-offset to transform
    // KALDI_VLOG(2) << "2-norm of transformed iVector mean is "
    //               << offset.Norm(2.0);

    
    std::map<std::string, std::vector<std::string> >::const_iterator siter;
    int32 i=0;
    for (siter = spk2utt.begin(); siter != spk2utt.end(); ++siter) {
      spk_mean_writer.Write(siter->first, Vector<BaseFloat>(spk_mean.Row(i)) ) ;
      i++;
    }
    Vector<BaseFloat> tot_spk_mean(dim);
    tot_spk_mean.AddRowSumMat(1.0/num_spk_,spk_mean,0.0);
    spk_mean_writer.Write("tot_spk",tot_spk_mean); // orignal lda
    
    // WriteKaldiObject(lda_mat, lda_wxfilename, binary);
    WriteKaldiObject(lda_mat, lda_wxfilename, binary);

    KALDI_LOG << "Wrote CLDA transform to "
              << PrintableWxfilename(lda_wxfilename);

    std::map<std::string, Vector<BaseFloat> *>::iterator iter;
    for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
      delete iter->second; //corresponding to new 
    utt2ivector.clear();

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
