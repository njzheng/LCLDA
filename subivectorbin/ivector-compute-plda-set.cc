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


void ComputePLDAM(
  int32 spk_indx,
  int32 plda_dim,
  const Matrix<BaseFloat> *weight_spk,
  const std::vector<std::string> *spk_list,
  std::vector<Matrix<BaseFloat> > * plda_set,
  const std::vector<Matrix<BaseFloat> > *lda_vec,
  const std::vector<Matrix<BaseFloat> > *ivector_all);


class PLDAMClass {
 public:
  PLDAMClass(int32 i,
            int32 plda_dim,
            Matrix<BaseFloat> *weight_spk,
            std::vector<std::string> *spk_list,
            std::vector<Matrix<BaseFloat> > *plda_set,
            std::vector<Matrix<BaseFloat> > *lda_vec,
            std::vector<Matrix<BaseFloat> > *ivector_all):
      i_(i), plda_dim_(plda_dim), weight_spk_(weight_spk),
      spk_list_(spk_list), 
      plda_set_(plda_set), 
      lda_vec_(lda_vec),
      ivector_all_(ivector_all){ }
  void operator () () {
    ComputePLDAM(i_, plda_dim_, weight_spk_, spk_list_, plda_set_, lda_vec_, ivector_all_);
  }
  ~PLDAMClass() { }
 private:
  int32 i_;
  int32 plda_dim_;
  Matrix<BaseFloat> *weight_spk_;
  std::vector<std::string> *spk_list_;
  std::vector<Matrix<BaseFloat> > * plda_set_;
  std::vector<Matrix<BaseFloat> > *lda_vec_;
  std::vector<Matrix<BaseFloat> > *ivector_all_;
};

void ComputePLDAM(
  int32 spk_indx,
  int32 plda_dim,
  const Matrix<BaseFloat> *weight_spk,
  const std::vector<std::string> *spk_list,
  std::vector<Matrix<BaseFloat> > * plda_set,
  const std::vector<Matrix<BaseFloat> > *lda_vec,
  const std::vector<Matrix<BaseFloat> > *ivector_all){

  std::string spk_name = *(spk_list->begin() + spk_indx);
  // KALDI_LOG << spk_indx <<" : "<< spk_name;
  PldaStats plda_stats;

  Matrix<BaseFloat> T(*(lda_vec->begin()+spk_indx));
  if(T.NumRows()!=plda_dim){
    // without this spker lda information
    KALDI_LOG << spk_indx <<" : "<< spk_name <<" has no LDA mat, skipping. " ;
    return;
  }

  Vector<BaseFloat> weight_vec(weight_spk->Row(spk_indx));

  for(int32 j=0; j<ivector_all->size(); j++){
    Matrix<BaseFloat> ivector_t((ivector_all->begin()+j)->NumRows(), plda_dim);
    transform_mat(T, *(ivector_all->begin()+j), ivector_t);
    length_normalization(ivector_t);
    // KALDI_LOG << ivector_t.Sum() ;
    // double weight = 1.0; 
    double weight = weight_vec(j) ;
    // if(weight<=0)
    //   KALDI_LOG << spk_indx <<" : "<< spk_name <<" with weight "<< weight ;

    plda_stats.AddSamples(weight, (Matrix<double>) ivector_t);
  }
  KALDI_LOG << spk_indx <<" : "<< spk_name <<" with T.Sum = "<< T.Sum()
            << "with weight_max ~ weight_min: " << weight_vec.Max() <<" ~ "
            << weight_vec.Min();

  plda_stats.Sort();
  PldaEstimator plda_estimator(plda_stats);
  Pldam pldam; //plda with mat convert
  PldaEstimationConfig plda_config; //defalut iter_num=10;
  plda_estimator.Estimate(plda_config, &pldam);

  Matrix<BaseFloat> pldam_mat(plda_dim+2, plda_dim);
  pldam.WriteToMatrix(pldam_mat);
  (plda_set->begin()+spk_indx)->CopyFromMat(pldam_mat);
  // plda_set_writer.Write(spk_list[spk_indx],plda_set[spk_indx]);

}

void ComputeCosineDistanceMat(
  const Matrix<BaseFloat> &mean_spk,
  SpMatrix<BaseFloat> &dist_mat){

  int32 num_spk_ = mean_spk.NumRows();
  // int32 dim = mean_spk.NumCols();
  KALDI_LOG << "num_spk is "<< num_spk_;

  SpMatrix<BaseFloat> mean2(num_spk_);
  mean2.AddMat2(1.0, mean_spk, kNoTrans, 0.0); //dist = mean*mean^T

  for(int32 i=0; i< num_spk_; i++){
    for(int32 j=0; j<num_spk_; j++){
      double tnorm = mean2(i,i) * mean2(j,j);
      //avoid zero-divide
      dist_mat(i,j) = (tnorm== 0 ? mean2(i,j) : mean2(i,j)/std::sqrt(tnorm));
    }
  }

  KALDI_LOG << "min of dist is " << dist_mat.Min()
            << ", max of dist is " << dist_mat.Max();
}

void ComputePdfWeightMat(
  const SpMatrix<BaseFloat> &dist_mat,
  Matrix<BaseFloat> &weight_spk,
  const Vector<BaseFloat> &Nc,
  const Matrix<BaseFloat> &gmm_pdf){

  int32 num_gmm_ = gmm_pdf.NumRows();

  double mix_prob[num_gmm_];
  double mean[num_gmm_];
  double sigma[num_gmm_];
  for(int32 i=0; i<num_gmm_; i++){
    mix_prob[i] = gmm_pdf(i,0);
    mean[i] = gmm_pdf(i,1);
    sigma[i] = gmm_pdf(i,2);
  }
  KALDI_LOG<<"read gmm pdf : mix_prob "<<mix_prob[0] <<", "<< mix_prob[1];
  KALDI_LOG<<"read gmm pdf : mean "<<mean[0] <<", "<< mean[1];
  KALDI_LOG<<"read gmm pdf : sigma "<<sigma[0] <<", "<< sigma[1]; 

  int32 num_spk_ = dist_mat.NumRows();
  // weight_spk.CopysFromSp(dist_mat);
  Matrix<BaseFloat> dist_mat_temp(dist_mat);


  double num_utt_ = Nc.Sum();
  KALDI_LOG<<"num_utt_:"<<num_utt_;

  // replace the diag with the max 
  for(int32 i=0; i< num_spk_; i++){
    double max_weight = 0.0;
    for(int32 j=0; j<num_spk_; j++){
      if(i==j){continue;}
      // get the largest weight
      if(max_weight < dist_mat_temp(i,j)){
        max_weight = dist_mat_temp(i,j);
      }
    }
    // value the diagonal element
    dist_mat_temp(i,i) = max_weight;

    // get the mean and sigma to fit Gaussian pdf
    Vector<BaseFloat> spk_dist(dist_mat_temp.NumCols());
    spk_dist.CopyRowFromMat(dist_mat_temp, i);

    double mu = VecVec(spk_dist,Nc)/num_utt_; // in term of utt 0.002; 
    spk_dist.ApplyPow(2.0);
    double scov = sqrt( VecVec(spk_dist,Nc)/num_utt_ - mu*mu);// 0.0468;

    // cal the weight
    // two gmms
    Vector<BaseFloat> dist_sub0(dist_mat_temp.NumCols()), // spk_dist
                   dist_sub1(dist_mat_temp.NumCols()), // gmm1
                   dist_sub2(dist_mat_temp.NumCols()); // gmm2
    dist_sub0.CopyRowFromMat(dist_mat_temp, i);
    dist_sub1.CopyRowFromMat(dist_mat_temp, i);
    dist_sub2.CopyRowFromMat(dist_mat_temp, i);
    
    dist_sub0.Add(-1.0*mu);
    dist_sub0.Scale(1.0/scov);
    
    dist_sub1.Add(-1.0*mean[0]);
    dist_sub1.Scale(1.0/sigma[0]);
    dist_sub2.Add(-1.0*mean[1]);
    dist_sub2.Scale(1.0/sigma[1]);
    // ((x-mu)/sigma)^2
    dist_sub0.ApplyPow(2.0);
    dist_sub1.ApplyPow(2.0);
    dist_sub2.ApplyPow(2.0);

    dist_sub1.Scale(-0.5);
    dist_sub2.Scale(-0.5);
    dist_sub1.AddVec(0.5, dist_sub0);
    dist_sub2.AddVec(0.5, dist_sub0);
    // exp()
    for(int32 j=0; j< num_spk_; j++){
      double ratio = mix_prob[0]*scov/sigma[0]*Exp(dist_sub1(j)) + 
                        mix_prob[1]*scov/sigma[1]*Exp(dist_sub2(j));

      weight_spk(i,j) = ratio > 10? 10: ratio + 1e-6;
    }

  }

  // normalization the weights
  Vector<BaseFloat> col_sum(num_spk_);
  col_sum.AddColSumMat(1.0, weight_spk, 0.0);
  col_sum.InvertElements();
  col_sum.Scale(num_spk_);
  weight_spk.MulRowsVec(col_sum);
}


// without gmm_mat piror information
template<class Real>
void ComputePdfWeightMat(
  const SpMatrix<Real> &dist_mat,
  Matrix<Real> &weight_spk,
  const Vector<Real> &Nc){


  int32 num_spk_ = dist_mat.NumRows();
  // weight_spk.CopysFromSp(dist_mat);
  Matrix<Real> dist_mat_temp(dist_mat);


  double num_utt_ = Nc.Sum();
  KALDI_LOG<<"num_utt_:"<<num_utt_;

  // replace the diag with the max 
  for(int32 i=0; i< num_spk_; i++){
    double max_weight = 0.0;
    for(int32 j=0; j<num_spk_; j++){
      if(i==j){continue;}
      // get the largest weight
      if(max_weight < dist_mat_temp(i,j)){
        max_weight = dist_mat_temp(i,j);
      }
    }
    // value the diagonal element
    dist_mat_temp(i,i) = max_weight;
  }
  
  // get the mu and cov of all utterances
  double mu_all = VecMatVec(Nc, dist_mat_temp, Nc)/(num_utt_*num_utt_);
  Matrix<Real> dist_mat_temp_sub(dist_mat_temp);
  
  dist_mat_temp_sub.Add(-1.0*mu_all);
  dist_mat_temp_sub.ApplyPow(2.0);
  double scov_all = sqrt(VecMatVec(Nc, dist_mat_temp_sub, Nc)/(num_utt_*num_utt_));

  Matrix<Real> dist_post_indx(dist_mat); //only record positive ones for MulElements()
  dist_post_indx.Set(1.0);
  // replace the diag with the max 
  for(int32 i=0; i< num_spk_; i++){
    for(int32 j=0; j<num_spk_; j++)
      if(dist_mat(i,j) <= mu_all)
        dist_post_indx(i,j) = 0;
  }

  // get the postive mu and cov of all utterances
  dist_mat_temp.MulElements(dist_post_indx); //zero the negative ones
  double pmu_all = VecMatVec(Nc, dist_mat_temp, Nc)/VecMatVec(Nc, dist_post_indx, Nc);

  KALDI_LOG<<"The global mean and std covariance is :( "<< mu_all <<" , "<< scov_all
            <<" ); and the postive mean is : " << pmu_all;

  for(int32 i=0; i< num_spk_; i++){
    // get the mean and sigma to fit Gaussian pdf
    Vector<double> spk_dist(dist_mat_temp.NumCols());
    spk_dist.CopyRowFromMat(dist_mat_temp, i);

    double mu = VecVec(spk_dist,Nc)/num_utt_; // in term of utt 0.002; 
    spk_dist.ApplyPow(2.0);
    double scov = sqrt( VecVec(spk_dist,Nc)/num_utt_ - mu*mu);// 0.0468;

    // double mu = mu_all;
    // double scov = scov_all;


    spk_dist.CopyRowFromMat(dist_mat_temp,i);
    // use positive mean, and the same cov to construct a guassian density
    Vector<double> pNc(Nc.Dim());
    for(int32 j=0; j< num_spk_; j++)
      if(spk_dist(j)>0)
        pNc(j) = Nc(j);

    double pmu = pmu_all;
    double pscov = scov_all;
    //if (pNc.Sum() > 0) // or value 0
     //pmu = VecVec(spk_dist, pNc)/pNc.Sum(); 
 
    // cal the weight

    // exp()
    for(int32 j=0; j< num_spk_; j++){
      double ratio = scov/pscov*(Exp(-(spk_dist(j)-pmu)*(spk_dist(j)-pmu)/(2*pscov*pscov) 
                + (spk_dist(j)-mu)*(spk_dist(j)-mu)/(2*scov*scov) )) ; 

      weight_spk(i,j) = ratio > 4? 4: ratio + 1e-6;
    }
  }

  // normalization the weights
  Vector<Real> col_sum(num_spk_);
  col_sum.AddColSumMat(1.0, weight_spk, 0.0);
  col_sum.InvertElements();
  col_sum.Scale(num_spk_);
  weight_spk.MulRowsVec(col_sum);
}





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
        " ivector-compute-plda-set --nj=16 ark:spk2utt ark:transform.ark ark,s,cs:ivectors.ark plda\n";

    ParseOptions po(usage);
    std::string pdf_rxfilename, spk_mean_rspecifier;
    int32 nj=16; // parallel threads to compute pldam
    bool binary = true;
    PldaEstimationConfig plda_config;

    plda_config.Register(&po);

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("nj", &nj, "Write output in binary mode");
    po.Register("pdf", &pdf_rxfilename, "ark to read gmm parameters matrix\n");
    po.Register("spk-mean", &spk_mean_rspecifier, "ark to read spk mean\n");

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

    if(pdf_rxfilename.size()==0 || spk_mean_rspecifier.size()==0){
      KALDI_ERR<<"There is no pdf file: "<< pdf_rxfilename
                <<"or spk_mean file: " << spk_mean_rspecifier;
    }
    RandomAccessBaseFloatVectorReader spk_mean_reader(spk_mean_rspecifier);

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

    // read pdf and Cal weight
    int32 ivector_dim = ivector_all[0].NumCols();
    Matrix<BaseFloat> gmm_mat;
    ReadKaldiObject(pdf_rxfilename, &gmm_mat);
    Matrix<BaseFloat> spk_mean(num_spk_, ivector_dim);
    Vector<BaseFloat> Nc(num_spk_);

    Matrix<BaseFloat> T(transform_reader.Value(spk_list[0]));
    int32 plda_dim = T.NumRows();
    int32 dim = T.NumCols(); // may be equal to ivector_dim+1

    KALDI_LOG<<"Read LDA and spk_mean";
    // It is easy to combine the parameters of Plda as a matrix: mean, psi, transform
    // and construct an other class to reconstruct Plda from the matrix
    std::vector<Matrix<BaseFloat> > plda_set(num_spk_);
    std::vector<Matrix<BaseFloat> > lda_vec(num_spk_);
    int32 num_spk_skip=0;
    for(int32 i=0; i<num_spk_; i++){
      plda_set[i].Resize(plda_dim+2, plda_dim); // PLDA matrix without the offset term.
      // store the LDA matrices into a vector
      if(!transform_reader.HasKey(spk_list[i])){
        KALDI_WARN << "transform has no entry for spk " << spk_list[i]
             << ", skipping it.";
        num_spk_skip++;
        continue;
      } 
      lda_vec[i].Resize(plda_dim,dim);
      lda_vec[i].CopyFromMat(transform_reader.Value(spk_list[i]));
      Nc(i) = ivector_all[i].NumRows();

      // read spk_mean
      Vector<BaseFloat> temp(ivector_dim);
      if(spk_list[i]!="LDA")
        temp.CopyFromVec(spk_mean_reader.Value(spk_list[i]));
      else
        temp.CopyFromVec(spk_mean_reader.Value("tot_spk"));

      spk_mean.CopyRowFromVec(temp, i);

    }
    KALDI_WARN << "There are "<< num_spk_skip <<" speakers with no entry ";

    // compute weight for each spk
    SpMatrix<BaseFloat> distance_spk(num_spk_); 
    ComputeCosineDistanceMat(spk_mean, distance_spk);
    Matrix<BaseFloat> weight_spk(num_spk_,num_spk_); 
    // ComputePdfWeightMat(distance_spk, weight_spk, Nc, gmm_mat);
    ComputePdfWeightMat(distance_spk, weight_spk, Nc);


    // KALDI_LOG<< "weight_spk.Row(0): "<<weight_spk.Row(0);
    // KALDI_LOG<< "Nc: "<<Nc;
    
    // BaseFloatMatrixWriter distance_writer("ark,t:/scratch/njzheng/kaldi/egs/sre10/v3/exp/distance.ark");
    // Matrix<BaseFloat> distance_spk_s(distance_spk);
    // distance_writer.Write("cos-distnace",distance_spk_s);

    // BaseFloatMatrixWriter weight_writer("ark,t:/scratch/njzheng/kaldi/egs/sre10/v3/exp/weight.ark");
    // Matrix<BaseFloat> weight_spk_s(weight_spk);
    // weight_writer.Write("cos-weight",weight_spk_s);


    // parallel compute pldam for each spk
    // for each spk call withinCovar and between Covariance
    // Note, we could have used RunMultiThreaded for this and similar tasks we have here,
    // the iteration for plda estimation is set as default 10.
    {
      TaskSequencerConfig sequencer_opts;
      sequencer_opts.num_threads = nj; // g_num_threads;
      TaskSequencer<PLDAMClass> sequencer(sequencer_opts);
      for(int32 spk_indx=0; spk_indx< num_spk_; spk_indx++){
        sequencer.Run(
          new PLDAMClass(
            spk_indx,
            plda_dim,            
            &weight_spk,
            &spk_list,
            &plda_set,
            &lda_vec,
            &ivector_all)
          );
      }
    }




    // orignal pldam

    for (int32 spk_indx=0; spk_indx< num_spk_; spk_indx++) 
    {
      // KALDI_LOG << spk_indx <<" : "<<spk_list[spk_indx];
      // PldaStats plda_stats;

      if(!transform_reader.HasKey(spk_list[spk_indx])){
        KALDI_WARN << "transform has no entry for spk " << spk_list[spk_indx]
             << ", skipping it.";
        num_spk_skip++;
        continue;
      }
      // Matrix<BaseFloat> T(transform_reader.Value(spk_list[spk_indx]));
      // KALDI_LOG << T.Sum() ;

      // for(int32 j=0; j<ivector_all.size(); j++){
      //   Matrix<BaseFloat> ivector_t(ivector_all[j].NumRows(), plda_dim);
      //   transform_mat(T, ivector_all[j], ivector_t);
      //   length_normalization(ivector_t);
      //   // KALDI_LOG << ivector_t.Sum() ;
      //   double weight = 1.0; 
      //   plda_stats.AddSamples(weight, (Matrix<double>) ivector_t);
      // }

      // plda_stats.Sort();
      // PldaEstimator plda_estimator(plda_stats);
      // Pldam pldam; //plda with mat convert
      // plda_estimator.Estimate(plda_config, &pldam);

      // pldam.WriteToMatrix(plda_set[spk_indx]);

      plda_set_writer.Write(spk_list[spk_indx],plda_set[spk_indx]);
    }

    // WriteKaldiObject(plda, plda_wxfilename, binary);

    return (num_spk_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
