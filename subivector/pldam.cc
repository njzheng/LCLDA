// ivector/plda.cc

// Copyright 2013     Daniel Povey
//           2015     David Snyder

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

#include <vector>
#include "ivector/plda.h"
#include "subivector/pldam.h"


namespace kaldi {


template<class Real>
/// This function computes a projection matrix that when applied makes the
/// covariance unit (i.e. all 1).
static void ComputeNormalizingTransform(const SpMatrix<Real> &covar,
                                        MatrixBase<Real> *proj) {
  int32 dim = covar.NumRows();
  TpMatrix<Real> C(dim);  // Cholesky of covar, covar = C C^T
  C.Cholesky(covar);
  C.Invert();  // The matrix that makes covar unit is C^{-1}, because
               // C^{-1} covar C^{-T} = C^{-1} C C^T C^{-T} = I.
  proj->CopyFromTp(C, kNoTrans);  // set "proj" to C^{-1}.
}


 
Pldam::Pldam(const Matrix<BaseFloat> &plda_mat){
  int32 dim = plda_mat.NumCols();

  KALDI_ASSERT(dim == plda_mat.NumRows()-2);
  mean_.Resize(dim);
  mean_.CopyFromVec(plda_mat.Row(0)); // the first row
  psi_.Resize(dim);
  psi_.CopyFromVec(plda_mat.Row(1)); // the second row

  const SubMatrix<BaseFloat> transform_sub(plda_mat, 2, dim, 0, dim);
  KALDI_ASSERT(transform_sub.NumRows()==dim && transform_sub.NumCols()==dim );


  transform_.Resize(dim, dim);
  transform_.CopyFromMat(transform_sub, kNoTrans);


  // Vector<double> mean_;  // mean of samples in original space.
  // Matrix<double> transform_; // of dimension Dim() by Dim();
  //                           // this transform makes within-class covar unit
  //                           // and diagonalizes the between-class covar.
  // Vector<double> psi_; // of dimension Dim().  The between-class
  //                     // (diagonal) covariance elements, in decreasing order.

  // Vector<double> offset_;  // derived variable: -1.0 * transform_ * mean_

   // ExpectToken(is, binary, "<Plda>");
   // mean_.Read(is, binary);

   // transform_.Read(is, binary);
   // psi_.Read(is, binary);
   // ExpectToken(is, binary, "</Plda>");
   ComputeDerivedVars();
}

Pldam::Pldam(const Matrix<double> &plda_mat){
  int32 dim = plda_mat.NumCols();

  KALDI_ASSERT(dim == plda_mat.NumRows()-2);
  mean_.Resize(dim);
  mean_.CopyFromVec(plda_mat.Row(0)); // the first row
  psi_.Resize(dim);
  psi_.CopyFromVec(plda_mat.Row(1)); // the second row

  const SubMatrix<double> transform_sub(plda_mat, 2, dim, 0, dim);
  KALDI_ASSERT(transform_sub.NumRows()==dim && transform_sub.NumCols()==dim );

  transform_.Resize(dim, dim);
  transform_.CopyFromMat(transform_sub, kNoTrans);

  ComputeDerivedVars();
}


void Pldam::WriteToMatrix(Matrix<BaseFloat> &plda_mat) const{
  int32 dim = plda_mat.NumCols();

  KALDI_ASSERT(dim == this->Dim());
  KALDI_ASSERT(plda_mat.NumRows() == dim + 2);
  Vector<BaseFloat> meanf(mean_);
  Vector<BaseFloat> psif(psi_);
  
  plda_mat.CopyRowFromVec(meanf, 0);
  plda_mat.CopyRowFromVec(psif, 1);

  SubMatrix<BaseFloat> transform_sub(plda_mat, 2, dim, 0, dim); 
  transform_sub.CopyFromMat(transform_, kNoTrans);

}

void Pldam::WriteToMatrix(Matrix<double> &plda_mat) const{
  int32 dim = plda_mat.NumCols(); 

  KALDI_ASSERT(dim == this->Dim());
  KALDI_ASSERT(plda_mat.NumRows() == dim + 2);

  plda_mat.CopyRowFromVec(mean_, 0);
  plda_mat.CopyRowFromVec(psi_, 1);

  SubMatrix<double> transform_sub(plda_mat, 2, dim, 0, dim); 
  transform_sub.CopyFromMat(transform_, kNoTrans);

}

/**
   This comment explains the thinking behind the function LogLikelihoodRatio.
   The reference is "Probabilistic Linear Discriminant Analysis" by
   Sergey Ioffe, ECCV 2006.

   I'm looking at the un-numbered equation between eqs. (4) and (5),
   that says
     P(u^p | u^g_{1...n}) =  N (u^p | \frac{n \Psi}{n \Psi + I} \bar{u}^g, I + \frac{\Psi}{n\Psi + I})

   Here, the superscript ^p refers to the "probe" example (e.g. the example
   to be classified), and u^g_1 is the first "gallery" example, i.e. the first
   training example of that class.  \psi is the between-class covariance
   matrix, assumed to be diagonalized, and I can be interpreted as the within-class
   covariance matrix which we have made unit.

   We want the likelihood ratio P(u^p | u^g_{1..n}) / P(u^p), where the
   numerator is the probability of u^p given that it's in that class, and the
   denominator is the probability of u^p with no class assumption at all
   (e.g. in its own class).

   The expression above even works for n = 0 (e.g. the denominator of the likelihood
   ratio), where it gives us
     P(u^p) = N(u^p | 0, I + \Psi)
   i.e. it's distributed with zero mean and covarance (within + between).
   The likelihood ratio we want is:
      N(u^p | \frac{n \Psi}{n \Psi + I} \bar{u}^g, I + \frac{\Psi}{n \Psi + I}) /
      N(u^p | 0, I + \Psi)
   where \bar{u}^g is the mean of the "gallery examples"; and we can expand the
   log likelihood ratio as
     - 0.5 [ (u^p - m) (I + \Psi/(n \Psi + I))^{-1} (u^p - m)  +  logdet(I + \Psi/(n \Psi + I)) ]
     + 0.5 [u^p (I + \Psi) u^p  +  logdet(I + \Psi) ]
   where m = (n \Psi)/(n \Psi + I) \bar{u}^g.

 */

double Pldam::GetNormalizationFactor(
    const VectorBase<double> &transformed_ivector,
    int32 num_examples) const {
  KALDI_ASSERT(num_examples > 0);
  // Work out the normalization factor.  The covariance for an average over
  // "num_examples" training iVectors equals \Psi + I/num_examples.
  Vector<double> transformed_ivector_sq(transformed_ivector);
  transformed_ivector_sq.ApplyPow(2.0);
  // inv_covar will equal 1.0 / (\Psi + I/num_examples).
  Vector<double> inv_covar(psi_);
  inv_covar.Add(1.0 / num_examples);
  inv_covar.InvertElements();
  // "transformed_ivector" should have covariance (\Psi + I/num_examples), i.e.
  // within-class/num_examples plus between-class covariance.  So
  // transformed_ivector_sq . (I/num_examples + \Psi)^{-1} should be equal to
  //  the dimension.
  // make var(u^bar) =  \Psi + I/num_examples  = var(v) + I/num_examples
  double dot_prod = VecVec(inv_covar, transformed_ivector_sq);
  return sqrt(Dim() / dot_prod);
}




} // namespace kaldi
