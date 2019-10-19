// ivector/plda.h

// Copyright 2013    Daniel Povey
//           2015    David Snyder


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

#ifndef KALDI_IVECTOR_PLDAM_H_
#define KALDI_IVECTOR_PLDAM_H_

#include <vector>
#include <algorithm>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"
#include "ivector/plda.h"

namespace kaldi {

/* This code implements Probabilistic Linear Discriminant Analysis: see
   "Probabilistic Linear Discriminant Analysis" by Sergey Ioffe, ECCV 2006.
   At least, that was the inspiration.  The E-M is an efficient method
   that I derived myself (note: it could be made even more efficient but
   it doesn't seem to be necessary as it's already very fast).

   This implementation of PLDA only supports estimating with a between-class
   dimension equal to the feature dimension.  If you want a between-class
   covariance that has a lower dimension, you can just remove the smallest
   elements of the diagonalized between-class covariance matrix.  This is not
   100% exact (wouldn't give you as good likelihood as E-M estimation with that
   dimension) but it's close enough.  */


class Pldam : public Plda
{
 public:
  Pldam() { }
  Pldam( const Matrix<BaseFloat> &plda_mat);
  Pldam( const Matrix<double> &plda_mat);

  void WriteToMatrix(Matrix<double> &plda_mat) const;
  void WriteToMatrix(Matrix<BaseFloat> &plda_mat) const;


 protected:


 private:
  Plda &operator = (const Plda &other);  // disallow assignment

  /// This returns a normalization factor, which is a quantity we
  /// must multiply "transformed_ivector" by so that it has the length
  /// that it "should" have.  We assume "transformed_ivector" is an
  /// iVector in the transformed space (i.e., mean-subtracted, and
  /// multiplied by transform_).  The covariance it "should" have
  /// in this space is \Psi + I/num_examples.
  double GetNormalizationFactor(const VectorBase<double> &transformed_ivector,
                                int32 num_examples) const;

};


}  // namespace kaldi

#endif
