// subivectorbin/subivector-dist.cc

// Copyright 2013-2014  Daniel Povey

// writen by z3c
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
#include <string.h>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Usage: subivector-dist  <subivector-rspecifier> "
        "<subivector-wspecifier> \n"
        "or:subivector-dist scp:$dir/ivector.scp $dir/ivector_dist.ark\n";


    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "If true, write output in binary "
                "(only applicable when writing files, not archives/tables.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    // Compute the dist of the input vectors and write it out.
    std::string subivector_rspecifier = po.GetArg(1),
        dist_wxfilename = po.GetArg(2);
    int32 num_done = 0;

    SequentialBaseFloatMatrixReader subivector_reader(subivector_rspecifier);

    Matrix<BaseFloat> mean_mat;
    Matrix<BaseFloat> norm_mat;
    Vector<BaseFloat> num_phone_prob; // phone_exist/num_utt
    // cal the covariance
    Vector<BaseFloat> norm_vec;
    Vector<BaseFloat> var_vec;
    // store the utt norm respectively
    std::vector<Vector<BaseFloat>> utt_norm_vv;
    for (; !subivector_reader.Done(); subivector_reader.Next()) {
      // if (sum.Dim() == 0) sum.Resize(subivector_reader.Value().Dim());
      if (mean_mat.NumRows() == 0){
        mean_mat.Resize(subivector_reader.Value().NumRows(),subivector_reader.Value().NumCols());
        norm_mat.Resize(subivector_reader.Value().NumRows(),subivector_reader.Value().NumRows());
        num_phone_prob.Resize(subivector_reader.Value().NumRows());
        // cal the covariance
        norm_vec.Resize(subivector_reader.Value().NumRows());
        var_vec.Resize(subivector_reader.Value().NumRows());

      }
      // test exist or not 
      Matrix<BaseFloat> temp_mat(subivector_reader.Value());
      Vector<BaseFloat> utt_norm_vec(temp_mat.NumRows());

      for(int32 i=0; i< temp_mat.NumRows(); i++)
      {
        SubVector<BaseFloat> row_of_m(temp_mat, i);
        float temp_norm = row_of_m.Norm(2.0);
        utt_norm_vec(i) = temp_norm;
        if(temp_norm > 0.01){
          num_phone_prob(i)++;
          norm_vec(i) = norm_vec(i) +  temp_norm * temp_norm;
        }

      }

      // store utt_norm_vec into utt_norm_vv
      utt_norm_vv.push_back(utt_norm_vec);

      mean_mat.AddMat(1.0, temp_mat);
      num_done++;
    }

    if (num_done == 0) {
      KALDI_ERR << "No subiVectors read";
    } else {
      num_phone_prob.Scale(1.0 / num_done);
      Vector<BaseFloat> num_phone_prob_inv(num_phone_prob);
      num_phone_prob_inv.InvertElements();
      mean_mat.MulRowsVec(num_phone_prob_inv);
      mean_mat.Scale(1.0 / num_done);

      norm_vec.MulElements(num_phone_prob_inv);
      norm_vec.Scale(1.0 / num_done);
    }

    for(int32 i=0; i< mean_mat.NumRows(); i++)
    {
      SubVector<BaseFloat> row_of_m2(mean_mat, i);
      float temp_norm2 = row_of_m2.Norm(2.0);
      var_vec(i) = norm_vec(i) - temp_norm2 * temp_norm2;
    }

    // convert utt_norm_vv into mat form
    Matrix<BaseFloat> utt_norm_mat(mean_mat.NumRows(),num_done);
    for(int32 j=0; j< num_done; j++){
      for(int32 i=0; i< mean_mat.NumRows(); i++){
        utt_norm_mat(i,j) = utt_norm_vv[j](i);
      }
    }



    
    if (num_done == 0) {
      KALDI_ERR << "No subiVectors read";
    } else {
      
      // WriteKaldiObject(mean_mat, dist_wxfilename, binary_write);
      Output ko(dist_wxfilename, binary);   
         

      WriteToken(ko.Stream(), binary, "<num_phone_prob>");
      num_phone_prob.Write(ko.Stream(), binary);

      WriteToken(ko.Stream(), binary, "<var_vec>");
      var_vec.Write(ko.Stream(), binary);

      WriteToken(ko.Stream(), binary, "<norm_vec>");
      norm_vec.Write(ko.Stream(), binary);
      
      WriteToken(ko.Stream(), binary, "<utt_num>");
      WriteBasicType(ko.Stream(), binary, num_done);

      WriteToken(ko.Stream(), binary, "<ivector_dim>");
      WriteBasicType(ko.Stream(), binary,mean_mat.NumRows());

      WriteToken(ko.Stream(), binary, "<mean_mat>");
      mean_mat.Write(ko.Stream(), binary);

      WriteToken(ko.Stream(), binary, "<utt_norm_mat>");
      utt_norm_mat.Write(ko.Stream(), binary);


      return 0;
    }



  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
