// bin/transform-vec-lclda.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2014  Johns Hopkins University (author: Daniel Povey)

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
#include "matrix/kaldi-matrix.h"
#include <algorithm>    // std::find

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
  try {
    using namespace kaldi;

    const char *usage =
        "This program applies a linear or affine transform set to individual vectors, e.g.\n"
        "iVectors.  It is transform-feats, except it works on vectors rather than matrices,\n"
        "and expects a table of matrices for transform according to different spk-means\n"
        "\n"
        "Usage: transform-vec-lclda [options] <transform-rxfilename> <spk-mean-ln-rxfilename> <feats-rspecifier> <feats-wspecifier>\n"
        "See also: transform-vec, est-pca\n";

        // transform-vec-lclda  ark:${plda_ivec_dir}/transform.ark ark:${plda_ivec_dir}/spk_mean_ln.ark  ark:- ark:-|
    
    ParseOptions po(usage);
    
    bool development = false;
    po.Register("development", &development, "development data is used to training");


    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string transform_rxfilename = po.GetArg(1);
    std::string spk_mean_rxfilename = po.GetArg(2);    
    std::string vec_rspecifier = po.GetArg(3);
    std::string vec_wspecifier = po.GetArg(4);

    RandomAccessBaseFloatMatrixReader transform_reader(transform_rxfilename);
    // spk mean vectors need to be length normalization for cos distance
    SequentialBaseFloatVectorReader spk_mean_reader(spk_mean_rxfilename);
    SequentialBaseFloatVectorReader vec_reader(vec_rspecifier);
    BaseFloatVectorWriter vec_writer(vec_wspecifier);
  

    // read the spk mean information: key -> spk_list; value-> spk_mean.Row()
    std::vector< std::string > spk_list;
    std::vector< Vector<BaseFloat> > spk_mean_temp;
    for (; !spk_mean_reader.Done(); spk_mean_reader.Next()) {
      spk_list.push_back(spk_mean_reader.Key());
      spk_mean_temp.push_back(spk_mean_reader.Value());
    }
    // since the last row is the total mean
    int32 num_spk_ = spk_mean_temp.size();
    KALDI_LOG << "num_spk_ " << num_spk_ ;

    Matrix<BaseFloat> spk_mean(spk_mean_temp.size(), spk_mean_temp[0].Dim());
    for(int32 i=0; i<spk_mean_temp.size() ; i++){
      spk_mean.CopyRowFromVec(spk_mean_temp[i], i);
    }
    spk_mean_temp.clear();


    // Matrix<BaseFloat> transform;
    // ReadKaldiObject(transform_rxfilename, &transform);

    int32 num_done = 0;
    
    for (; !vec_reader.Done(); vec_reader.Next()) {
      std::string key = vec_reader.Key();
      const Vector<BaseFloat> &vec(vec_reader.Value());

      // find the nearest spker and get the transform matrix
      int32 spk_indx;
      if(development){
        // Check if element key exists in vector
        // extract first two substring of key, e.g. 1026-m
        std::string delimiter = "-";
        std::string token = key.substr(0, key.find(delimiter)+2); // token is "1026"+"-m/f" for sre

        std::vector<std::string>::iterator it = std::find(spk_list.begin(), spk_list.end(), token);
        if (it != spk_list.end())
          // std::cout << "Element Found" << std::endl;
          spk_indx = std::distance(spk_list.begin(), it);
        
        else{ 
          // token is "id00012" for voxceleb
          std::string token2 = key.substr(0, key.find(delimiter)); 
          std::vector<std::string>::iterator it2 = std::find(spk_list.begin(), spk_list.end(), token2);
          if(it2 != spk_list.end()){
            spk_indx = std::distance(spk_list.begin(), it2);
          }
          else
            KALDI_ERR << "Spk "<< key <<" is not in development set";
        }
      }
      else
      {
        // find the nearest spker
        BaseFloat dis = FindSpk(spk_mean, vec, spk_indx);
        // spk_indx = 0;
      }
      KALDI_LOG << "FindSpk " << key <<" : "<< spk_indx ;
      Matrix<BaseFloat> transform(transform_reader.Value(spk_list[spk_indx]));

      int32 transform_rows = transform.NumRows(),
          transform_cols = transform.NumCols(),
          vec_dim = vec.Dim();
      
      Vector<BaseFloat> vec_out(transform_rows);

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
      vec_writer.Write(key, vec_out);
      num_done++;
    }






    KALDI_LOG << "Applied transform to " << num_done << " vectors.";
    
    return (num_done != 0 ? 0 : 1);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
