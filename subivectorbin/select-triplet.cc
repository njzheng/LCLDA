// subivectorbin/select-triplet.cc

// Copyright 2013-2014  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
// modified by z3c
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
#include <algorithm>
#include <map>


template<typename KeyType, typename ValueType> 
std::pair<KeyType,ValueType> get_max( const std::map<KeyType,ValueType>& x ) {
  using pairtype=std::pair<KeyType,ValueType>; 
  return *std::max_element(x.begin(), x.end(), 
    [] (const pairtype & p1, const pairtype & p2) {
        return p1.second < p2.second;
      }); 
}



int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Usage: select-triplet --k-nearest $k_nearest <spk2utt-rspecifier> <ivector-rspecifier> \n" 
        "<positive_spk_utt> <negative_spk_utt>\n"
        "e.g.: select-triplet --k-nearest 10 ark:$data/spk2utt \n"
        "scp:$dir/phone_vectors.$j.scp ark:$dir/spk_mv.$j.ark \n"
        "ark:$dir/spk2utt_positive ark:$dir/spk2utt_negative \n";

    ParseOptions po(usage);

    int32 k_nearest = 8;

    po.Register("knearest", &k_nearest, "The times to find negative samples");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 5 ) {
      po.PrintUsage();
      exit(1);
    }


    std::string spk2utt_rspecifier = po.GetArg(1),
        ivector_rspecifier = po.GetArg(2),
        mv_rspecifier = po.GetArg(3),
        spk2utt_wspecifier_pos = po.GetArg(4),
        spk2utt_wspecifier_neg = po.GetArg(5);


    // double spk_sumsq = 0.0;
    Vector<BaseFloat> spk_sum;

    int32 num_spk_done = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    
    RandomAccessBaseFloatMatrixReader mv_reader(mv_rspecifier);
    TokenVectorWriter spk2utt_writer_pos(spk2utt_wspecifier_pos);
    TokenVectorWriter spk2utt_writer_neg(spk2utt_wspecifier_neg);

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      // different utterances
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      if (uttlist.empty()) {
        KALDI_ERR << "Speaker with no utterances.";
      }


      KALDI_LOG << "start processing spk and its positive samples "<< spk;
      //record the positive utt samples for spk
      std::vector<std::string> uttlist_pos;

      int32 utt_count_pos = 0;

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No subiVector present in input for utterance " << utt;
        } else {

          // record 
          uttlist_pos.push_back(utt);
          utt_count_pos++;
        }
      }

      spk2utt_writer_pos.Write(spk, uttlist_pos);


      // ===================================================================


      //record the negative utt samples for spk
      std::vector<std::string> uttlist_neg;
      std::map<std::string, BaseFloat> map_utt_neg;
      std::pair<std::string, BaseFloat> utt_neg_max_pair; //record the utt name with largest distance

      Vector<BaseFloat> spk_mean;      
      Vector<BaseFloat> spk_std_var;

      // ignore this speaker
      if(utt_count_pos == 0){
        KALDI_WARN << "No positive utt sample for spk " << spk;
        continue;
      }


      int32 utt_count_neg = utt_count_pos * k_nearest;
      KALDI_LOG << "start processing spk and its negative samples "<< spk 
                << " with " << utt_count_neg;



      if (!mv_reader.HasKey(spk)){
        // may be all zeros
        KALDI_WARN << "No mean and variance in input for spk " << spk;
        continue;
      }else{
        Matrix<BaseFloat> spk_mv(mv_reader.Value(spk));
        if(spk_mean.Dim() == 0){
          spk_mean.Resize(spk_mv.NumCols());
          spk_std_var.Resize(spk_mv.NumCols());
        }
        spk_mean.CopyRowFromMat(spk_mv,0);
        spk_std_var.CopyRowFromMat(spk_mv,1);
        spk_std_var.ApplyPow(0.5);

      }

      // each time renew a reader
      SequentialBaseFloatVectorReader seq_ivector_reader(ivector_rspecifier);

      // compare mean with all phone vectors
      for (; !seq_ivector_reader.Done(); seq_ivector_reader.Next()) {
        std::string utt_temp = seq_ivector_reader.Key();

        // except the spk utt
        if (std::find(uttlist_pos.begin(), uttlist_pos.end(), utt_temp) != uttlist_pos.end())
          continue;

        Vector<BaseFloat> phone_vec_temp(seq_ivector_reader.Value());

        // calculate the whiten distance between vec and mean
        phone_vec_temp.AddVec(-1.0,spk_mean);
        phone_vec_temp.DivElements(spk_std_var);
        BaseFloat distance= phone_vec_temp.Norm(2.0);
        
        // find the minimine utt_count_neg utt pair
        if(map_utt_neg.size() < utt_count_neg -1 ){
          std::pair<std::string, BaseFloat> temp_pair(utt_temp, distance);
          map_utt_neg.insert(temp_pair);
        }
        else if(map_utt_neg.size() == utt_count_neg -1){
          std::pair<std::string, BaseFloat> temp_pair(utt_temp, distance);
          map_utt_neg.insert(temp_pair);

          // find the key with the largest value
          utt_neg_max_pair = get_max(map_utt_neg);
        }
        else{
          if(utt_neg_max_pair.second > distance){
            map_utt_neg.erase(utt_neg_max_pair.first);
            map_utt_neg.insert(std::pair<std::string, BaseFloat>(utt_temp,distance));
          
            // find the key with the largest value
             utt_neg_max_pair = get_max(map_utt_neg);
          }
        }

      }


      // record the negative utt 
      if(map_utt_neg.size()>utt_count_neg){
        KALDI_ERR<<"The map size for spk "<< spk <<"is larger than designed " 
        << utt_count_neg;
      }
      // convert map into vector
      for(auto iter = map_utt_neg.begin(); iter != map_utt_neg.end(); iter++){
        uttlist_neg.push_back(iter->first);
      }  
      spk2utt_writer_neg.Write(spk, uttlist_neg);
      num_spk_done++;

    }

    KALDI_LOG << "Process " << num_spk_done ;

    return (num_spk_done != 0 ? 0 : 1);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
