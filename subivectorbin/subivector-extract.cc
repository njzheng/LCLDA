// subivectorbin/subivector-extract.cc

// Copyright 2013  Daniel Povey

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
#include "gmm/am-diag-gmm.h"
// #include "ivector/ivector-extractor.h"
#include "subivector/subivector-extractor.h"
#include "util/kaldi-thread.h"
#include <string.h>

namespace kaldi {

// This class will be used to parallelize over multiple threads the job
// that this program does.  The work happens in the operator (), the
// output happens in the destructor.
class SubIvectorExtractTask {
 public:
  SubIvectorExtractTask(const SubIvectorExtractor &extractor,
                     std::string utt,
                     const Matrix<BaseFloat> &feats,
                     const Posterior &posterior,
                     BaseFloatMatrixWriter *writer,
                     BaseFloatVectorWriter *gamma_writer,
                     double *tot_auxf_change,
                     const std::vector<int32> &phone_triphone_list,
                     Vector<double> *avg_phone_gamma):
      extractor_(extractor), utt_(utt), feats_(feats), posterior_(posterior),
      tot_auxf_change_(tot_auxf_change), writer_(writer), gamma_writer_(gamma_writer),
      ph_tri_list_(phone_triphone_list),avg_phone_gamma_(avg_phone_gamma) { }

  void operator () () {
    bool need_2nd_order_stats = false;

    SubIvectorExtractorUtteranceStats utt_stats(extractor_.NumGauss(),
                                             extractor_.FeatDim(),
                                             need_2nd_order_stats);

    utt_stats.AccStats(feats_, posterior_);

    // record the gamma for each phone in utt_stats
    phone_gamma_.Resize(extractor_.PhoneDim());
    utt_stats.Get_phone_gamma(phone_gamma_,ph_tri_list_);

    if(avg_phone_gamma_->Dim()==0)
      avg_phone_gamma_->Resize(extractor_.PhoneDim());
    // the sum of each frame
    avg_phone_gamma_->AddVec(1.0, phone_gamma_);

    // ivector_.Resize(extractor_.IvectorDim());
    // ivector_(0) = extractor_.PriorOffset();

    // each row is a subivector
    subivector_.Resize(extractor_.PhoneDim(), extractor_.IvectorDim());
    // get the offset vector from extractor
    Vector<double> offset_vec;
    phone_num_ = extractor_.PhoneDim();
    offset_vec.Resize(phone_num_);
    extractor_.GetPriorOffsetVec(&offset_vec);
    //Copy vector into specific column of matrix.
    subivector_.CopyColFromVec(offset_vec,0); 

    // use empty vmat to denote no vmat
    std::vector< SpMatrix<double> > empty_vmat;
    empty_vmat.resize(0);

    if (tot_auxf_change_ != NULL) {
      double old_auxf = 0.0;//extractor_.GetAuxf(utt_stats, ivector_);
      extractor_.GetSubIvectorDistribution(utt_stats, subivector_, empty_vmat,
        ph_tri_list_, phone_num_ );
      double new_auxf = 0.0;//extractor_.GetAuxf(utt_stats, ivector_);
      auxf_change_ = new_auxf - old_auxf;
    } else {
      // need load ph_tri_list_ and phone_num_
      extractor_.GetSubIvectorDistribution(utt_stats, subivector_, empty_vmat,
        ph_tri_list_, phone_num_ );
    }
  }
  ~SubIvectorExtractTask() {
    if (tot_auxf_change_ != NULL) {
      double T = TotalPosterior(posterior_);
      *tot_auxf_change_ += auxf_change_;
      KALDI_VLOG(2) << "Auxf change for utterance " << utt_ << " was "
                    << (auxf_change_ / T) << " per frame over " << T
                    << " frames (weighted)";
    }
    // We actually write out the offset of the iVectors from the mean of the
    // prior distribution; this is the form we'll need it in for scoring.  (most
    // formulations of iVectors have zero-mean priors so this is not normally an
    // issue).

    // ivector_(0) -= extractor_.PriorOffset();
    // KALDI_VLOG(2) << "Ivector norm for utterance " << utt_
    //               << " was " << ivector_.Norm(2.0);
    // writer_->Write(utt_, Vector<BaseFloat>(ivector_));

    Vector<double> offset_vec;
    phone_num_ = extractor_.PhoneDim();
    offset_vec.Resize(phone_num_);
    extractor_.GetPriorOffsetVec(&offset_vec);
    double T = TotalPosterior(posterior_);

    KALDI_VLOG(2) << "SubIvector for utterance " << utt_
              << " the augmented frame number was "<< T ;
    for (int32 i=0; i< phone_num_; i++){
      subivector_(i,0) -= offset_vec(i);
    //   KALDI_VLOG(2) << subivector_.Row(i).Norm(2.0)<<" ";
    }
    writer_->Write(utt_, Matrix<BaseFloat>(subivector_));
    gamma_writer_->Write(utt_, Vector<BaseFloat>(phone_gamma_));
  }
 private:
  const SubIvectorExtractor &extractor_;
  std::string utt_;
  Matrix<BaseFloat> feats_;
  Posterior posterior_;
  // BaseFloatVectorWriter *writer_;

  double *tot_auxf_change_; // if non-NULL we need the auxf change.
  // Vector<double> ivector_;
  double auxf_change_;

  // newly added
  BaseFloatMatrixWriter *writer_;
  BaseFloatVectorWriter *gamma_writer_;
  
  Matrix<double> subivector_;
  Vector<double> phone_gamma_;
  Vector<double> *avg_phone_gamma_;
  
  std::vector<int32> ph_tri_list_;
  int32 phone_num_;
};

}

int main(int argc, char *argv[]) 
{
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Extract subiVectors for utterances, using a trained subiVector extractor,\n"
        "and features and Gaussian-level posteriors\n"
        "Usage:  subivector-extract [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <phone-triphone-list> <subivector-wspecifier>\n"
        "e.g.: \n"
        " fgmm-global-gselect-to-post 1.ubm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
        "  subivector-extract final.ie '$feats' ark,s,cs:- list ark,t:subivectors.1.ark\n";

    ParseOptions po(usage);
    bool compute_objf_change = false; // the original is true
    SubIvectorEstimationOptions opts;
    std::string spk2utt_rspecifier;
    TaskSequencerConfig sequencer_config;
    po.Register("compute-objf-change", &compute_objf_change,
                "If true, compute the change in objective function from using "
                "nonzero iVector (a potentially useful diagnostic).  Combine "
                "with --verbose=2 for per-utterance information");
    po.Register("spk2utt", &spk2utt_rspecifier, "Supply this option if you "
                "want iVectors to be output at the per-speaker level, estimated "
                "using stats accumulated from multiple utterances.  Note: this "
                "is not the normal way iVectors are obtained for speaker-id. "
                "This option will cause the program to ignore the --num-threads "
                "option.");

    double agument_time = 1.0;
    po.Register("agument", &agument_time, "Supply this option if you "
                "want short duration utt post to be enlarged .");

    opts.Register(&po);
    sequencer_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string subivector_extractor_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posterior_rspecifier = po.GetArg(3),
        phone_triphone_rspecifier = po.GetArg(4), // the new added
        subivectors_wspecifier = po.GetArg(5);
    // record the post sum for each utt 
    std::string subivectors_gamma_wspecifier = po.GetArg(6);

    // This is for align triphones into phones, only a vector 
    // where list[triphone_index] = phone_index
    // [ 1.1 2.0 3.4 ]\n
    bool list_binary = false;
    std::ifstream is(phone_triphone_rspecifier); // maybe need <fstream.h>
    Vector<float> phone_triphone_list_f;
    phone_triphone_list_f.Read(is, list_binary);
    KALDI_LOG << "Read " << phone_triphone_rspecifier << " files\n ";

    // convert float into int32
    std::vector<int32> phone_triphone_list;
    for(int32 i=0; i<phone_triphone_list_f.Dim();i++ ){
      phone_triphone_list.push_back(static_cast<int32>(phone_triphone_list_f(i) ) );
      // char buffer [33];
      // std::sprintf (buffer, "%d", phone_triphone_list[i]);
      // KALDI_LOG << buffer <<" ";
    }
    // KALDI_LOG<<"\n";




    if (spk2utt_rspecifier.empty()) {
      // g_num_threads affects how ComputeDerivedVars is called when we read the
      // extractor.
      g_num_threads = sequencer_config.num_threads;
      SubIvectorExtractor extractor;
      ReadKaldiObject(subivector_extractor_rxfilename, &extractor);

      double tot_auxf_change = 0.0, tot_t = 0.0;
      int32 num_done = 0, num_err = 0;

      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
      // BaseFloatVectorWriter subivector_writer(subivectors_wspecifier);
      BaseFloatMatrixWriter subivector_writer(subivectors_wspecifier);

      BaseFloatVectorWriter subivector_gamma_writer(subivectors_gamma_wspecifier);
      // store the average phone gamma
      Vector<double> avg_phone_gamma;

      {
        TaskSequencer<SubIvectorExtractTask> sequencer(sequencer_config);
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          if (!posterior_reader.HasKey(utt)) {
            KALDI_WARN << "No posteriors for utterance " << utt;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &mat = feature_reader.Value();
          Posterior posterior = posterior_reader.Value(utt);

          if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
            KALDI_WARN << "Size mismatch between posterior " << posterior.size()
                       << " and features " << mat.NumRows() << " for utterance "
                       << utt;
            num_err++;
            continue;
          }

          double *auxf_ptr = (compute_objf_change ? &tot_auxf_change : NULL );

          double this_t = opts.acoustic_weight * TotalPosterior(posterior),
              max_count_scale = 1.0;
          if (opts.max_count > 0 && this_t > opts.max_count) {
            max_count_scale = opts.max_count / this_t;
            KALDI_LOG << "Scaling stats for utterance " << utt << " by scale "
                      << max_count_scale << " due to --max-count="
                      << opts.max_count;
            this_t = opts.max_count;
          }
          
          // ScalePosterior(opts.acoustic_weight * max_count_scale, &posterior);
          KALDI_VLOG(2) << "SubIvector for utterance " << utt
              << " the orginal frame number was "<< this_t ;

          // for short utt, agument its frame with time * 18, 
          // make the ivector distribution like long utterance
          if (this_t < 1500)
            ScalePosterior(agument_time, &posterior);
          // note: now, this_t == sum of posteriors.

          sequencer.Run(new SubIvectorExtractTask(extractor, utt, mat, posterior,
                                       &subivector_writer, &subivector_gamma_writer, 
                                       auxf_ptr, phone_triphone_list, &avg_phone_gamma));

          tot_t += this_t;
          num_done++;
        }
        // Destructor of "sequencer" will wait for any remaining tasks.
      }


      // cal the phone_gamma for the whole set 
      double tot_frame = avg_phone_gamma.Sum();
      avg_phone_gamma.Scale(1.0/tot_frame);
      subivector_gamma_writer.Write("avg_phone_gamma", Vector<BaseFloat>(avg_phone_gamma));


      KALDI_LOG << "Done " << num_done << " files, " << num_err
                << " with errors.  Total (weighted) frames " << tot_t;
      if (compute_objf_change)
        KALDI_LOG << "Overall average objective-function change from estimating "
                  << "ivector was " << (tot_auxf_change / tot_t) << " per frame "
                  << " over " << tot_t << " (weighted) frames.";

      return (num_done != 0 ? 0 : 1);
    } else {
      KALDI_ASSERT(sequencer_config.num_threads == 1 &&
                   "--spk2utt option is incompatible with --num-threads option");
      KALDI_ERR<<"Do not have RunPerSpeaker function for subivetor";
      return 0;
      //RunPerSpeaker(subivector_extractor_rxfilename,
                           // opts,
                           // compute_objf_change,
                           // spk2utt_rspecifier,
                           // feature_rspecifier,
                           // posterior_rspecifier,
                           // subivectors_wspecifier);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


