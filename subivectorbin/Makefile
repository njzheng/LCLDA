
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = ivector-mean-var \
           ivector-compute-dot-products-lclda \
           ivector-compute-lclda-para ivector-compute-clda \
           transform-vec-lclda ivector-lclda-plda-scoring \
           ivector-compute-nda compute-eer-mdcf ivector-compute-lplda \
           ivector-compute-lcplda-para ivector-compute-lcplda \
           ivector-compute-lclda-pdf ivector-compute-lcplda-pdf \
           ivector-compute-plda-set ivector-lclda-pldaset-scoring
           
           
OBJFILES =



TESTFILES =


ADDLIBS = ../subivector/kaldi-subivector.a ../ivector/kaldi-ivector.a  ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
