all:

OPENFST_CXXFLAGS =
OPENFST_LDLIBS =
include ../kaldi.mk

TESTFILES = 

OBJFILES = subivector-extractor.o pldam.o

LIBNAME = kaldi-subivector

ADDLIBS = ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../ivector/kaldi-ivector.a 


include ../makefiles/default_rules.mk