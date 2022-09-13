ifeq ($(COMP),)
COMP=cray
endif

COMMONDIR=../../common

ifeq ($(COMP),pgi)
CXX=mpicxx
CC=pgcc
CCFLAGS=-g -O3 -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng 
endif

ifeq ($(COMP),gnu)
CXX=mpicxx
CC=gcc
# CCFLAGS=-g -O3 -ffast-math -mfma -march=native -Wall -I$(COMMONDIR) -DNDEBUG -funsafe-math-optimizations -fopenmp 
#CCFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG -fopenmp 
CCFLAGS=-g -O3 -march=native -I$(COMMONDIR) -DNDEBUG -fopenmp  -DMPI_DATATYPES -DMPI_NEIGHBORHOOD
# CCFLAGS=-g -O3 -Wall -I$(COMMONDIR) -DNDEBUG -fno-tree-vectorize
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),cray)
CXX=CC
CC=cc
# CCFLAGS=-g -O3 -ffast-math -mfma -march=native -Wall -I$(COMMONDIR) -DNDEBUG -funsafe-math-optimizations -fopenmp
#CCFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG -fopenmp
#OMPFLAGS=-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908
OMPFLAGS=-fopenmp
CCFLAGS=-g -O3 -march=native -I$(COMMONDIR) -DNDEBUG -DDISABLE_PNG $(OMPFLAGS)
# CCFLAGS=-g -O3 -Wall -I$(COMMONDIR) -DNDEBUG -fno-tree-vectorize
LDFLAGS=--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64
# LIBS=-lpng
endif


ifeq ($(COMP),intel)
CXX=mpicxx
CC=gcc
CCFLAGS=-g -O3 -I$(COMMONDIR) -DNDEBUG
LDFLAGS=
LIBS=-lpng
endif

EXE=heat_offload
OBJS=main.o heat.o core.o setup.o io.o utilities.o
# OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp parallel.hpp matrix.hpp
core.o: core.cpp parallel.hpp heat.hpp
setup.o: setup.cpp parallel.hpp heat.hpp
io.o: io.cpp heat.hpp parallel.hpp matrix.hpp
main.o: main.cpp parallel.hpp heat.hpp

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
