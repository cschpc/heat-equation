ifeq ($(COMP),)
COMP=cray
endif

COMMONDIR=../../common

ifeq ($(COMP),pgi)
CXX=mpicxx
CC=pgcc
CCFLAGS=-g -O3 -I$(COMMONDIR)
LDFLAGS=
LIBS=
endif

ifeq ($(COMP),gnu)
CXX=CC
CC=gcc
HIPCC=hipcc 
# GPU_MEMORY=-DUNIFIED_MEMORY -D__HIP_PLATFORM_NVCC__ -Wno-deprecated-declarations
# GPU_MEMORY=-DPINNED_MEMORY -D__HIP_PLATFORM_NVCC__ -Wno-deprecated-declarations
HIPFLAGS=-O3 -DNO_MPI $(GPU_MEMORY)
GPU_MEMORY=-DPINNED_MEMORY -D__HIP_PLATFORM_AMD__
# CCFLAGS=-g -O3 -ffast-math -mfma -march=native -Wall -I$(COMMONDIR) -DNDEBUG -funsafe-math-optimizations -fopenmp 
#CCFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG -fopenmp 
CCFLAGS=-g -O3 -march=native -I$(COMMONDIR) -DNDEBUG -fopenmp -DNO_MPI $(GPU_MEMORY)
# CCFLAGS=-g -O3 -Wall -I$(COMMONDIR) -DNDEBUG -fno-tree-vectorize
LDFLAGS=-lamdhip64
# LIBS=
endif

ifeq ($(COMP),cray)
NOPNG=true
CXX=CC
CC=cc
# HIPCC=CC
HIPCC=hipcc
GPU_MEMORY=-DPINNED_MEMORY -D__HIP_PLATFORM_AMD__
# HIPFLAGS=$(GPU_MEMORY) -O3 -std=c++11 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip -DUSE_STREAMS
HIPFLAGS=$(GPU_MEMORY) -O3 -std=c++11 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -DUSE_STREAMS 
# CCFLAGS=-g -O3 -ffast-math -mfma -march=native -Wall -I$(COMMONDIR) -DNDEBUG -funsafe-math-optimizations -fopenmp
#CCFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG -fopenmp
CCFLAGS=-g -O3 -march=native -I$(COMMONDIR) -DNDEBUG -fopenmp -DNDEBUG -DDISABLE_PNG $(GPU_MEMORY)
# CCFLAGS=-g -O3 -Wall -I$(COMMONDIR) -DNDEBUG -fno-tree-vectorize
# LDFLAGS=--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64
LDFLAGS=-L${ROCM_PATH}/lib -lamdhip64
# LIBS=
LIBS=
endif


ifeq ($(COMP),intel)
CXX=mpicxx
CC=gcc
CCFLAGS=-g -O3 -I$(COMMONDIR) -DNDEBUG
LDFLAGS=
LIBS=
endif

EXE=heat_hip
OBJS=main.o heat.o core.o setup.o io.o utilities.o
OBJS_HIP=core_hip.o
ifneq ($(NOPNG),true)
OBJS_PNG=$(COMMONDIR)/pngwriter.o
endif

all: $(EXE)


utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp parallel.hpp matrix.hpp
core.o: core.cpp parallel.hpp heat.hpp
core_hip.o: core_hip.cpp heat.hpp
setup.o: setup.cpp parallel.hpp heat.hpp
io.o: io.cpp heat.hpp parallel.hpp matrix.hpp
main.o: main.cpp parallel.hpp heat.hpp

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)
$(OBJS): FLAGS := $(CCFLAGS)
$(OBJS_HIP): C_COMPILER := $(HIPCC)
$(OBJS_HIP): FLAGS := $(HIPFLAGS)


$(EXE): $(OBJS) $(OBJS_HIP) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_HIP) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(C_COMPILER) $(FLAGS) -c $< -o $@

%.o: %.c
	$(C_COMPILER) $(FLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
