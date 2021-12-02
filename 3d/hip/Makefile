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
CXX=g++
CC=gcc
HIPCC=hipcc --x cu
# GPU_MEMORY=-DUNIFIED_MEMORY -D__HIP_PLATFORM_NVCC__ -Wno-deprecated-declarations
# GPU_MEMORY=-DPINNED_MEMORY -D__HIP_PLATFORM_NVCC__ -Wno-deprecated-declarations
HIPFLAGS=-O3 -DNO_MPI $(GPU_MEMORY)
# CCFLAGS=-g -O3 -ffast-math -mfma -march=native -Wall -I$(COMMONDIR) -DNDEBUG -funsafe-math-optimizations -fopenmp 
#CCFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG -fopenmp 
CCFLAGS=-g -O3 -march=native -I$(COMMONDIR) -DNDEBUG -fopenmp -DNO_MPI $(GPU_MEMORY)
# CCFLAGS=-g -O3 -Wall -I$(COMMONDIR) -DNDEBUG -fno-tree-vectorize
LDFLAGS=-lcudart
LIBS=-lpng
endif

ifeq ($(COMP),cray)
CXX=CC
CC=cc
HIPCC=/opt/rocm-4.3.1/bin/hipcc
HIPFLAGS=-O0
# CCFLAGS=-g -O3 -ffast-math -mfma -march=native -Wall -I$(COMMONDIR) -DNDEBUG -funsafe-math-optimizations -fopenmp
#CCFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG -fopenmp
CCFLAGS=-g -O0 -march=native -I$(COMMONDIR) -DNDEBUG -DMPI_DATATYPES -DMPI_NEIGHBORHOOD -DDISABLE_PNG 
# CCFLAGS=-g -O3 -Wall -I$(COMMONDIR) -DNDEBUG -fno-tree-vectorize
LDFLAGS=-L/opt/rocm-4.3.1/lib -lamdhip64
# LIBS=-lpng
LIBS=
endif


ifeq ($(COMP),intel)
CXX=mpicxx
CC=gcc
CCFLAGS=-g -O3 -I$(COMMONDIR) -DNDEBUG
LDFLAGS=
LIBS=-lpng
endif

EXE=heat_hip
OBJS=main.o heat.o core.o setup.o io.o utilities.o
OBJS_HIP=core_hip.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp matrix.hpp
core.o: core.cpp heat.hpp
core_hip.o: core_hip.cpp heat.hpp
setup.o: setup.cpp heat.hpp
io.o: io.cpp heat.hpp matrix.hpp
main.o: main.cpp heat.hpp

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
