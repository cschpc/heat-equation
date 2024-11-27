default: all

ifndef KOKKOS_PATH
  $(error "Please set KOKKOS_PATH")
endif

ifeq ($(COMP),)
COMP=cuda
endif

COMMONDIR=../common

ifeq ($(COMP),cuda)
CXX=mpicxx
export OMPI_CXX=${KOKKOS_PATH}/bin/nvcc_wrapper
CC=gcc
CXXFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG
CCFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG
#CCFLAGS=-g -O0 -march=native -Wall -I$(COMMONDIR) 
LDFLAGS=
LIBS=-lpng

KOKKOS_DEVICES = "Cuda"
KOKKOS_ARCH = "Ampere80"
endif

ifeq ($(COMP),hip)
CXX=CC
CC=gcc
CXXFLAGS=-x hip -g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG
CCFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG
#CCFLAGS=-g -O0 -march=native -Wall -I$(COMMONDIR) 
LDFLAGS=
LIBS=-lpng

KOKKOS_DEVICES = "HIP"
KOKKOS_ARCH = "AMD_GFX90A"
endif

include $(KOKKOS_PATH)/Makefile.kokkos

EXE=heat_kokkos
OBJS=main.o heat.o core.o parallel.o setup.o io.o utilities.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o

all: $(EXE)

utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp matrix.hpp
core.o: core.cpp heat.hpp
parallel.o: parallel.cpp heat.hpp
setup.o: setup.cpp heat.hpp
io.o: io.cpp heat.hpp matrix.hpp
main.o: main.cpp heat.hpp

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS) $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: allclean clean
allclean: kokkos-clean
	-/bin/rm -f $(EXE) a.out *.o *.png *~

clean: 
	-/bin/rm -f $(EXE) a.out $(OBJS) $(OBJS_PNG) *.png *~
