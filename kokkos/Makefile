default: all

ifndef KOKKOS_PATH
  $(error "Please set KOKKOS_PATH")
endif

ifeq ($(COMP),)
COMP=gnu
endif

COMMONDIR=../common

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
CCFLAGS=-g -O3 -march=native -Wall -I$(COMMONDIR) -DNDEBUG
#CCFLAGS=-g -O0 -march=native -Wall -I$(COMMONDIR) 
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),intel)
CXX=mpicxx
CC=gcc
CCFLAGS=-g -O3 -I$(COMMONDIR) -DNDEBUG
LDFLAGS=
LIBS=-lpng
endif

KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "Zen2"
include $(KOKKOS_PATH)/Makefile.kokkos

EXE=heat_mpi
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
	$(CXX) $(CCFLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: allclean clean
allclean: kokkos-clean
	-/bin/rm -f $(EXE) a.out *.o *.png *~

clean: 
	-/bin/rm -f $(EXE) a.out *.o *.png *~
