ifeq ($(COMP),)
COMP=pgi
endif

COMMONDIR=../../../common

ifeq ($(COMP),pgi)
FC=mpif90
CC=gcc
FCFLAGS=-O3 -acc
CCFLAGS=-O3 -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),gnu)
FC=mpif90
CC=gcc
# FCFLAGS=-O3 -Wall -fopenacc
FCFLAGS=-O3 -Wall 
CCFLAGS=-O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),cray)
NOPNG=true
FC=ftn
CC=gcc
HIPCC=hipcc
HIPFLAGS=-O3 -std=c++11 --offload-arch=gfx90a
FCFLAGS=-O3 -DDISABLE_PNG -DGPU_MPI -hacc
CCFLAGS=-O3 -Wall -I$(COMMONDIR)
LDFLAGS=-hacc # -lamdhip64
# LIBS=-lpng
endif

EXE=heat_openacc_hip
OBJS=main.o heat_mod.o core.o setup.o utilities.o io.o
OBJS_HIP=core_hip.o
ifneq ($(NOPNG),true)
OBJS_PNG=$(COMMONDIR)/pngwriter.o pngwriter_mod.o
endif

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
heat_mod.o: heat_mod.F90
core.o: core.F90 heat_mod.o
utilities.o: utilities.F90 heat_mod.o
io.o: io.F90 heat_mod.o pngwriter_mod.o 
setup.o: setup.F90 heat_mod.o utilities.o io.o
pngwriter_mod.o: pngwriter_mod.F90 heat_mod.o
main.o: main.F90 heat_mod.o core.o io.o setup.o utilities.o

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)
$(OBJS): FLAGS := $(CCFLAGS)
$(OBJS_HIP): C_COMPILER := $(HIPCC)
$(OBJS_HIP): FLAGS := $(HIPFLAGS)

$(EXE): $(OBJS) $(OBJS_HIP) $(OBJS_PNG)
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_HIP) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.cpp
	$(C_COMPILER) $(FLAGS) -c $< -o $@

%.o: %.c
	$(C_COMPILER) $(FLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
