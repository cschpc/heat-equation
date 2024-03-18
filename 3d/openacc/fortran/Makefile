ifeq ($(COMP),)
COMP=cray
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

ifeq ($(COMP),gnu-lumi)
FC=ftn
CC=gcc
#FCFLAGS=-DDISABLE_PNG -DGPU_MPI -hacc -g -G0 -O0
FCFLAGS=-DDISABLE_PNG -fopenacc -O3 -DGPU_MPI
CCFLAGS=-Wall -I$(COMMONDIR) -g 
LDFLAGS=-fopenacc -g -Wl,-rpath -Wl,${LD_LIBRARY_PATH}
# LIBS=-lpng
endif

ifeq ($(COMP),cray)
FC=ftn
CC=gcc
#FCFLAGS=-DDISABLE_PNG -DGPU_MPI -hacc -g -G0 -O0
FCFLAGS=-DDISABLE_PNG -hacc -O3 -DGPU_MPI
CCFLAGS=-Wall -I$(COMMONDIR) -g 
LDFLAGS=-hacc -g -Wl,-rpath -Wl,${LD_LIBRARY_PATH}
# LIBS=-lpng
LIBS=-lroctx64 -lroctracer64
endif

EXE=heat_openacc
OBJS=main.o heat_mod.o core.o setup.o utilities.o io.o mo_roctx.o
# OBJS_PNG=$(COMMONDIR)/pngwriter.o pngwriter_mod.o

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
heat_mod.o: heat_mod.F90
core.o: core.F90 heat_mod.o 
utilities.o: utilities.F90 heat_mod.o
io.o: io.F90 heat_mod.o pngwriter_mod.o 
setup.o: setup.F90 heat_mod.o utilities.o io.o
pngwriter_mod.o: pngwriter_mod.F90 heat_mod.o
main.o: main.F90 heat_mod.o core.o io.o setup.o utilities.o mo_roctx.o
mo_roctx.o: mo_roctx.F90

$(EXE): $(OBJS) $(OBJS_PNG)
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
