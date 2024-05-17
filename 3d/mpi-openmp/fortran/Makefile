ifeq ($(COMP),)
COMP=pgi
endif

COMMONDIR=../../../common

ifeq ($(COMP),gnu)
FC=mpif90
CC=gcc
FCFLAGS=-O3 -Wall -fopenmp
FCFLAGS=-O3 -Wall 
CCFLAGS=-O3 -Wall -I$(COMMONDIR)
LDFLAGS=-fopenmp
LIBS=
endif

ifeq ($(COMP),cray)
FC=ftn
CC=gcc
FCFLAGS=-g -DDISABLE_PNG -DGPU_MPI -fopenmp
CCFLAGS=-O3 -g -Wall -I$(COMMONDIR)
LDFLAGS=-fopenmp
# LIBS=
endif

EXE=heat_hybrid
OBJS=main.o heat_mod.o core.o setup.o utilities.o io.o
# OBJS_PNG=$(COMMONDIR)/pngwriter.o pngwriter_mod.o

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
heat_mod.o: heat_mod.F90
core.o: core.F90 heat_mod.o
utilities.o: utilities.F90 heat_mod.o
io.o: io.F90 heat_mod.o pngwriter_mod.o 
setup.o: setup.F90 heat_mod.o utilities.o io.o
pngwriter_mod.o: pngwriter_mod.F90 heat_mod.o
main.o: main.F90 heat_mod.o core.o io.o setup.o utilities.o

$(EXE): $(OBJS) $(OBJS_PNG)
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
