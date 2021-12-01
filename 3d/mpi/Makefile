ifeq ($(COMP),)
COMP=gnu
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
CCFLAGS=-g -O3 -march=native -I$(COMMONDIR) -DNDEBUG -fopenmp  -DMPI_DATATYPES -DMPI_NEIGHBORHOOD -DDISABLE_PNG
# CCFLAGS=-g -O3 -Wall -I$(COMMONDIR) -DNDEBUG -fno-tree-vectorize
LDFLAGS=
# LIBS=-lpng
endif


ifeq ($(COMP),intel)
CXX=mpicxx
CC=gcc
CCFLAGS=-g -O3 -I$(COMMONDIR) -DNDEBUG
LDFLAGS=
LIBS=-lpng
endif

EXE=heat_mpi
OBJS=main.o heat.o core.o setup.o io.o utilities.o
# OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp matrix.hpp
core.o: core.cpp heat.hpp
setup.o: setup.cpp heat.hpp
io.o: io.cpp heat.hpp matrix.hpp
main.o: main.cpp heat.hpp

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
