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
CCFLAGS=-g -O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

ifeq ($(COMP),intel)
CXX=mpicxx
CC=gcc
CCFLAGS=-g -O3 -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng
endif

EXE=heat_mpi
OBJS=main.o core.o setup.o utilities.o io.o 
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


core.o: core.cpp heat.h
utilities.o: utilities.cpp heat.h
setup.o: setup.cpp heat.h
io.o: io.cpp heat.h
main.o: main.cpp heat.h

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
