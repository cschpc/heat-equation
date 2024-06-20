ifeq ($(COMP),)
COMP=cray
endif

export HIPFORT_HOME=./hipfort/hipfortbin
include Makefile.hipfort 



EXE=heat_hipfort
OBJS=main.o kernel.o heat_mod.o core.o utilities.o io.o setup.o 

all: $(EXE)

hipfort:
	@git clone https://github.com/ROCmSoftwarePlatform/hipfort.git

	
./hipfort/hipfortbin/lib/libhipfort-amdgcn.a: hipfort
	@echo "running the command"
	@bash install_hipfort.sh $(COMP)

kernel.o: kernel.cpp ./hipfort/hipfortbin/lib/libhipfort-amdgcn.a
heat_mod.o: heat_mod.F90 ./hipfort/hipfortbin/lib/libhipfort-amdgcn.a
core.o: core.F90 heat_mod.o ./hipfort/hipfortbin/lib/libhipfort-amdgcn.a
utilities.o: utilities.F90 heat_mod.o ./hipfort/hipfortbin/lib/libhipfort-amdgcn.a
io.o: io.F90 heat_mod.o ./hipfort/hipfortbin/lib/libhipfort-amdgcn.a
setup.o: setup.F90 heat_mod.o utilities.o io.o ./hipfort/hipfortbin/lib/libhipfort-amdgcn.a
main.o: main.F90 heat_mod.o core.o io.o setup.o utilities.o ./hipfort/hipfortbin/lib/libhipfort-amdgcn.a



$(EXE): $(OBJS) hipfort/hipfortbin/lib/libhipfort-amdgcn.a
	$(FC) $(OBJS) -o $@ $(LDFLAGS) $(LIBS) $(LINKOPTS)

%.o: %.cpp
	$(CXX) -c $< -o $@

%.o: %.F90
	$(FC) -DGPU_MPI -c $< -o $@


.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
