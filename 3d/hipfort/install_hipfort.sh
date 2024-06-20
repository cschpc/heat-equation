cd hipfort
mkdir build
cd build
echo " compiler is $1"
if ($1 == cray); then 
cmake .. -DHIPFORT_COMPILER=ftn -DHIPFORT_COMPILER_FLAGS="-ffree -e Z" -DCMAKE_INSTALL_PREFIX=../hipfortbin
elif ($1 == gnu); then 
cmake .. -DHIPFORT_COMPILER=gfortran -DCMAKE_Fortran_FLAGS=" -ffree-form -cpp -ffree-line-length-none -fmax-errors=5 -std=f2008 -fno-underscoring" -DCMAKE_INSTALL_PREFIX=../hipfortbin
fi

make 
make install
