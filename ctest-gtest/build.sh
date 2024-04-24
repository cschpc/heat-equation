#!/usr/bin/bash

if [ $# -eq 0 ] || [ "$1" != "Release" -a "$1" != "Debug" ]
then
    echo "Give \"Release\" or \"Debug\" as argumet"
    exit 1
fi

if [ $# -eq 0 ] || [ ! -d "$2" ]
then
    echo "Give the base directory of the project"
    exit 1
fi

build_type=$1
base_dir=$(realpath $2)

source_dir=$base_dir
build_dir=$base_dir/build/$build_type
install_dir=$base_dir

mkdir -p $build_dir
mkdir -p $install_dir

if cmake \
    -B $build_dir \
    -S $source_dir \
    -DCMAKE_BUILD_TYPE:STRING=$build_type \
    -DCMAKE_INSTALL_PREFIX:PATH=$install_dir \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ; then
    if cmake \
        --build $build_dir \
        --config $build_type \
        --target install -j8 ; then
        cp $build_dir/compile_commands.json $source_dir/
    fi
fi

