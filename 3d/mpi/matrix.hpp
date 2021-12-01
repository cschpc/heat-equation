#pragma once
#include <vector>
#include <cassert>

// Generic 3D matrix array class.
//
// Internally data is stored in 1D vector but is
// accessed using index function that maps i and j
// indices to an element in the flat data vector.
// Row major storage is used
// For easier usage, we overload parentheses () operator
// for accessing matrix elements in the usual (i,j)
// format.

template<typename T>
class Matrix
{

private:

    // Internal storage
    std::vector<T> _data;

    // Internal 1D indexing
    const int indx(int i, int j, int k) const {
        //assert that indices are reasonable
        assert(i >= 0 && i <  nx);
        assert(j >= 0 && j <  ny);
        assert(k >= 0 && k <  nz);

        return i * ny * nz + j * nz + k;
    }

public:

    // matrix dimensions
    int nx, ny, nz;

    // Default constructor
    Matrix() = default;
    // Allocate at the time of construction
    Matrix(int nx, int ny=1, int nz=1) : nx(nx), ny(ny), nz(nz) {
        _data.resize(nx * ny * nz);
    };

    void allocate(int nx_in=1, int ny_in=1, int nz_in=1) {
        nx = nx_in;
        ny = ny_in;
        nz = nz_in;
        _data.resize(nx * ny * nz);
    };

    // standard (i,j) syntax for setting elements
    T& operator()(int i=0, int j=0, int k=0) {
        return _data[ indx(i, j, k) ];
    }

    // standard (i,j) syntax for getting elements
    const T& operator()(int i=0, int j=0, int k=0) const {
        return _data[ indx(i, j, k) ];
    }

    // provide possibility to get raw pointer for data at index (i,j,k) (needed for MPI)
    T* data(int i=0, int j=0, int k=0) {return _data.data() + i * ny * nz + j * nz + k;}

};
