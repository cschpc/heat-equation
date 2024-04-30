#pragma once
#include <algorithm>
#include <cassert>
#include <vector>

// Generic 2D matrix array class.
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
    const int indx(int i, int j) const {
        //assert that indices are reasonable
        assert(i >= 0 && i < num_rows);
        assert(j >= 0 && j < num_cols);

        return i * num_cols + j;
    }

public:

    // matrix dimensions
  int num_rows = 0;
  int num_cols = 0;

  // Default constructor
  Matrix() = default;
  // Allocate at the time of construction
  Matrix(int num_rows, int num_cols) : num_rows(num_rows), num_cols(num_cols) {
      _data.resize(num_rows * num_cols);
  };

  Matrix(std::vector<T> &&data, int num_rows, int num_cols)
      : _data(std::move(data)), num_rows(num_rows), num_cols(num_cols) {}

  void allocate(int nr, int nc) {
      num_rows = nr;
      num_cols = nc;
      _data.resize(num_rows * num_cols);
  };

    // standard (i,j) syntax for setting elements
    T& operator()(int i, int j) {
        return _data[ indx(i, j) ];
    }

    // standard (i,j) syntax for getting elements
    const T& operator()(int i, int j) const {
        return _data[ indx(i, j) ];
    }

    // provide possibility to get raw pointer for data at index (i,j) (needed for MPI)
    T *data(int i = 0, int j = 0) { return _data.data() + i * num_cols + j; }

    static Matrix<T> make_with_ghost_layers(std::vector<T> &&data, int num_rows,
                                            int num_cols) {
        return make_with_ghost_layers(
            Matrix<T>(std::move(data), num_rows, num_cols));
    }

    static Matrix<T> make_with_ghost_layers(Matrix<T> &&m) {
        const int num_rows = m.num_rows + 2;
        const int num_cols = m.num_cols + 2;
        std::vector<double> data(num_rows * num_cols);

        for (int i = 1; i < m.num_rows + 1; i++) {
            for (int j = 1; j < m.num_cols + 1; j++) {
                const int index = i * num_cols + j;
                data[index] = m(i - 1, j - 1);
            }
        }

        for (int i = 0; i < num_rows; i++) {
            const int first = i * num_cols;
            const int last = (i + 1) * num_cols - 1;
            const int inner_i = std::max(0, std::min(m.num_rows - 1, i - 1));
            // Left
            data[first] = m(inner_i, 0);
            // Right
            data[last] = m(inner_i, m.num_cols - 1);
        }

        for (int j = 0; j < num_cols; j++) {
            const int first = j;
            const int last = data.size() - num_cols + j;
            const int inner_j = std::max(0, std::min(m.num_cols - 1, j - 1));
            // Top
            data[first] = m(0, inner_j);
            // Bottom
            data[last] = m(m.num_rows - 1, inner_j);
        }

        return Matrix<T>(std::move(data), num_rows, num_cols);
    }
};
