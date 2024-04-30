#include "matrix.hpp"
#include <gtest/gtest.h>
#include <numeric>

TEST(matrix_test, construct_with_vector) {
    constexpr int nr = 20;
    constexpr int nc = 100;
    std::vector<double> v(nr * nc);
    std::iota(v.begin(), v.end(), 0);
    const Matrix<double> m(v, nr, nc);
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            const int index = i * nc + j;
            ASSERT_EQ(m(i, j), v[index]);
        }
    }
}

TEST(matrix_test, add_ghost_layers) {
    constexpr int nr = 20;
    constexpr int nc = 100;
    std::vector<double> v(nr * nc);
    std::iota(v.begin(), v.end(), 0);
    Matrix<double> n(v, nr, nc);
    Matrix<double> m = Matrix<double>::make_with_ghost_layers(n);

    ASSERT_EQ(m.num_rows, nr + 2);
    ASSERT_EQ(m.num_cols, nc + 2);

    for (int i = 1; i < nr + 1; i++) {
        for (int j = 1; j < nc + 1; j++) {
            const int index = (i - 1) * nc + j - 1;
            ASSERT_EQ(m(i, j), v[index]);
        }
    }
}

TEST(matrix_test, add_ghost_layers_from_raw) {
    constexpr int nr = 20;
    constexpr int nc = 100;
    std::vector<double> v(nr * nc);
    std::iota(v.begin(), v.end(), 0);
    Matrix<double> m = Matrix<double>::make_with_ghost_layers(v, nr, nc);

    ASSERT_EQ(m.num_rows, nr + 2);
    ASSERT_EQ(m.num_cols, nc + 2);

    for (int i = 1; i < nr + 1; i++) {
        for (int j = 1; j < nc + 1; j++) {
            const int index = (i - 1) * nc + j - 1;
            ASSERT_EQ(m(i, j), v[index]);
        }
    }
}
