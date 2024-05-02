#include "matrix.hpp"
#include <gtest/gtest.h>
#include <numeric>

TEST(matrix_test, construct_with_vector) {
    constexpr int nr = 20;
    constexpr int nc = 100;
    std::vector<double> v(nr * nc);
    std::iota(v.begin(), v.end(), 0);
    const Matrix<double> m(std::move(v), nr, nc);
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            const double value = static_cast<double>(i * nc + j);
            ASSERT_EQ(m(i, j), value);
        }
    }
}

TEST(matrix_test, add_ghost_layers) {
    constexpr int nr = 20;
    constexpr int nc = 100;
    std::vector<double> v(nr * nc);
    std::iota(v.begin(), v.end(), 0);
    Matrix<double> m = Matrix<double>::make_with_ghost_layers(
        Matrix<double>(std::move(v), nr, nc));

    ASSERT_EQ(m.num_rows, nr + 2);
    ASSERT_EQ(m.num_cols, nc + 2);

    for (int i = 1; i < nr + 1; i++) {
        for (int j = 1; j < nc + 1; j++) {
            const double value = static_cast<double>((i - 1) * nc + j - 1);
            ASSERT_EQ(m(i, j), value);
        }
    }
}

TEST(matrix_test, add_ghost_layers_from_raw) {
    constexpr int nr = 20;
    constexpr int nc = 100;
    std::vector<double> v(nr * nc);
    std::iota(v.begin(), v.end(), 0);
    Matrix<double> m =
        Matrix<double>::make_with_ghost_layers(std::move(v), nr, nc);

    ASSERT_EQ(m.num_rows, nr + 2);
    ASSERT_EQ(m.num_cols, nc + 2);

    for (int i = 1; i < nr + 1; i++) {
        for (int j = 1; j < nc + 1; j++) {
            const double value = static_cast<double>((i - 1) * nc + j - 1);
            ASSERT_EQ(m(i, j), value);
        }
    }
}

TEST(matrix_test, data_access_correct) {
    constexpr int nr = 32;
    constexpr int nc = 16;
    std::vector<double> v(nr * nc);
    std::iota(v.begin(), v.end(), 0);
    const Matrix<double> m(std::move(v), nr, nc);
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            ASSERT_EQ(*m.data(i, j), m(i, j));
        }
    }
}
