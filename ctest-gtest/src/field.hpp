#pragma once

#include <cassert>
#include <vector>

struct Field {
    const int num_rows = 0;
    const int num_cols = 0;

  private:
    // This contains (num_rows + 2) * (num_cols + 2) values, so ghost layers as
    // well
    std::vector<double> temperatures;

    // Internal 1D indexing
    const int index(int i, int j) const {
        // The width of each row is num_cols + 2 with ghost layers
        const int idx = i * (num_cols + 2) + j;
        assert(idx >= 0 && idx < temperatures.size());

        return idx;
    }

  public:
    Field(std::vector<double> &&temperatures, int num_rows, int num_cols);

    // standard (i,j) syntax for setting/getting elements
    // i and j are both offset by one to skip the ghost layers
    // This means the ghost layers are invisible to the user, if they index with
    // i in range [0, num_rows] and with j in range [0, num_cols]
    double &operator()(int i, int j) {
        return temperatures[index(i + 1, j + 1)];
    }
    const double &operator()(int i, int j) const {
        return temperatures[index(i + 1, j + 1)];
    }

    double sum() const;
    std::vector<double> get_temperatures() const;
    void swap(Field &f) { std::swap(temperatures, f.temperatures); }

    // TODO: maybe make a opaque function that gives the correct send/receive
    // row. This way indexing details are internal to field
    // N.B. this differs from operator(i, j) These are not offset by one!
    double *data(int i = 0, int j = 0) {
        return temperatures.data() + index(i, j);
    }
    const double *data(int i = 0, int j = 0) const {
        return temperatures.data() + index(i, j);
    }

    static std::pair<int, int> partition_domain(int num_rows, int num_cols,
                                                int num_partitions);
};
