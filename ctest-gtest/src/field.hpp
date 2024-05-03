#pragma once

#include <cassert>
#include <vector>

struct Field {
    int num_rows = 0;
    int num_cols = 0;

  private:
    std::vector<double> temperatures;

    // Internal 1D indexing
    const int index(int i, int j) const {
        const int idx = i * (num_cols + 2) + j;
        assert(idx >= 0 && idx < temperatures.size());

        return idx;
    }

  public:
    Field(std::vector<double> &&temperatures, int num_rows, int num_cols);

    // standard (i,j) syntax for setting elements
    // i and j are both offset by one to skip the ghost layers
    double &operator()(int i, int j) {
        return temperatures[index(i + 1, j + 1)];
    }
    const double &operator()(int i, int j) const {
        return temperatures[index(i + 1, j + 1)];
    }

    double sum() const;
    std::vector<double> get_temperatures() const;

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
