#pragma once

#include <cassert>
#include <vector>

namespace heat {
struct Constants;

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
    // The pointer to values that are sent up, i.e. to the first non-ghost value
    const double *to_up() const { return temperatures.data() + index(1, 1); }
    // The pointer to values that are sent down, i.e. the first non-ghost value
    // on the last non-ghost row
    const double *to_down() const {
        return temperatures.data() + index(num_rows, 1);
    }
    // The pointer to values that are received from up, i.e. top ghost layer
    double *from_up() { return temperatures.data() + index(0, 1); }
    // The pointer to values that are received from down, i.e. bottom ghost
    // layer
    double *from_down() { return temperatures.data() + index(num_rows + 1, 1); }
    int num_to_exchange() const { return num_cols; }
    double sample(int i, int j, const Constants &constants) const;

    static std::pair<int, int> partition_domain(int num_rows, int num_cols,
                                                int num_partitions);
};
} // namespace heat
