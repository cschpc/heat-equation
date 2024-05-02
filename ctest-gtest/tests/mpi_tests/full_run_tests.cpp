#include <cstdint>
#include <filesystem>

#include "field.hpp"
#include "mpi_test_environment.hpp"
#include "parallel.hpp"
#include "pngwriter.h"
#include "utilities.hpp"

namespace heat {
void run(std::string &&);
}

struct PngData {
    int nx = 0;
    int ny = 0;
    int channels = 0;
    uint8_t *data = nullptr;

    ~PngData() {
        if (data != nullptr) {
            release_png(data);
        }
    }
};

bool loadPng(const char *fname, PngData &png) {
    auto path = (std::filesystem::current_path() / fname);

    if (std::filesystem::exists(path)) {
        path = std::filesystem::is_symlink(path)
                   ? std::filesystem::read_symlink(path)
                   : path;

        const auto filename = path.c_str();
        png.data = load_png(filename, &png.nx, &png.ny, &png.channels);

        return true;
    }

    return false;
}

TEST(integration_test, image_matches_reference) {
    heat::run("testdata/input.json");

    PngData reference_data = {};
    ASSERT_TRUE(loadPng("testdata/heat_0500.png", reference_data))
        << "Could not load reference data from png";

    PngData computed_data = {};
    ASSERT_TRUE(loadPng("heat_0500.png", computed_data))
        << "Could not load computed data from png";

    ASSERT_EQ(reference_data.nx, computed_data.nx);
    ASSERT_EQ(reference_data.ny, computed_data.ny);
    ASSERT_EQ(reference_data.channels, computed_data.channels);

    const int num_bytes =
        reference_data.nx * reference_data.ny * reference_data.channels;
    for (int i = 0; i < num_bytes; i++) {
        ASSERT_EQ(reference_data.data[i], computed_data.data[i])
            << "Computed data differs from reference data at byte " << i
            << "\nReference: " << reference_data.data[i]
            << ", computed: " << computed_data.data[i];
    }
}

TEST(utilities_test, zero_field_average_is_zero) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 2000;
    const Field field(std::vector<double>(num_rows * num_cols), num_rows,
                      num_cols);
    ParallelData pd;
    ASSERT_EQ(heat::average(field, pd), 0.0);
}

TEST(utilities_test, unity_field_average_is_one) {
    constexpr int num_rows = 2000;
    constexpr int num_cols = 100;
    const Field field(std::vector<double>(num_rows * num_cols, 1.0), num_rows,
                      num_cols);
    ParallelData pd;
    ASSERT_EQ(heat::average(field, pd), 1.0);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
    return RUN_ALL_TESTS();
}
