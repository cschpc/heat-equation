#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>

#include "io.hpp"
#include "pngwriter.h"

namespace heat {
void run(int, char **);
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
    char *argv[] = {};
    heat::run(1, argv);

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

TEST(integration_test, default_input_ok) {
    const heat::Input default_input = {};
    const heat::Input input = heat::read_input("", 0);
    ASSERT_EQ(input, default_input) << "input is different from default_input";
}

TEST(integration_test, input_from_file_ok) {
    const heat::Input input = heat::read_input("testdata/input.json", 0);
    const heat::Input default_input = {};
    ASSERT_NE(input, default_input) << "input is equal to default_input";
}

TEST(integration_test, input_from_nullptr_throws_exception) {
    EXPECT_THROW(
        {
            try {
                const heat::Input input = heat::read_input(nullptr, 0);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("Filename is a nullptr", e.what());
                throw;
            }
        },
        std::runtime_error);
}

TEST(integration_test, input_from_nonexistent_path_throws_exception) {
    EXPECT_THROW(
        {
            try {
                const heat::Input input =
                    heat::read_input("batman vs superman", 0);
            } catch (const std::runtime_error &e) {
                EXPECT_STREQ("Non-existent path: \"batman vs superman\"",
                             e.what());
                throw;
            }
        },
        std::runtime_error);
}

TEST(integration_test, read_field_data_from_file_rank_0) {
    constexpr int rank = 0;
    auto [num_rows, num_cols, data] =
        heat::read_field("testdata/bottle.dat", rank);
    ASSERT_EQ(data.size(), 40000);
    ASSERT_EQ(num_rows * num_cols, data.size());
    ASSERT_GT(data.size(), 0) << "data vector should contain some data";
}

TEST(integration_test, read_field_data_from_file_rank_1) {
    constexpr int rank = 1;
    auto [num_rows, num_cols, data] =
        heat::read_field("testdata/bottle.dat", rank);
    ASSERT_EQ(num_rows, 200);
    ASSERT_EQ(num_cols, 200);
    ASSERT_TRUE(data.empty()) << "data vector should be empty";
}
