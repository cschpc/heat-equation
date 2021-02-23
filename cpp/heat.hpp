#pragma once
#include <vector>
#include <string>

// We use here fixed grid spacing
#define DX 0.01
#define DY 0.01

// Class for basic parallelization information
struct ParallelData {
    int size;            // Number of MPI tasks
    int rank;
    int nup, ndown;      // Ranks of neighbouring MPI tasks

    ParallelData();      // Constructor
};

// Class for temperature field
struct Field {
    /* nx and ny are the true dimensions of the field. The array data
     * contains also ghost layers, so it will have dimensions nx+2 x ny+2 */
    int nx;                     // Local dimensions of the field
    int ny;
    int nx_full;                // Global dimensions of the field
    int ny_full;                // Global dimensions of the field
    double dx = DX;
    double dy = DY;
    std::vector<double> temperature;

    void setup(int nx_in, int ny_in, ParallelData parallel);

    void swap(Field& other);

    void generate(ParallelData parallel);

};

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData parallel);

void exchange(Field& field, ParallelData const parallel);

void evolve(Field& curr, Field& prev, double const a, double const dt);

void write_field(Field& field, int const iter, ParallelData const parallel);

void read_field(Field& field, std::string filename,
                ParallelData const parallel);

double average(Field& field);
