#pragma once
#include <string>
#include <cstdio>
#include "matrix.hpp"

// Forward delaration class for basic parallelization information
struct ParallelData; 

// Class for temperature field
struct Field {
    // nx and ny are the true dimensions of the field. The temperature matrix
    // contains also ghost layers, so it will have dimensions nx+2 x ny+2 
    int nx;                     // Local dimensions of the field
    int ny;
    int nz;
    int nx_full;                // Global dimensions of the field
    int ny_full;                // Global dimensions of the field
    int nz_full;                // Global dimensions of the field
    double dx = 0.01;           // Grid spacing
    double dy = 0.01;
    double dz = 0.01;

    Matrix<double> temperature;

    double *devdata;

    void setup(int nx_in, int ny_in, int nz_in, ParallelData& parallel);

    void generate(const ParallelData& parallel);

    // standard (i,j) syntax for setting elements
    double& operator()(int i, int j, int k) {return temperature(i, j, k);}

    // standard (i,j) syntax for getting elements
    const double& operator()(int i, int j, int k) const {return temperature(i, j, k);}

};

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData& parallel);

void exchange(Field& field, ParallelData& parallel);

void evolve(Field& curr, Field& prev, const double a, const double dt);

void enter_data(Field& curr, Field& prev);

void exit_data(Field& curr, Field& prev);

void update_host(Field& field);

void update_device(Field& field);

void write_field(Field& field, const int iter, const ParallelData& parallel);

void read_field(Field& field, std::string filename,
                ParallelData& parallel);

double average(const Field& field);
