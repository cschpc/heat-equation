#pragma once
#include "heat.hpp"
#include "parallel.hpp"

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData& parallel);

void exchange(Field& field, ParallelData& parallel);

void evolve(Field& curr, const Field& prev, const double a, const double dt);

void write_field(Field& field, const int iter, const ParallelData& parallel);

void read_field(Field& field, std::string filename,
                ParallelData& parallel);

double average(const Field& field);

double timer();
