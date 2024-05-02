#pragma once

struct Field;
struct ParallelData;

void exchange(Field &field, const ParallelData &parallel);
void evolve(Field &curr, const Field &prev, double diffusion_constant,
            double dt);
