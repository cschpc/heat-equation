#pragma once

namespace heat {
struct Constants;
}

struct Field;
struct ParallelData;

void exchange(Field &field, const ParallelData &parallel);
void evolve(Field &curr, const Field &prev, const heat::Constants &constants);
