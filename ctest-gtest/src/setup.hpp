#pragma once

namespace heat {
struct Input;
}

struct Field;
struct ParallelData;

Field initialize(const heat::Input &input, const ParallelData &parallel);
