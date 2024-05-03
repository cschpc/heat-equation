#pragma once

namespace heat {
struct Input;
struct Field;
struct ParallelData;

Field initialize(const Input &input, const ParallelData &parallel);
} // namespace heat
