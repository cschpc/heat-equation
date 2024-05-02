#include <mpi.h>
#include <gtest/gtest.h>

class MPIEnvironment : public ::testing::Environment
{
public:
  virtual void SetUp() {
    char** argv;
    int argc = 0;
    int mpiError = MPI_Init(&argc, &argv);
    ASSERT_FALSE(mpiError);
  }
  virtual void TearDown() {
    int mpiError = MPI_Finalize();
    ASSERT_FALSE(mpiError);
  }
  virtual ~MPIEnvironment() {}
};
