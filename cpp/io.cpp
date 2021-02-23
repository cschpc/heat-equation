/* I/O related functions for heat equation solver */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mpi.h>

#include "heat.hpp"
#include "pngwriter.h"

/* Output routine that prints out a picture of the temperature
 * distribution. */
void write_field(Field& field, int const iter, ParallelData const parallel)
{
    char filename[64];

    /* The actual write routine takes only the actual data
     * (without ghost layers) so we need array for that. */
    double *full_data;
    double *tmp_data;          // array for MPI sends and receives

    auto height = field.nx * parallel.size;
    auto width = field.ny;

    auto data = field.temperature.data();

    tmp_data = new double [field.nx * field.ny];

    if (parallel.rank == 0) {
        /* Copy the inner data */
        full_data = new double [height * width];
        for (int i = 0; i < field.nx; i++)
	  memcpy(&full_data[i * width], &data[(i + 1) * (width + 2) + 1],
                   field.ny * sizeof(double));
        /* Receive data from other ranks */
        for (int p = 1; p < parallel.size; p++) {
            MPI_Recv(tmp_data, field.nx * field.ny,
                     MPI_DOUBLE, p, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            /* Copy data to full array */
            memcpy(&full_data[p * field.nx * width], tmp_data,
                   field.nx * field.ny * sizeof(double));
        }
        /* Write out the data to a png file */
        std::sprintf(filename, "%s_%04d.png", "heat", iter);
        save_png(full_data, height, width, filename, 'c');
        delete[] full_data;
    } else {
        /* Send data */
        for (int i = 0; i < field.nx; i++)
	  memcpy(&tmp_data[i * width], &data[(i + 1) * (width + 2) + 1],
                   field.ny * sizeof(double));
        MPI_Send(tmp_data, field.nx * field.ny,
                 MPI_DOUBLE, 0, 22, MPI_COMM_WORLD);
    }

    delete[] tmp_data;

}

/* Read the initial temperature distribution from a file and
 * initialize the temperature fields temperature1 and
 * temperature2 to the same initial state. */
void read_field(Field& field, std::string filename,
                ParallelData const parallel)
{
    FILE *fp;
    int nx, ny, ind;
    double *full_data;
    double *inner_data;

    int nx_local, ny_local, count;

    fp = fopen(filename.c_str(), "r");
    /* Read the header */
    count = fscanf(fp, "# %d %d \n", &nx, &ny);
    if (count < 2) {
        fprintf(stderr, "Error while reading the input file!\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }


    field.setup(nx, ny, parallel);

    inner_data = new double[field.nx * field.ny];

    if (parallel.rank == 0) {
        /* Full array */
        full_data = new double [nx * ny];

        /* Read the actual data */
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
	        ind = i * ny + j;
                count = fscanf(fp, "%lf", &full_data[ind]);
            }
        }
    } else {
        /* Dummy array for full data. Some MPI implementations
         * require that this array is actually allocated... */
        full_data = new double[1];
    }

    nx_local = field.nx;
    ny_local = field.ny;

    MPI_Scatter(full_data, nx_local * ny, MPI_DOUBLE, inner_data,
                nx_local * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Copy to the array containing also boundaries */
    auto data = field.temperature.data();
    for (int i = 0; i < nx_local; i++)
      memcpy(&data[(i + 1) * (ny_local + 2) + 1], &inner_data[i * ny_local],
               ny * sizeof(double));

    /* Set the boundary values */
    for (int i = 1; i < nx_local + 1; i++) {
        data[i * (ny_local + 2)] = data[i * (ny_local + 2) + 1];
        data[i * (ny_local + 2) + ny + 1] = data[i * (ny_local + 2) + ny];
    }
    for (int j = 0; j < ny + 2; j++) {
        data[j] = data[ny_local + j];
        data[(nx_local + 1) * (ny_local + 2) + j] = data[nx_local * (ny_local + 2) + j];
    }

    delete[] full_data;
    delete[] inner_data; 
    fclose(fp);
}
