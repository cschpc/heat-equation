// SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

/* Setup routines for heat equation solver */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "heat.h"

#define NSTEPS 500 // Default number of iteration steps

/* Initialize the heat equation solver */
void initialize(int argc, char *argv[], field *current, field *previous,
                int *nsteps, parallel_data *parallel) {
    /*
     * Following combinations of command line arguments are possible:
     * No arguments:    use default field dimensions and number of time steps
     * One argument:    read initial field from a given file
     * Two arguments:   initial field from file and number of time steps
     * Three arguments: field dimensions (rows,cols) and number of time steps
     */

    int rows = 2000; //!< Field dimensions with default values
    int cols = 2000;

    char input_file[64]; //!< Name of the optional input file

    int read_file = 0;

    *nsteps = NSTEPS;

    switch (argc) {
    case 1:
        /* Use default values */
        break;
    case 2:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_file = 1;
        break;
    case 3:
        /* Read initial field from a file */
        strncpy(input_file, argv[1], 64);
        read_file = 1;

        /* Number of time steps */
        *nsteps = atoi(argv[2]);
        break;
    case 4:
        /* Field dimensions */
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        /* Number of time steps */
        *nsteps = atoi(argv[3]);
        break;
    default:
        printf("Unsupported number of command line arguments\n");
        exit(-1);
    }

    if (read_file) {
#pragma omp master
        read_field(current, previous, input_file, parallel);
#pragma omp barrier
    } else {
#pragma omp master
        {
            parallel_setup(parallel, rows, cols);
            set_field_dimensions(current, rows, cols, parallel);
            set_field_dimensions(previous, rows, cols, parallel);
        }
#pragma omp barrier
        generate_field(current, parallel);
#pragma omp master
        {
            allocate_field(previous);
            copy_field(current, previous);
        }
#pragma omp barrier
    }
}

/* Generate initial temperature field.  Pattern is disc with a radius
 * of nx_full / 6 in the center of the grid.
 * Boundary conditions are (different) constant temperatures outside the grid */
void generate_field(field *temperature, parallel_data *parallel) {
    double radius;

    /* Allocate the temperature array, note that
     * we have to allocate also the ghost layers */
#pragma omp master
    temperature->data =
        new double[(temperature->nx + 2) * (temperature->ny + 2)];
#pragma omp barrier

    /* Radius of the source disc */
    radius = temperature->nx_full / 6.0;
#pragma omp for
    for (int i = 0; i < temperature->nx + 2; i++) {
        for (int j = 0; j < temperature->ny + 2; j++) {
            int ind = i * (temperature->ny + 2) + j;
            /* Distance of point i, j from the origin */
            int dx = i + parallel->rank * temperature->nx -
                     temperature->nx_full / 2 + 1;
            int dy = j - temperature->ny / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                temperature->data[ind] = 5.0;
            } else {
                temperature->data[ind] = 65.0;
            }
        }
    }

/* Boundary conditions */
#pragma omp for
    for (int i = 0; i < temperature->nx + 2; i++) {
        temperature->data[i * (temperature->ny + 2)] = 20.0;
        temperature->data[i * (temperature->ny + 2) + temperature->ny + 1] =
            70.0;
    }

    if (parallel->rank == 0) {
#pragma omp for
        for (int j = 0; j < temperature->ny + 2; j++) {
            temperature->data[j] = 85.0;
        }
    }
    if (parallel->rank == parallel->size - 1) {
#pragma omp for
        for (int j = 0; j < temperature->ny + 2; j++) {
            temperature
                ->data[(temperature->nx + 1) * (temperature->ny + 2) + j] = 5.0;
        }
    }
}

/* Set dimensions of the field. Note that the nx is the size of the first
 * dimension and ny the second. */
void set_field_dimensions(field *temperature, int nx, int ny,
                          parallel_data *parallel) {
    int nx_local;

    nx_local = nx / parallel->size;

    temperature->dx = DX;
    temperature->dy = DY;
    temperature->nx = nx_local;
    temperature->ny = ny;
    temperature->nx_full = nx;
    temperature->ny_full = ny;
}

void parallel_setup(parallel_data *parallel, int nx, int ny) {
    MPI_Comm_size(MPI_COMM_WORLD, &parallel->size);
    MPI_Comm_rank(MPI_COMM_WORLD, &parallel->rank);

    parallel_set_dimensions(parallel, nx, ny);

    parallel->nup = parallel->rank - 1;
    parallel->ndown = parallel->rank + 1;

    if (parallel->nup < 0) {
        parallel->nup = MPI_PROC_NULL;
    }
    if (parallel->ndown > parallel->size - 1) {
        parallel->ndown = MPI_PROC_NULL;
    }
}

void parallel_set_dimensions(parallel_data *parallel, int nx, int ny) {
    int nx_local;

    nx_local = nx / parallel->size;
    if (nx_local * parallel->size != nx) {
        printf("Cannot divide grid evenly to processors\n");
        MPI_Abort(MPI_COMM_WORLD, -2);
    }
}

/* Deallocate the 2D arrays of temperature fields */
void finalize(field *temperature1, field *temperature2) {
    delete[] temperature1->data;
    delete[] temperature2->data;
}
