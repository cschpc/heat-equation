/*
 * SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __HEAT_H__
#define __HEAT_H__


/* Datatype for temperature field */
typedef struct {
    /* nx and ny are the true dimensions of the field. The array data
     * contains also ghost layers, so it will have dimensions nx+2 x ny+2 */
    int nx;                     /* Local dimensions of the field */
    int ny;
    int nx_full;                /* Global dimensions of the field */
    int ny_full;                /* Global dimensions of the field */
    double dx;
    double dy;
    double *data;
} field;

/* We use here fixed grid spacing */
#define DX 0.01
#define DY 0.01

#if __cplusplus
  extern "C" {
#endif
/* Function prototypes */
void set_field_dimensions(field *temperature, int nx, int ny);

void initialize(int argc, char *argv[], field *temperature1,
                field *temperature2, int *nsteps);

void generate_field(field *temperature);

double average(field *temperature);

void evolve(field *curr, field *prev, double a, double dt);

void write_field(field *temperature, int iter);

void read_field(field *temperature1, field *temperature2,
                char *filename);

void copy_field(field *temperature1, field *temperature2);

void swap_fields(field *temperature1, field *temperature2);

void allocate_field(field *temperature);

void finalize(field *temperature1, field *temperature2);

#if __cplusplus
  }
#endif
#endif  /* __HEAT_H__ */

