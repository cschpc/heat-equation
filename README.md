<!--
SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Heat equation solver

This repository contains various implementations of simple heat equation with 
various parallel programming approaches.

The two dimensional versions under [2d](2d) are more suitable as basis training material, the three dimensional versions under [3d](3d) can be used also for simple performance
testing of different programming approaches.

## Description of two dimensional problem

<!-- Equation
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
--> 
![img](images/heat_equation.png)

where **u(x, y, t)** is the temperature field that varies in space and time,
and α is thermal diffusivity constant. The two dimensional Laplacian can be
discretized with finite differences as

<!-- Equation
\begin{align*}
\nabla^2 u  &= \frac{u(i-1,j)-2u(i,j)+u(i+1,j)}{(\Delta x)^2} \\
 &+ \frac{u(i,j-1)-2u(i,j)+u(i,j+1)}{(\Delta y)^2}
 \end{align*}
 --> 
 ![img](images/laplacian.png)

 Given an initial condition (u(t=0) = u0) one can follow the time dependence
 of
 the temperature field with explicit time evolution method:

 <!-- Equation
 u^{m+1}(i,j) = u^m(i,j) + \Delta t \alpha \nabla^2 u^m(i,j) 
 --> 
 ![img](images/time_dependence.png)

 Note: Algorithm is stable only when

 <!-- Equation
 \Delta t < \frac{1}{2 \alpha} \frac{(\Delta x \Delta y)^2}{(\Delta x)^2
 (\Delta y)^2}
 -->
 ![img](images/stability.png)

## How to build

For building and running the example one needs to have the
[libpng](http://www.libpng.org/pub/png/libpng.html) library installed. Working MPI 
environment is required for all cases except for the serial one. In addition:

 * Hybrid MPI-OpenMP version requires MPI implementation with
   MPI_THREAD_MULTIPLE support
 * CUDA version requires CUDA environment and CUDA aware MPI

 Move to proper subfolder and modify the top of the **Makefile**
 according to your environment (proper compiler commands and compiler flags).
 Code can be built simply with **make**

## How to run

The number of MPI ranks has to be a factor of the grid dimension (default 
dimension is 2000). For GPU versions, number of MPI tasks per node has to be the
same as number of GPUs per node. 

The default initial temperature field is a disk. Initial
temperature field can be read also from a file, the provided [bottle.dat](common/bottle.dat)
illustrates what happens to a cold soda bottle in sauna.


 * Running with defaults: mpirun -np 4 ./heat_mpi
 * Initial field from a file: mpirun -np 4 ./heat_mpi bottle.dat
 * Initial field from a file, given number of time steps:
   mpirun -np 4 ./heat_mpi bottle.dat 1000
 * Defauls pattern with given dimensions and time steps:
   mpirun -np 4 ./heat_mpi 800 800 1000

  The program produces a series of heat_XXXX.png files which show the
  time development of the temperature field

