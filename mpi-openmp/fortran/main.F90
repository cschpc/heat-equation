! Heat equation solver in 2D.

program heat_solve
  use heat
  use core
  use io
  use setup
  use utilities
  use mpi_f08
#ifdef _OPENMP
  use omp_lib
#endif

  implicit none

  real(dp), parameter :: a = 0.5 ! Diffusion constant
  type(field) :: current, previous    ! Current and previus temperature fields

  real(dp) :: dt     ! Time step
  integer :: nsteps       ! Number of time steps
  integer, parameter :: image_interval = 1500 ! Image output interval

  type(parallel_data) :: parallelization
  integer :: ierr, provided

  integer :: iter

  real(dp) :: average_temp   !  Average temperature

  real(kind=dp) :: start, stop ! Timers

  call mpi_init_thread(MPI_THREAD_MULTIPLE, provided, ierr)
  if (provided < MPI_THREAD_MULTIPLE) then
     write (*,*) ' MPI_THREAD_MULTIPLE required for the thread support level '
     call mpi_abort(MPI_COMM_WORLD, 5, ierr)
  end if

!$OMP PARALLEL PRIVATE(iter)

  call initialize(current, previous, nsteps, parallelization)

  ! Draw the picture of the initial state
!$OMP SINGLE
  call write_field(current, 0, parallelization)

  average_temp = average(current)
  if (parallelization % rank == 0) then
     write(*,'(A,F9.6)') 'Average temperature at start: ', average_temp
  end if

  ! Largest stable time step
  dt = current%dx**2 * current%dy**2 / &
       & (2.0 * a * (current%dx**2 + current%dy**2))
!$OMP END SINGLE

  ! Main iteration loop, save a picture every
  ! image_interval steps

!$OMP MASTER
  start =  mpi_wtime()
!$OMP END MASTER

  do iter = 1, nsteps
!$OMP SINGLE
     call exchange(previous, parallelization)
!$OMP END SINGLE
     call evolve(current, previous, a, dt)
     if (mod(iter, image_interval) == 0) then
!$OMP SINGLE
        call write_field(current, iter, parallelization)
!$OMP END SINGLE
     end if
!$OMP SINGLE
     call swap_fields(current, previous)
!$OMP END SINGLE
  end do

!$OMP MASTER
  stop = mpi_wtime()
!$OMP END MASTER
!$OMP END PARALLEL

  ! Average temperature for reference
  average_temp = average(previous)

  if (parallelization % rank == 0) then
     write(*,'(A,F7.3,A)') 'Iteration took ', stop - start, ' seconds.'
     write(*,'(A,F9.6)') 'Average temperature: ',  average_temp
     if (command_argument_count() == 0) then
         write(*,'(A,F9.6)') 'Reference value with default arguments: ', 59.281239
     end if
  end if

  call finalize(current, previous)

  call mpi_finalize(ierr)

end program heat_solve
