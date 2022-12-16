! Main solver routines for heat equation solver
module core
  use heat
  use openacc
  use, intrinsic :: iso_c_binding

  interface

    subroutine evolve_hip(curr, prev, a, dt, nx, ny, nz, dx, dy, dz) bind(C, name='evolve')
      USE, intrinsic :: iso_c_binding
#ifdef _CRAYFTN
      type(c_ptr), value :: curr, prev
#else
      type(c_devptr), intent(inout) :: curr, prev
#endif
      real(c_double), value :: a, dt, dx, dy, dz
      integer(c_int), value :: nx, ny, nz
    end subroutine evolve_hip

  end interface

contains

  ! Exchange the boundary data between MPI tasks
  subroutine exchange(field0, parallel)
    use mpi

    implicit none

    type(field), target, intent(inout) :: field0
    type(parallel_data), intent(in) :: parallel

    real(dp), pointer, contiguous, dimension(:,:,:) :: data

    integer :: buf_size

    integer :: ierr

    data => field0%data

    buf_size = (field0%nx + 2) * (field0%ny + 2)

#ifdef GPU_MPI
!$acc host_data use_device(data)
#else
!$acc update host(data)
#endif
    ! Send to left, receive from right
    call mpi_sendrecv(data(:, :, 1), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%nleft, 11, &
         & data(:, :, field0%nz + 1), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%nright, 11, &
         & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

    ! Send to right, receive from left
    call mpi_sendrecv(data(:, :, field0%nz), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%nright, 12, &
         & data(:, :, 0), buf_size, MPI_DOUBLE_PRECISION,&
         & parallel%nleft, 12, &
         & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
#ifdef GPU_MPI
!$acc end host_data
#else
!$acc update device(data)
#endif
  end subroutine exchange

  ! Compute one time step of temperature evolution
  ! Arguments:
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  !   a (real(dp)): update equation constant
  !   dt (real(dp)): time step value
  subroutine evolve(curr, prev, a, dt)

    implicit none

    type(field), target, intent(inout) :: curr, prev
    real(kind=dp), intent(in) :: a, dt

    real(dp), pointer, contiguous, dimension(:,:,:) :: currdata, prevdata
    type(c_ptr) :: curr_p, prev_p
    integer :: i, j, k, nx, ny, nz
    real(kind=dp) :: dx, dy, dz

    nx = curr%nx
    ny = curr%ny
    nz = curr%nz

    dx = curr%dx
    dy = curr%dy
    dz = curr%dz

    currdata => curr%data
    prevdata => prev%data

    ! call evolve_hip(acc_deviceptr(currdata), acc_deviceptr(prevdata), a, dt, nx, ny, nz, dx, dy, dz)
!$acc host_data use_device(currdata, prevdata)
    curr_p = c_loc(currdata)
    prev_p = c_loc(prevdata)
!$acc end host_data
    call evolve_hip(curr_p, prev_p, a, dt, nx, ny, nz, dx, dy, dz)
  end subroutine evolve

  ! Start a data region and copy temperature fields to the device
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  subroutine enter_data(curr, prev)
    implicit none
    type(field), target, intent(in) :: curr, prev
    real(kind=dp), pointer, contiguous :: currdata(:,:,:), prevdata(:,:,:)

    currdata => curr%data
    prevdata => prev%data

    !$acc enter data copyin(currdata, prevdata)
  end subroutine enter_data

  ! End a data region and copy temperature fields back to the host
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  subroutine exit_data(curr, prev)
    implicit none
    type(field), target :: curr, prev
    real(kind=dp), pointer, contiguous :: currdata(:,:,:), prevdata(:,:,:)

    currdata => curr%data
    prevdata => prev%data

    !$acc exit data copyout(currdata, prevdata)
  end subroutine exit_data

  ! Copy a temperature field from the device to the host
  !   temperature (type(field)): temperature field
  subroutine update_host(temperature)
    implicit none
    type(field), target :: temperature
    real(kind=dp), pointer, contiguous :: tempdata(:,:,:)

    tempdata => temperature%data

    !$acc update host(tempdata)
  end subroutine update_host

end module core
