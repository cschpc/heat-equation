! Main solver routines for heat equation solver
module core
  use heat

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
!$acc update host(data(:,:,1), data(:,:,field0%nz))
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
!$acc update device(data(:,:,0), data(:,:,field0%nz+1))
#endif
  end subroutine exchange

  subroutine exchange_init(field0, parallel)
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
!$acc update host(data(:,:,1), data(:,:,field0%nz))
#endif
    ! Start to receive
    call mpi_irecv(data(:, :, field0%nz + 1), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%nright, 11, MPI_COMM_WORLD, parallel%requests(1), ierr)
    call mpi_irecv(data(:, :, 0), buf_size, MPI_DOUBLE_PRECISION,&
         & parallel%nleft, 12, MPI_COMM_WORLD, parallel%requests(2), ierr)

    ! Start to send
    call mpi_isend(data(:, :, 1), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%nleft, 11, MPI_COMM_WORLD, parallel%requests(3), ierr)
    call mpi_isend(data(:, :, field0%nz), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%nright, 12, MPI_COMM_WORLD, parallel%requests(4), ierr)

#ifdef GPU_MPI
!$acc end host_data
#endif

  end subroutine exchange_init

  subroutine exchange_finalize(field0, parallel)
    use mpi

    implicit none

    type(field), target, intent(inout) :: field0
    type(parallel_data), intent(in) :: parallel

    real(dp), pointer, contiguous, dimension(:,:,:) :: data

    integer :: buf_size

    integer :: ierr

    data => field0%data

    call mpi_waitall(4, parallel%requests, mpi_statuses_ignore, ierr)

#ifndef GPU_MPI
!$acc update device(data(:,:,0), data(:,:,field0%nz+1))
#endif

  end subroutine exchange_finalize

  ! Compute one time step of temperature evolution
  ! Arguments:
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  !   a (real(dp)): update equation constant
  !   dt (real(dp)): time step value
  subroutine evolve(curr, prev, a, dt)

    implicit none

    type(field), target, intent(inout) :: curr, prev
    real(dp) :: a, dt
    integer :: i, j, k, nx, ny, nz
    real(dp) :: inv_dx2, inv_dy2, inv_dz2
    ! variables for memory access outside of a type
    real(dp), pointer, contiguous, dimension(:,:,:) :: currdata, prevdata

    ! HINT: to help the compiler do not access type components
    !       within OpenACC parallel regions
    nx = curr%nx
    ny = curr%ny
    nz = curr%nz
    inv_dx2 = 1.0 / curr%dx**2
    inv_dy2 = 1.0 / curr%dy**2
    inv_dz2 = 1.0 / curr%dz**2
    currdata => curr%data
    prevdata => prev%data

    !$acc parallel loop private(i,j,k) &
    !$acc present(prevdata(0:nx+1,0:ny+1,0:nz+1), currdata(0:nx+1,0:ny+1,0:nz+1)) collapse(3)
    do k = 1, nz
       do j = 1, ny
          do i = 1, nx
             currdata(i, j, k) = prevdata(i, j, k) + a * dt * &
                  & ((prevdata(i-1, j, k) - 2.0 * prevdata(i, j, k) + &
                  &   prevdata(i+1, j, k)) * inv_dx2 + &
                  &  (prevdata(i, j-1, k) - 2.0 * prevdata(i, j, k) + &
                  &   prevdata(i, j+1, k)) * inv_dy2 + &
                  &  (prevdata(i, j, k-1) - 2.0 * prevdata(i, j, k) + &
                  &   prevdata(i, j, k+1)) * inv_dz2)

          end do
       end do
    end do
    !$end parallel loop

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
