! SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
!
! SPDX-License-Identifier: MIT

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

    !$omp parallel do private(i,j,k) collapse(2)
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
    !$omp end parallel do

  end subroutine evolve

end module core
