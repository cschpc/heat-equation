! I/O routines for heat equation solver
module io
  use heat
  use mpi
  use iso_c_binding
  use hipfort
  use hipfort_check

contains

  ! Output routine, saves the temperature distribution as a png image
  ! Arguments:
  !   curr (type(field)): variable with the temperature data
  !   iter (integer): index of the time step
  ! not yet supported, left and commented out the old code. 
  ! only required change SHOULD be to add the memcpy d2h.
  subroutine write_field(curr, iter, parallel)

!#ifndef DISABLE_PNG
!    use pngwriter
!#endif
!    implicit none
!    type(field), intent(in) :: curr
!    integer, intent(in) :: iter
!    type(parallel_data), intent(in) :: parallel
!
!    character(len=85) :: filename
!
!    integer :: stat
!    real(dp), dimension(:,:,:), allocatable, target :: full_data
!    integer :: p, ierr
!
!    if (parallel%rank == 0) then
!       allocate(full_data(curr%nx_full, curr%ny_full, curr%nz_full))
!       ! Copy rand #0 data to the global array
!       full_data(1:curr%nx, 1:curr%ny, 1:curr%nz) = curr%data(1:curr&
!            &%nx, 1:curr%ny, 1:curr%nz)
!
!       ! Receive data from other ranks
!       do p = 1, parallel%size - 1
!          call mpi_recv(full_data(1:curr%nx, 1:curr%ny, p*curr%nz + 1:(p + 1) * curr%ny), &
!               & curr%nx * curr%ny * curr%nz, MPI_DOUBLE_PRECISION, p, 22, &
!               & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
!       end do
!       write(filename,'(A5,I4.4,A4,A)')  'heat_', iter, '.png'
!#ifdef DISABLE_PNG
!       if (parallel%rank == 0) then
!           write(*,*) "No libpng, file not written"
!       end if
!#else
!       stat = save_png(full_data(:, :, curr%nz_full / 2), curr%nx_full, curr%ny_full, filename)
!#endif
!       deallocate(full_data)
!    else
!       ! Send data
!       call mpi_send(curr%data(1:curr%nx, 1:curr%ny, 1:curr%nz), curr%nx * curr%ny * curr%nz, MPI_DOUBLE_PRECISION, 0, 22, &
!            & MPI_COMM_WORLD, ierr)
!    end if

  end subroutine write_field


  ! Reads the temperature distribution from an input file
  ! Arguments:
  !   field0 (type(field)): field variable that will store the
  !                         read data
  !   filename (char): name of the input file
  ! Note that this version assumes the input data to be in C memory layout
  ! untested routine, taken from openacc/fortran. I only inserted the hipfort calls
  subroutine read_field(field0, filename, parallel)

    implicit none
    type(field), intent(out) :: field0
    character(len=85), intent(in) :: filename
    type(parallel_data), intent(out) :: parallel

    integer :: nx, ny, nz, i, ierr
    character(len=2) :: dummy

    real(dp), dimension(:,:,:), allocatable :: full_data, inner_data

    open(10, file=filename)
    ! Read the header
    read(10, *) dummy, nx, ny, nz

    call parallel_setup(parallel, nx, ny, nz)
    call set_field_dimensions(field0, nx, ny, nz, parallel)

    ! The arrays for temperature field contain also a halo region
    allocate(field0%data(0:field0%inner_field%nx+1, 0:field0%inner_field%ny+1, 0:field0%inner_field%nz+1))
    
    call hipCheck(hipMalloc(field0%inner_field%data, size(field0%data)* 8_c_size_t)) 

    allocate(inner_data(field0%inner_field%nx, field0%inner_field%ny, field0%inner_field%nz))

    if (parallel%rank == 0) then
       allocate(full_data(nx, ny, nz))
       ! Read the data
       do i = 1, nx
          read(10, *) full_data(i, 1:ny, 1)
       end do
    end if

    call mpi_scatter(full_data, nx * field0%inner_field%ny, MPI_DOUBLE_PRECISION, inner_data, &
         & nx * field0%inner_field%ny, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
    ! Copy to full array containing also boundaries
    field0%data(1:field0%inner_field%nx, 1:field0%inner_field%ny, 1:field0%inner_field%nz) = inner_data(:,:,:)
    !copy data array to gpu array
    call hipCheck(hipMemcpy(field0%inner_field%data, c_loc(field0%data), size(field0%data)* 8_c_size_t, hipMemcpyHostToDevice))


    ! Set the boundary values
    ! field0%data(1:field0%nx, 0) = field0%data(1:field0%nx, 1)
    ! field0%data(1:field0%nx, field0%ny + 1) = field0%data(1:field0%nx, field0%ny)
    ! field0%data(0, 0:field0%ny + 1) = field0%data(1, 0:field0%ny + 1)
    ! field0%data(field0%nx + 1, 0:field0%ny + 1) = field0%data(field0%nx, 0:field0%ny + 1)

    close(10)
    deallocate(inner_data)
    if (parallel%rank == 0) then
       deallocate(full_data)
    end if

  end subroutine read_field

end module io
