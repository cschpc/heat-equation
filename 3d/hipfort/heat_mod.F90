! Field metadata for heat equation solver
module heat
  use iso_fortran_env, only : REAL64
  use iso_c_binding
  use hipfort
  use hipfort_check
  
  implicit none

  integer, parameter :: dp = REAL64
  real(dp), parameter :: DX = 0.01, DY = 0.01, DZ = 0.01  ! Fixed grid spacing

  !data types structures.
  !i want to have a struct that contains all the details that can be used from the GPU side (i.e. into C side of the program, device memory)
  !and a wrapper of that struct that will contain the fortran declaration of the memory (i.e. the host memory)
  !the device memory must be declared as a c_ptr, and will be allocated and managed manually with hipfort function calls
  !the host memory has to be a pointer (allocatable,target was not valid. I am unsure of the reason and don't remember the actual compiler error)
  type, bind(c) :: field_c
     integer (c_int) :: nx          ! local dimension of the field
     integer (c_int) :: ny
     integer (c_int) :: nz
     integer (c_int) :: nx_full     ! global dimension of the field
     integer (c_int) :: ny_full
     integer (c_int) :: nz_full
     real(c_double) :: dx
     real(c_double) :: dy
     real(c_double) :: dz
     type(c_ptr) :: data = c_null_ptr
  end type field_c

  type :: field
      type(field_c) :: inner_field
      real(c_double), dimension(:,:,:), pointer :: data
  end type field

  type :: parallel_data
     integer :: size
     integer :: rank
     integer :: nleft, nright  ! Ranks of neighbouring MPI tasks
     integer (c_int) :: dev_count !c_int because it needs to go with the hipfort functions
  end type parallel_data

contains
  ! Initialize the field type metadata
  ! Arguments:
  !   field0 (type(field)): input field
  !   nx, ny, dx, dy: field dimensions and spatial step size
  subroutine set_field_dimensions(field0, nx, ny, nz, parallel)
    implicit none

    type(field), intent(out) :: field0
    integer, intent(in) :: nx, ny, nz
    type(parallel_data), intent(in) :: parallel

    integer :: nx_local, ny_local, nz_local

    nx_local = nx
    ny_local = ny
    nz_local = nz / parallel%size

    field0%inner_field%dx = DX
    field0%inner_field%dy = DY
    field0%inner_field%dz = DZ
    field0%inner_field%nx = nx_local
    field0%inner_field%ny = ny_local
    field0%inner_field%nz = nz_local
    field0%inner_field%nx_full = nx
    field0%inner_field%ny_full = ny
    field0%inner_field%nz_full = nz

  end subroutine set_field_dimensions

  subroutine parallel_setup(parallel, nx, ny, nz)
    use mpi


    implicit none

    type(parallel_data), intent(out) :: parallel
    integer, intent(in), optional :: nx, ny, nz

    integer :: nz_local
    integer :: ierr


    integer :: node_rank, node_procs, my_device
    integer :: intranodecomm


    call mpi_comm_size(MPI_COMM_WORLD, parallel%size, ierr)

    if (present(nz)) then
       nz_local = nz / parallel%size
       if (nz_local * parallel%size /= nz) then
          write(*,*) 'Cannot divide grid evenly to processors'
          call mpi_abort(MPI_COMM_WORLD, -2, ierr)
       end if
    end if

    call mpi_comm_rank(MPI_COMM_WORLD, parallel%rank, ierr)
      

    parallel%nleft = parallel%rank - 1
    parallel%nright = parallel%rank + 1

    if (parallel%nleft < 0) then
       parallel%nleft = MPI_PROC_NULL
    end if
    if (parallel%nright > parallel%size - 1) then
       parallel%nright = MPI_PROC_NULL
    end if

    parallel%dev_count = 0

    call mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,  MPI_INFO_NULL, intranodecomm, ierr);

    call mpi_comm_rank(intranodecomm, node_rank, ierr);
    call mpi_comm_size(intranodecomm, node_procs, ierr);

    call mpi_comm_free(intranodecomm, ierr)

    call hipCheck(hipGetDeviceCount(parallel%dev_count))
    my_device = mod(node_rank, parallel%dev_count)

    if (node_procs > parallel%dev_count) then
       if (parallel%rank == 0) then
          write(*, '(AI4AI4)') 'Oversubscriging GPUs: MPI tasks per node: ', &
                &               node_procs, ' GPUs per node: ', parallel%dev_count
       end if
    end if

    call hipCheck(hipSetDevice(my_device))



  end subroutine parallel_setup

end module heat
