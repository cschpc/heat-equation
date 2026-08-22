! Main solver routines for heat equation solver
module core
  use heat
  use iso_c_binding
  use hipfort
  use hipfort_check

  !binding the evolve function to the c wrapper that includes the cpp/hip kernel.
  !note that I'm working with the field_c struct, that is a struct declared in both c and fortran
  !that contains the details of the field plus a c_ptr that is pointing to the DEVICE memory.
  !kernel details:
  ! Compute one time step of temperature evolution
  ! Arguments:
  !   curr (type(field_c)): current temperature values
  !   prev (type(field_c)): values from previous time step
  !   a (real(c_double)): update equation constant
  !   dt (real(c_double)): time step value
  
  interface
    subroutine evolve(curr, prev, a, dt) , bind(c,name="evolve")
      use iso_c_binding
      import field_c
      implicit none

      type(field_c), target, intent(inout) :: curr, prev
      real(c_double),value :: a, dt

    end subroutine evolve
  end interface


contains

  ! Exchange the boundary data between MPI tasks
  subroutine exchange(field0, parallel)
    use mpi

    implicit none

    type(field), target, intent(inout) :: field0
    type(parallel_data), intent(in) :: parallel

    !!could be a cray issue. have to check with other compilers
    real(c_double), dimension(:,:,:),allocatable, target :: data !!!!!! THIS MUST BE ALLOCATABLE NOT POINTER FOR MULTIDIMENSIONAL THINGS !!!!!!!

    integer :: buf_size

    integer :: ierr

    call c_f_pointer (field0%inner_field%data , data, shape(field0%data))
    !write(*,*) '#@#@#@#@ data pointer shape is ', shape(data), 'old ptr shape is ', shape(field0%data)
    buf_size = (field0%inner_field%nx + 2) * (field0%inner_field%ny + 2)

    !#ifdef GPU_MPI
    !for now let's suppose that it's always device2device
    !TODO manage data transfer via host
        
    ! Send to left, receive from right
    call mpi_sendrecv(data(:, :, 1), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%nleft, 11, &
         & data(:, :, field0%inner_field%nz+2), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%nright, 11, &
         & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    
    ! Send to right, receive from left
    call mpi_sendrecv(data(:, :, field0%inner_field%nz+1), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%nright, 12, &
         & data(:, :, 1), buf_size, MPI_DOUBLE_PRECISION,&
         & parallel%nleft, 12, &
         & MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    !write(15,*) '########mpi stuff here, data pointer is at ', c_loc(data(1,1,1)), 'doublecheck ', c_loc(data), 'triplecheck', field0%inner_field%data
    !write(15,*) '########mpi stuff here, second element is at ', c_loc(data(1,1,2)), 'or ', c_loc(data(2,1,1))
    !write(15,*) 'size of data is', size(data), 'address of last thing is ', c_loc(data(6,6,4)), 'and it contains ',data(6,6,4)
    !write(15,*) 'first element over data is ',c_loc(data(7,6,4)), 'that contains ',data(7,6,4), ' or ', c_loc(data(6,6,5)),' that contains ', data(6,6,5) !want to be sure they are out of bounds.
    !write(15,*) 'first sendrecv data (left send, right recv), send ',c_loc(data(:,:,2)),' recv ', c_loc(data(:, :, field0%inner_field%nz + 2))
    !write(15,*) 'second sendrecv data, send ',c_loc(data(:, :, field0%inner_field%nz+1)),' recv ', c_loc(data(:, :, 1))
    !write(15,*) 'field host data is at ', c_loc(field0%data), ' and device data is at ', field0%inner_field%data, ' data pointer is at ', c_loc(data)
    !write(15,*) 'buf size is ', buf_size
    !write(15,*) 'shape of data is ', shape(data)
    !call flush (15)
    !write(15,*) 'buf_size', buf_size
    !call flush (15)
    !write(15,*) 'precision', MPI_DOUBLE_PRECISION
    !call flush (15)
    !write(15,*) 'right proc' ,parallel%nright
    !call flush (15)
    !write(15,*) 'pointer' ,data(:, :, 1)
    !call flush (15)
    !write(15,*) 'left', parallel%nleft
    !call flush (15)

    !write(15,*) 'second done'
    

  end subroutine exchange

end module core
