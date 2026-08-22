! Utility routines for heat equation solver
!   NOTE: This file does not need to be edited!
module utilities
  use heat
  use iso_c_binding
  use hipfort
  use hipfort_check

contains

  ! Swap the data fields of two variables of type field
  ! Arguments:
  !   curr, prev (type(field)): the two variables that are swapped
  subroutine swap_fields(curr, prev)

    implicit none

    type(field), intent(inout) :: curr, prev
    real(dp), pointer, dimension(:,:,:) :: tmp
    type (c_ptr) :: tmp_cptr

    ! i'm now using pointers, so no more move alloc.
    tmp => curr%data       !call move_alloc(curr%data, tmp)
    curr%data => prev%data !call move_alloc(prev%data, curr%data)
    prev%data => tmp       !call move_alloc(tmp, prev%data)
    ! also need to swap the c_ptr with the device memory
    tmp_cptr = curr%inner_field%data
    curr%inner_field%data = prev%inner_field%data
    prev%inner_field%data = tmp_cptr 
    !TODO wouldn't be better to have the structs as pointers and just swap the struct?
    

  end subroutine swap_fields

  ! Copy the data from one field to another
  ! Arguments:
  !   from_field (type(field)): variable to copy from
  !   to_field (type(field)): variable to copy to
  !FIXME UPDATE FOR NEW DATA STRUCT
  subroutine copy_fields(from_field, to_field)

    implicit none

    type(field), intent(in) :: from_field
    type(field), intent(out) :: to_field

    ! Consistency checks
    if (.not.associated(from_field%data) .or. from_field%inner_field%data == c_null_ptr) then
       write (*,*) "Can not copy from a field without allocated data"
       stop
    end if
    if (.not.associated(to_field%data)) then
       ! Target is not initialize, allocate memory
       allocate(to_field%data(lbound(from_field%data, 1):ubound(from_field%data, 1), &
            & lbound(from_field%data, 2):ubound(from_field%data, 2), &
            & lbound(from_field%data, 3):ubound(from_field%data, 3)))
    else if (any(shape(from_field%data) /= shape(to_field%data))) then
       write (*,*) "Wrong field data sizes in copy routine"
       print *, shape(from_field%data), shape(to_field%data)
       stop
    end if

    if (to_field%inner_field%data == c_null_ptr) then
      call hipCheck(hipMalloc(to_field%inner_field%data ,size(from_field%data)* 8_c_size_t)) !https://github.com/ROCmSoftwarePlatform/hipfort/blob/master/README.md
    end if

    to_field%data = from_field%data !should be valid even with pointers, this is an assigment of value https://stackoverflow.com/questions/61127387/fortran-pointer-assignment-difference-between-and
    
    call hipCheck(hipmemcpydtod(to_field%inner_field%data,from_field%inner_field%data,size(from_field%data)* 8_c_size_t))
    
    to_field%inner_field%nx = from_field%inner_field%nx
    to_field%inner_field%ny = from_field%inner_field%ny
    to_field%inner_field%nz = from_field%inner_field%nz
    to_field%inner_field%nx_full = from_field%inner_field%nx_full
    to_field%inner_field%ny_full = from_field%inner_field%ny_full
    to_field%inner_field%nz_full = from_field%inner_field%nz_full
    to_field%inner_field%dx = from_field%inner_field%dx
    to_field%inner_field%dy = from_field%inner_field%dy
    to_field%inner_field%dz = from_field%inner_field%dz
  end subroutine copy_fields

  function average(field0, parallel) 
    use mpi

    implicit none

    real(dp) :: average
    type(field) :: field0
    type(parallel_data), intent(in) :: parallel

    real(dp) :: local_average
    integer :: rc
    integer :: i,j,k
    !update host memory
    call hipCheck(hipMemcpy(c_loc(field0%data), field0%inner_field%data, size(field0%data)* 8_c_size_t, hipMemcpyDeviceToHost))

    !set of prints to be used with small grids and other initialization values to 
    !peek under the hood of the matrix after the evolve kernel is called.
    !!if (parallel%rank == 1) then    
    !  do i=0,field0%inner_field%nx+1
    !    do j=0,field0%inner_field%ny+1
    !      do k=0,field0%inner_field%nz+1
    !        write (*,"(F15.10)", advance="no") field0%data(i,j,k)
    !      enddo
    !      write (*,*) parallel%rank, 'end of line'
    !    enddo
    !    write (*,*) parallel%rank, 'end of page'
    !  enddo
    !!endif


    local_average = sum(field0%data(1:field0%inner_field%nx, 1:field0%inner_field%ny, 1:field0%inner_field%nz))
    call mpi_allreduce(local_average, average, 1, MPI_DOUBLE_PRECISION, MPI_SUM,  &
               &       MPI_COMM_WORLD, rc)
    average = average / (field0%inner_field%nx_full * field0%inner_field%ny_full * field0%inner_field%nz_full)
    
  end function average

end module utilities
