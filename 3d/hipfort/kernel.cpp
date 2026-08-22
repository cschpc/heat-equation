#include <hip/hip_runtime.h>

//same data struct declared on the fortran side.
struct Field {
    // nx and ny are the true dimensions of the field. The temperature matrix
    // contains also ghost layers, so it will have dimensions nx+2 x ny+2 
    int nx;                     // Local dimensions of the field
    int ny;
    int nz;
    int nx_full;                // Global dimensions of the field
    int ny_full;                // Global dimensions of the field
    int nz_full;                // Global dimensions of the field
    double dx = 0.01;           // Grid spacing
    double dy = 0.01;
    double dz = 0.01;
    double* data;
};


// Update the temperature values using five-point stencil */
__global__ void evolve_kernel(double *currdata, double *prevdata, double a, double dt, int nx, int ny, int nz,
                       double inv_dx2, double inv_dy2, double inv_dz2, int max_threads)
{

    //linearize the thread id to have coalesced accesses
    int ythr_offs = threadIdx.y * blockDim.x + threadIdx.x;
    int zthr_offs = threadIdx.z * blockDim.y * blockDim.x+ ythr_offs;
    int xblock_offs = blockIdx.x * blockDim.z * blockDim.y * blockDim.x + zthr_offs;
    int yblock_offs = blockIdx.y * blockDim.z * blockDim.y * blockDim.x * gridDim.x + xblock_offs;
    int tid = blockIdx.z * blockDim.z * blockDim.y * blockDim.x * gridDim.x * gridDim.y + yblock_offs;


    //i only want to run significative computations, so i need to map the tid to the real memory addresses
    //this means that i have some "holes" where the halo is mapped.
    //to avoid too many uncoalesced accesses, we decide to "mask" the small skip (i.e. the halo on the central dimension)
    //but we will avoid to evaluate the big skip (i.e. the halo surface when we change on the third dimension)
    int start_offs = (ny + 2) * (nx + 2) +  (nx + 2);// 1 face + 1 line are the halos, must not be computated. 
    int big_skip_size = (ny + 2) + (nx + 2) ;//+ 2 //not completely sure, could be 2* nx+2 or 2*ny+2 
    int num_big_skips = tid / ((nx+2)*ny); //again not completely sure, could be 2nx or 2ny
    int ind = tid + start_offs + num_big_skips * big_skip_size;
     
               

    //keep load coalesced (i.e. do load also for thread that must not compute)
    auto tmpdata = prevdata[ind];
    auto ip = prevdata[ind + ((nx+2)*(ny+2))];
    auto im = prevdata[ind - ((nx+2)*(ny+2))];
    auto jp = prevdata[ind + (nx+2)];
    auto jm = prevdata[ind - (nx+2)];
    auto kp = prevdata[ind + 1];
    auto km = prevdata[ind - 1];

    //here we really exclude from the computation the "small skips", they are the first and the last on the innermost dimension
    //e.g., if nx is 4, we need to exclude tid 0,5,6,11,12,... because there is where the halo is.
    if (!((tid % (nx+2)) == 0 || ((tid - (nx+1)) % (nx+2)) == 0) && tid < max_threads)
      
      //these printfs could be useful to understand the mapping of tid to memory idx.
      //to use them you need to reduce the total dimension of the problem to something manageable (e.g. 4,4,4)
      //and initialize the grid values with understandable numbers (i.e. 100*x+10*y+z)
      //in this way you will have a set of print with all the "active" indexes

      /*printf ("thread is block %d,%d,%d, is thread %d,%d,%d, tid %d, ind is %d, and value %f \n ",
        (int)blockIdx.x,(int)blockIdx.y,(int)blockIdx.z,
        (int)threadIdx.x,(int)threadIdx.y,(int)threadIdx.z,
        tid,ind,
        prevdata[ind]);*/
        /*printf("ind is %f and my halo are %f %f %f %f %f %f \n",
        prevdata[ind],
        prevdata[ind+(nx+2)],           //y
        prevdata[ind-(nx+2)],           //y
        prevdata[ind+((nx+2)*(ny+2))],  //x   //i think the x-z inversion is due to fortran col major
        prevdata[ind-((nx+2)*(ny+2))],  //x   //i think the x-z inversion is due to fortran col major
        prevdata[ind+1],                //z   //i think the x-z inversion is due to fortran col major
        prevdata[ind-1]                 //z   //i think the x-z inversion is due to fortran col major
        );*/

      tmpdata = tmpdata + a * dt * (
                  ( ip - 2.0 * tmpdata + im ) * inv_dx2 + //fortran matrix, column major. farthest point is x dimension
                  ( jp - 2.0 * tmpdata + jm ) * inv_dy2 +
                  ( kp - 2.0 * tmpdata + km ) * inv_dz2
      );
      //keep store coalesced. note that we used tmpdata local variable only for this reason.
      //halo points will load and store it, without updating.
      currdata[ind] = tmpdata;
    
    
}
//i did try to keep everything cpp as an attempt to make the HIP API work, however it failed
//i still prefer having this structure because it can allow templating of kernels and so on,
//so i'll keep it as a "good practice" thing
void evolvecpp(Field* curr, const Field* prev, const double a, const double dt)
{
  //note that in the innermost dimension we do nx +2 to insert the "small skips"
  //in the set of threads running. ny and nz don't need to be increased.
  int max_threads = (curr->nx+2)* (curr->ny) * (curr->nz);
  int blocks,grids;
  //i'd like to use the optimal occupancy but the API seems to be broken.
  //I'm not sure what is happening there but cannot use that
  //hipOccupancyMaxPotentialBlockSize(&grids,&blocks,evolve_kernel, 0, max_threads);
  
  //lets assign a block size of 256, this parameter can be played with.
  blocks = 256;
  grids = (max_threads + blocks -1) / blocks;

  auto inv_dx2 = 1.0 / (prev->dx * prev->dx);
  auto inv_dy2 = 1.0 / (prev->dy * prev->dy);
  auto inv_dz2 = 1.0 / (prev->dz * prev->dz);
  
  hipLaunchKernelGGL((evolve_kernel), grids,blocks, 0, 0, curr->data, prev->data, a, dt, curr->nx, curr->ny, curr->nz,inv_dx2,inv_dy2,inv_dz2, max_threads );
}



extern "C"
{
  void evolve(Field* curr, const Field* prev, const double a, const double dt)
  {
    //launch the wrapper
    evolvecpp(curr,prev,a,dt);
  }
}


