#include <math.h>
#include <cstdlib>
#include "neighbor_list.h"
#include "global.h"
#include "GPUvars.h"
#include <vector_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#define SECTION_SIZE 1024
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define cudaCheck(error) \
  if (error != cudaSuccess) { \
    printf("CUDA Error: %s at %s:%d\n", \
      cudaGetErrorString(error), \
      __FILE__, __LINE__); \
    exit(1); \
              }

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

__device__ __constant__ double dev_coeff_att[3][3] = {
    {0.0, 0.0, 0.0},
	{0.0, 0.7, 0.8},
	{0.0, 0.8, 1.0}
};

__device__ __constant__ double dev_coeff_rep[3][3] = {
    {0.0, 0.0, 0.0},
	{0.0, 1.0, 1.0},
	{0.0, 1.0, 1.0}
};

__device__ __constant__ double dev_force_coeff_att[3][3] = {
    {0.0,       0.0,       0.0},
	{0.0, -12.0*1.0, -12.0*0.8},
	{0.0, -12.0*0.8, -12.0*0.7}
};

__device__ __constant__ double dev_force_coeff_rep[3][3] = {
    {0.0,       0.0,       0.0},
	{0.0,  -6.0*1.0,  -6.0*1.0},
	{0.0,  -6.0*1.0,  -6.0*1.0}
};

__device__ __constant__ double dev_sigma_rep[3][3] = {
    {0.0, 0.0, 0.0},
	{0.0, 3.8, 5.4},
	{0.0, 5.4, 7.0}
};

__device__ __constant__ double dev_rcut_nat[3][3] = {
    { 0.0,  0.0,  0.0},
    { 0.0,  8.0, 11.0},
    { 0.0, 11.0, 14.0}
};


void update_neighbor_list_CL(){
    int N;

    host_to_device(0);

    N = ncon_att+1;

    calculate_array_native(boxl, N);

    compact_native_CL();

    N = ncon_rep+1;

    calculate_array_non_native(boxl, N);

    compact_non_native_CL();

}

void compact_native_CL(){
    int N;
    
    N = ncon_att+1;

    // typedef these iterators for shorthand
    typedef thrust::device_vector<int>::iterator   IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;

    // typedef a tuple of these iterators
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator, DoubleIterator, DoubleIterator, DoubleIterator, DoubleIterator> IteratorTuple;
    typedef thrust::tuple<int *, int *, int *, int *, double *, double *, double *, double *> HostIteratorTuple;

    // typedef the zip_iterator of this tuple
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::zip_iterator<HostIteratorTuple> HostZipIterator;

    // Create device initial vectors
    thrust::device_vector<int> dev_ibead_lj_nat_vec(dev_ibead_lj_nat, dev_ibead_lj_nat+N);
    thrust::device_vector<int> dev_jbead_lj_nat_vec(dev_jbead_lj_nat, dev_jbead_lj_nat+N);
    thrust::device_vector<int> dev_itype_lj_nat_vec(dev_itype_lj_nat, dev_itype_lj_nat+N);
    thrust::device_vector<int> dev_jtype_lj_nat_vec(dev_jtype_lj_nat, dev_jtype_lj_nat+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist_vec(dev_lj_nat_pdb_dist, dev_lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist2_vec(dev_lj_nat_pdb_dist2, dev_lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist6_vec(dev_lj_nat_pdb_dist6, dev_lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist12_vec(dev_lj_nat_pdb_dist12, dev_lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value_vec(dev_value_int, dev_value_int+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_att_vec(N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att_vec(N);
    thrust::device_vector<int> dev_itype_neighbor_list_att_vec(N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att_vec(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist_vec(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2_vec(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6_vec(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12_vec(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_lj_nat_vec.begin(), dev_jbead_lj_nat_vec.begin(), dev_itype_lj_nat_vec.begin(), dev_jtype_lj_nat_vec.begin(),
                                            dev_lj_nat_pdb_dist_vec.begin(), dev_lj_nat_pdb_dist2_vec.begin(), dev_lj_nat_pdb_dist6_vec.begin(), dev_lj_nat_pdb_dist12_vec.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_lj_nat_vec.end(), dev_jbead_lj_nat_vec.end(), dev_itype_lj_nat_vec.end(), dev_jtype_lj_nat_vec.end(),
                                            dev_lj_nat_pdb_dist_vec.end(), dev_lj_nat_pdb_dist2_vec.end(), dev_lj_nat_pdb_dist6_vec.end(), dev_lj_nat_pdb_dist12_vec.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_att_vec.begin(), dev_jbead_neighbor_list_att_vec.begin(), dev_itype_neighbor_list_att_vec.begin(),
                                            dev_jtype_neighbor_list_att_vec.begin(), dev_nl_lj_nat_pdb_dist_vec.begin(), dev_nl_lj_nat_pdb_dist2_vec.begin(), 
                                            dev_nl_lj_nat_pdb_dist6_vec.begin(), dev_nl_lj_nat_pdb_dist12_vec.begin()));

    thrust::sort_by_key(dev_value_vec.begin(), dev_value_vec.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value_vec.begin(), dev_value_vec.end(), dev_value_vec.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value_vec.end()-1, dev_value_vec.end(), &arrSize);

    nnl_att = arrSize;

    HostZipIterator host_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_att, dev_jbead_neighbor_list_att, dev_itype_neighbor_list_att,
                                            dev_jtype_neighbor_list_att, dev_nl_lj_nat_pdb_dist, dev_nl_lj_nat_pdb_dist2, dev_nl_lj_nat_pdb_dist6, dev_nl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_neighbor_list_att_vec.begin() + nnl_att, dev_jbead_neighbor_list_att_vec.begin() + nnl_att,
                                                dev_itype_neighbor_list_att_vec.begin() + nnl_att, dev_jtype_neighbor_list_att_vec.begin() + nnl_att, 
                                                dev_nl_lj_nat_pdb_dist_vec.begin() + nnl_att, dev_nl_lj_nat_pdb_dist2_vec.begin() + nnl_att, 
                                                dev_nl_lj_nat_pdb_dist6_vec.begin() + nnl_att, dev_nl_lj_nat_pdb_dist12_vec.begin() + nnl_att));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nnl_att--;
}

void compact_non_native_CL(){
    int N;

    N = ncon_rep+1;

    // typedef these iterators for shorthand
    typedef thrust::device_vector<int>::iterator   IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;

    // typedef a tuple of these iterators
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator> IteratorTuple;
    typedef thrust::tuple<int *, int *, int *, int *> HostIteratorTuple;

    // typedef the zip_iterator of this tuple
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::zip_iterator<HostIteratorTuple> HostZipIterator;

    // Create device initial vectors
    thrust::device_vector<int> dev_ibead_lj_non_nat_vec(dev_ibead_lj_non_nat, dev_ibead_lj_non_nat+N);
    thrust::device_vector<int> dev_jbead_lj_non_nat_vec(dev_jbead_lj_non_nat, dev_jbead_lj_non_nat+N);
    thrust::device_vector<int> dev_itype_lj_non_nat_vec(dev_itype_lj_non_nat, dev_itype_lj_non_nat+N);
    thrust::device_vector<int> dev_jtype_lj_non_nat_vec(dev_jtype_lj_non_nat, dev_jtype_lj_non_nat+N);

    // Create device value vector
    thrust::device_vector<int> dev_value_vec(dev_value_int, dev_value_int+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_rep_vec(N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep_vec(N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep_vec(N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep_vec(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_lj_non_nat_vec.begin(), dev_jbead_lj_non_nat_vec.begin(), dev_itype_lj_non_nat_vec.begin(), dev_jtype_lj_non_nat_vec.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_lj_non_nat_vec.end(), dev_jbead_lj_non_nat_vec.end(), dev_itype_lj_non_nat_vec.end(), dev_jtype_lj_non_nat_vec.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep_vec.begin(), dev_jbead_neighbor_list_rep_vec.begin(), dev_itype_neighbor_list_rep_vec.begin(),
                                            dev_jtype_neighbor_list_rep_vec.begin()));

    thrust::sort_by_key(dev_value_vec.begin(), dev_value_vec.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value_vec.begin(), dev_value_vec.end(), dev_value_vec.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value_vec.end()-1, dev_value_vec.end(), &arrSize);

    nnl_rep = arrSize;

    HostZipIterator host_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep, dev_jbead_neighbor_list_rep, dev_itype_neighbor_list_rep, dev_jtype_neighbor_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_neighbor_list_rep_vec.begin() + nnl_rep, dev_jbead_neighbor_list_rep_vec.begin() + nnl_rep,
                                                dev_itype_neighbor_list_rep_vec.begin() + nnl_rep, dev_jtype_neighbor_list_rep_vec.begin() + nnl_rep));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nnl_rep--;
}

void update_neighbor_list() {
  device_to_host(0);

  double dx, dy, dz;
  double d2;
  int ibead, jbead, itype, jtype;
  double rcut, rcut2;

  nnl_att = 0;
  nnl_rep = 0;

  // calculations for native (attractiction) contacts
  for (int i=1; i<=ncon_att; i++) {
    // record sigma for ibead and jbead
    ibead = ibead_lj_nat[i];
    jbead = jbead_lj_nat[i];

    // record type of bead for ibead and jbead
    itype = itype_lj_nat[i];
    jtype = jtype_lj_nat[i];
    
    // calculate distance in x, y, and z for ibead and jbead
    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    // apply periodic boundary conditions to dx, dy, and dz
    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    // compute square of distance between ibead and jbead
    d2 = dx*dx+dy*dy+dz*dz;

    /* 
    Compute the cutoff distance for the given bead
    This is based off of lj_nat_pdb_dist[i], which is the distance 
    from ibead to jbead in the resulting folded structure
    */
    rcut = 3.2*lj_nat_pdb_dist[i];

    // square cutoff distance, since sqrt(d2) is computationally expensive
    rcut2 = rcut*rcut;

    // checks if distance squared is less than the cutoff distance squared
    if (d2 < rcut2) {
      // add to neighbor list
      nnl_att++;
      // add pair to respective attraction neighbor lists
      ibead_neighbor_list_att[nnl_att] = ibead;
      jbead_neighbor_list_att[nnl_att] = jbead;
      
      // record type of each bead
      itype_neighbor_list_att[nnl_att] = itype;
      jtype_neighbor_list_att[nnl_att] = jtype;

      // record values, so that calculatons are not repeated (look-up table)
      nl_lj_nat_pdb_dist[nnl_att] = lj_nat_pdb_dist[i];
      nl_lj_nat_pdb_dist2[nnl_att] = lj_nat_pdb_dist2[i];
      nl_lj_nat_pdb_dist6[nnl_att] = lj_nat_pdb_dist6[i];
      nl_lj_nat_pdb_dist12[nnl_att] = lj_nat_pdb_dist12[i];
    }
  }

  // calculations for non-native (repulsive) contacts
  for (int i=1; i<=ncon_rep; i++) {
    // record sigma for ibead and jbead
    ibead = ibead_lj_non_nat[i];
    jbead = jbead_lj_non_nat[i];

    // record type of bead for ibead and jbead
    itype = itype_lj_non_nat[i];
    jtype = jtype_lj_non_nat[i];

    // calculate distance in x, y, and z for ibead and jbead
    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    // apply periodic boundary conditions to dx, dy, and dz
    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    // compute square of distance between ibead and jbead
    d2 = dx*dx+dy*dy+dz*dz;

    /* 
    Compute the cutoff distance for the given bead
    This is based off of sigma_rep[itype][jtype],
    is based on the sigma for the types of ibead and jbead
    */
    rcut = 3.2*sigma_rep[itype][jtype];

    // square cutoff distance, since sqrt(d2) is computationally expensive
    rcut2 = rcut*rcut;

    // checks if distance squared is less than the cutoff distance squared
    if (d2 < rcut2) {
      // add to neighbor list
      nnl_rep++;

      // add pair to respective repulsive neighbor lists
      ibead_neighbor_list_rep[nnl_rep] = ibead;
      jbead_neighbor_list_rep[nnl_rep] = jbead;

      // record type of each bead
      itype_neighbor_list_rep[nnl_rep] = itype;
      jtype_neighbor_list_rep[nnl_rep] = jtype;
    }
  }
}

void update_neighbor_list_thrust(){
    int N;

    host_to_device(0);

    N = ncon_att+1;

    calculate_array_native(boxl, N);

    compact_native_thrust();

    N = ncon_rep+1;

    calculate_array_non_native(boxl, N);

    compact_non_native_thrust();
}

void compact_native_thrust(){
    int N;

    N = ncon_att+1;

    // typedef these iterators for shorthand
    typedef thrust::device_vector<int>::iterator   IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;

    // typedef a tuple of these iterators
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator, DoubleIterator, DoubleIterator, DoubleIterator, DoubleIterator> IteratorTuple;
    typedef thrust::tuple<int *, int *, int *, int *, double *, double *, double *, double *> HostIteratorTuple;

    // typedef the zip_iterator of this tuple
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::zip_iterator<HostIteratorTuple> HostZipIterator;

    // Create device initial vectors
    thrust::device_vector<int> dev_ibead_lj_nat_vec(dev_ibead_lj_nat, dev_ibead_lj_nat+N);
    thrust::device_vector<int> dev_jbead_lj_nat_vec(dev_jbead_lj_nat, dev_jbead_lj_nat+N);
    thrust::device_vector<int> dev_itype_lj_nat_vec(dev_itype_lj_nat, dev_itype_lj_nat+N);
    thrust::device_vector<int> dev_jtype_lj_nat_vec(dev_jtype_lj_nat, dev_jtype_lj_nat+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist_vec(dev_lj_nat_pdb_dist, dev_lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist2_vec(dev_lj_nat_pdb_dist2, dev_lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist6_vec(dev_lj_nat_pdb_dist6, dev_lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist12_vec(dev_lj_nat_pdb_dist12, dev_lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value_vec(dev_value_int, dev_value_int+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_att_vec(N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att_vec(N);
    thrust::device_vector<int> dev_itype_neighbor_list_att_vec(N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att_vec(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist_vec(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2_vec(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6_vec(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12_vec(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_lj_nat_vec.begin(), dev_jbead_lj_nat_vec.begin(), dev_itype_lj_nat_vec.begin(), dev_jtype_lj_nat_vec.begin(),
                                            dev_lj_nat_pdb_dist_vec.begin(), dev_lj_nat_pdb_dist2_vec.begin(), dev_lj_nat_pdb_dist6_vec.begin(), dev_lj_nat_pdb_dist12_vec.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_lj_nat_vec.end(), dev_jbead_lj_nat_vec.end(), dev_itype_lj_nat_vec.end(), dev_jtype_lj_nat_vec.end(),
                                            dev_lj_nat_pdb_dist_vec.end(), dev_lj_nat_pdb_dist2_vec.end(), dev_lj_nat_pdb_dist6_vec.end(), dev_lj_nat_pdb_dist12_vec.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_att_vec.begin(), dev_jbead_neighbor_list_att_vec.begin(), dev_itype_neighbor_list_att_vec.begin(),
                                            dev_jtype_neighbor_list_att_vec.begin(), dev_nl_lj_nat_pdb_dist_vec.begin(), dev_nl_lj_nat_pdb_dist2_vec.begin(), 
                                            dev_nl_lj_nat_pdb_dist6_vec.begin(), dev_nl_lj_nat_pdb_dist12_vec.begin()));

    nnl_att = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value_vec.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    HostZipIterator host_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_att, dev_jbead_neighbor_list_att, dev_itype_neighbor_list_att,
                                            dev_jtype_neighbor_list_att, dev_nl_lj_nat_pdb_dist, dev_nl_lj_nat_pdb_dist2, dev_nl_lj_nat_pdb_dist6, dev_nl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_neighbor_list_att_vec.begin() + nnl_att, dev_jbead_neighbor_list_att_vec.begin() + nnl_att,
                                                dev_itype_neighbor_list_att_vec.begin() + nnl_att, dev_jtype_neighbor_list_att_vec.begin() + nnl_att, 
                                                dev_nl_lj_nat_pdb_dist_vec.begin() + nnl_att, dev_nl_lj_nat_pdb_dist2_vec.begin() + nnl_att, 
                                                dev_nl_lj_nat_pdb_dist6_vec.begin() + nnl_att, dev_nl_lj_nat_pdb_dist12_vec.begin() + nnl_att));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nnl_att--;
}

void compact_non_native_thrust(){
    int N;

    N = ncon_rep+1;

    // typedef these iterators for shorthand
    typedef thrust::device_vector<int>::iterator   IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;

    // typedef a tuple of these iterators
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator> IteratorTuple;
    typedef thrust::tuple<int *, int *, int *, int *> HostIteratorTuple;

    // typedef the zip_iterator of this tuple
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::zip_iterator<HostIteratorTuple> HostZipIterator;

    // Create device initial vectors
    thrust::device_vector<int> dev_ibead_lj_non_nat_vec(dev_ibead_lj_non_nat, dev_ibead_lj_non_nat+N);
    thrust::device_vector<int> dev_jbead_lj_non_nat_vec(dev_jbead_lj_non_nat, dev_jbead_lj_non_nat+N);
    thrust::device_vector<int> dev_itype_lj_non_nat_vec(dev_itype_lj_non_nat, dev_itype_lj_non_nat+N);
    thrust::device_vector<int> dev_jtype_lj_non_nat_vec(dev_jtype_lj_non_nat, dev_jtype_lj_non_nat+N);

    // Create device value vector
    thrust::device_vector<int> dev_value_vec(dev_value_int, dev_value_int+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_rep_vec(N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep_vec(N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep_vec(N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep_vec(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_lj_non_nat_vec.begin(), dev_jbead_lj_non_nat_vec.begin(), dev_itype_lj_non_nat_vec.begin(), dev_jtype_lj_non_nat_vec.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_lj_non_nat_vec.end(), dev_jbead_lj_non_nat_vec.end(), dev_itype_lj_non_nat_vec.end(), dev_jtype_lj_non_nat_vec.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep_vec.begin(), dev_jbead_neighbor_list_rep_vec.begin(), dev_itype_neighbor_list_rep_vec.begin(),
                                            dev_jtype_neighbor_list_rep_vec.begin()));

    nnl_rep = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value_vec.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    HostZipIterator host_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep, dev_jbead_neighbor_list_rep, dev_itype_neighbor_list_rep, dev_jtype_neighbor_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_neighbor_list_rep_vec.begin() + nnl_rep, dev_jbead_neighbor_list_rep_vec.begin() + nnl_rep,
                                                dev_itype_neighbor_list_rep_vec.begin() + nnl_rep, dev_jtype_neighbor_list_rep_vec.begin() + nnl_rep));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nnl_rep--;
}

void update_neighbor_list_RL(){
    // Declare N
	int N;
	
    // Make sure all required GPU arrays are on the device
    host_to_device(0);

	// Set N
	N = ncon_att+1;
	
	// Calculate binary list for att
	calculate_array_native(boxl, N);
    
    nnl_att = compact_native(N) - 1;
	
	/**********************************
	 *								  *
	 * End of Attractive Calculations *
	 *								  *
	 **********************************/
	
	// Set N
	N = ncon_rep+1;
	
	// Calculate binary list for rep
	calculate_array_non_native(boxl, N);
    
    nnl_rep = compact_non_native(N) - 1;
}


void calculate_array_native(int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

   	// Compute binary array
	array_native_kernel<<<blocks, threads>>>(dev_ibead_lj_nat, dev_jbead_lj_nat, dev_itype_lj_nat, dev_jtype_lj_nat, dev_unc_pos, dev_lj_nat_pdb_dist, dev_value_int, boxl, N);
    CudaCheckError();
    // Sync device
    cudaDeviceSynchronize();
}

__global__ void array_native_kernel(int *dev_ibead_lj_nat, int *dev_jbead_lj_nat, int *dev_itype_lj_nat, int *dev_jtype_lj_nat, double3 *dev_unc_pos, double *dev_lj_nat_pdb_dist, 
                            int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i < N){
    double dx, dy, dz;
    double d2;
    int ibead, jbead, itype, jtype;
    double rcut, rcut2;

    // record sigma for ibead and jbead
    ibead = dev_ibead_lj_nat[i];

    jbead = dev_jbead_lj_nat[i];

    // record type of bead for ibead and jbead
    itype = dev_itype_lj_nat[i];

    jtype = dev_jtype_lj_nat[i];
    
    // calculate distance in x, y, and z for ibead and jbead
    dx = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;

    dy = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;

    dz = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;

    // apply periodic boundary conditions to dx, dy, and dz
    //dx -= boxl*rnd(dx/boxl);
    double rnd_value;

    rnd_value = ( ((dx/boxl)>0) ? std::floor((dx/boxl)+0.5) : std::ceil((dx/boxl)-0.5) );
    dx -= boxl*rnd_value;

    //dy -= boxl*rnd(dy/boxl);
    rnd_value = ( ((dy/boxl)>0) ? std::floor((dy/boxl)+0.5) : std::ceil((dy/boxl)-0.5) );
    dy -= boxl*rnd_value;

    //dz -= boxl*rnd(dz/boxl);
    rnd_value = ( ((dz/boxl)>0) ? std::floor((dz/boxl)+0.5) : std::ceil((dz/boxl)-0.5) );
    dz -= boxl*rnd_value;

    // compute square of distance between ibead and jbead
    d2 = dx*dx+dy*dy+dz*dz;

    /* 
    Compute the cutoff distance for the given bead
    This is based off of lj_nat_pdb_dist[i], which is the distance 
    from ibead to jbead in the resulting folded structure
    */
    rcut = 3.2*dev_lj_nat_pdb_dist[i];

    // square cutoff distance, since sqrt(d2) is computationally expensive
    rcut2 = rcut*rcut;

    if(d2 < rcut2){
      dev_value[i] = 1;
    }else{
      dev_value[i] = 0;
    }
  }else if(i == 0){
      dev_value[i] = 1;
  }
}

void calculate_array_non_native(int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	// Compute binary array
	array_non_native_kernel<<<blocks, threads>>>(dev_ibead_lj_non_nat, dev_jbead_lj_non_nat, dev_itype_lj_non_nat, dev_jtype_lj_non_nat, dev_unc_pos, dev_value_int, boxl, N);

    // Sync device
    cudaDeviceSynchronize();
}

__global__ void array_non_native_kernel(int *dev_ibead_lj_non_nat, int *dev_jbead_lj_non_nat, int *dev_itype_lj_non_nat, int *dev_jtype_lj_non_nat, 
                                        double3 *dev_unc_pos, int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i < N){
    double dx, dy, dz;
    double d2;
    int ibead, jbead, itype, jtype;
    double rcut, rcut2;
    
    // record sigma for ibead and jbead
    ibead = dev_ibead_lj_non_nat[i];
    jbead = dev_jbead_lj_non_nat[i];

    // record type of bead for ibead and jbead
    itype = dev_itype_lj_non_nat[i];
    jtype = dev_jtype_lj_non_nat[i];
    
    // calculate distance in x, y, and z for ibead and jbead
    dx = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;
    dy = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;
    dz = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;

    // apply periodic boundary conditions to dx, dy, and dz
    //dx -= boxl*rnd(dx/boxl);
    double rnd_value;

    rnd_value = ( ((dx/boxl)>0) ? std::floor((dx/boxl)+0.5) : std::ceil((dx/boxl)-0.5) );
    dx -= boxl*rnd_value;

    //dy -= boxl*rnd(dy/boxl);
    rnd_value = ( ((dy/boxl)>0) ? std::floor((dy/boxl)+0.5) : std::ceil((dy/boxl)-0.5) );
    dy -= boxl*rnd_value;

    //dz -= boxl*rnd(dz/boxl);
    rnd_value = ( ((dz/boxl)>0) ? std::floor((dz/boxl)+0.5) : std::ceil((dz/boxl)-0.5) );
    dz -= boxl*rnd_value;
    
    // compute square of distance between ibead and jbead
    d2 = dx*dx+dy*dy+dz*dz;

	// May need to change to dev_sigma_rep[N*itype + jtype]
    rcut = 3.2*dev_sigma_rep[itype][jtype];

    // square cutoff distance, since sqrt(d2) is computationally expensive
    rcut2 = rcut*rcut;

    if(d2 < rcut2){
      dev_value[i] = 1;
    }else{
      dev_value[i] = 0;
    }
  }else if(i == 0){
      dev_value[i] = 1;
  }
}

/*
 * Function: hier_ks_scan
 * -----------------
 *  
 *
 *  dev_index: array of indices to check (on GPU)
 *  dev_value: binary value indicating if the corresponding dev_index value is true (1) or false (0) (on GPU)
 *  N: number of elements in dev_index and dev_value
 *  dev_result: pointer where compacted array is stored (on GPU)
 */

void hier_ks_scan(int *dev_X, int *dev_Y, int N, int re){
    if(N <= SECTION_SIZE){
        ksScanInc<<<1, N>>>(dev_X, dev_Y, N);

        cudaDeviceSynchronize();
        CudaCheckError();

        return;
    }else{
        int threads = (int)min(N, SECTION_SIZE);
        int blocks = (int)ceil(1.0*N/SECTION_SIZE);

        int *dev_S;
        cudaCheck(cudaMalloc((void**)&dev_S, (int)ceil(1.0*N/SECTION_SIZE) * sizeof(int)));
        
        ksScanAuxInc<<<blocks, threads>>>(dev_X, dev_Y, N, dev_S);
        cudaDeviceSynchronize();
        CudaCheckError();

        hier_ks_scan(dev_S, dev_S, (int)ceil(1.0*N/SECTION_SIZE), 1);
        cudaDeviceSynchronize();
        
        sumIt<<<blocks, threads>>>(dev_Y, dev_S, N);
        cudaDeviceSynchronize();
        CudaCheckError();

        cudaCheck(cudaFree(dev_S));

        return;
    }
}

__global__ void ksScanAuxExc (int *X, int *Y, int InputSize, int *S) {
    int val;
    
    __shared__ int XY[SECTION_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < InputSize && threadIdx.x != 0){
        XY[threadIdx.x] = X[i-1];
    }else{
        XY[threadIdx.x] = 0;
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *=2){
        __syncthreads();
        if(threadIdx.x >= stride){
            val = XY[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[threadIdx.x] += val;
        }
    }

    __syncthreads();
    if(i < InputSize){
        Y[i] = XY[threadIdx.x];
    }
    
    __syncthreads();
    if(threadIdx.x == 0){
        S[blockIdx.x] = XY[SECTION_SIZE-1];
    }
}

__global__ void ksScanAuxInc (int *X, int *Y, int InputSize, int *S) {
    int val;
    
    __shared__ int XY[SECTION_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < InputSize){
        XY[threadIdx.x] = X[i];
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *=2){
        __syncthreads();
        if(threadIdx.x >= stride){
            val = XY[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[threadIdx.x] += val;
        }
    }

    __syncthreads();
    if(i < InputSize){
        Y[i] = XY[threadIdx.x];
    }
    
    __syncthreads();
    if(threadIdx.x == 0){
        S[blockIdx.x] = XY[SECTION_SIZE-1];
    }
}

__global__ void ksScanExc (int *X, int *Y, int InputSize) {
    int val;
    
    __shared__ int XY[SECTION_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < InputSize && threadIdx.x != 0){
        XY[threadIdx.x] = X[i-1];
    }else{
        XY[threadIdx.x] = 0;
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *=2){
        __syncthreads();
        if(threadIdx.x >= stride){
            val = XY[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[threadIdx.x] += val;
        }
    }

    __syncthreads();
    if(i < InputSize){
        Y[i] = XY[threadIdx.x];
    }
}

__global__ void ksScanInc (int *X, int *Y, int InputSize) {
    int val;
    
    __shared__ int XY[SECTION_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < InputSize){
        XY[threadIdx.x] = X[i];
    }

    for(unsigned int stride = 1; stride < blockDim.x; stride *=2){
        __syncthreads();
        if(threadIdx.x >= stride){
            val = XY[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[threadIdx.x] += val;
        }
    }

    __syncthreads();
    if(i < InputSize){
        Y[i] = XY[threadIdx.x];
    }
}

__global__ void sumIt (int *Y, int *S, int InputSize) {
    if(blockIdx.x > 0){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < InputSize){
            Y[i] += S[blockIdx.x-1];
        }
    }
}

template <typename T>
void allocate_and_copy(T *dev_index, int *dev_value, int *dev_output, int N, int arrSize, T *dev_result_index){
    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    copyElements<<<blocks, threads>>>(dev_index, dev_value, dev_output, dev_result_index, N);

    cudaDeviceSynchronize();
}

/*
 * Function: copyElements
 * -----------------
 *  Copys values marked true (1) from index array to result array
 *
 *  dev_index: array of indices to check (on GPU)
 *  dev_value: binary value indicating if the corresponding dev_index value is true (1) or false (0) (on GPU)
 *  N: number of elements in dev_index and dev_value
 *  dev_result: pointer where compacted array is stored (on GPU)
 */

__global__ void copyElements(double *dev_index, int *dev_value, int *dev_output, double *dev_result, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(dev_value[i] && i < N){
        dev_result[dev_output[i]-1] = dev_index[i];
    }
    return;
}

/*
 * Function: copyElements
 * -----------------
 *  Copys values marked true (1) from index array to result array
 *
 *  dev_index: array of indices to check (on GPU)
 *  dev_value: binary value indicating if the corresponding dev_index value is true (1) or false (0) (on GPU)
 *  N: number of elements in dev_index and dev_value
 *  dev_result: pointer where compacted array is stored (on GPU)
 */

__global__ void copyElements(int *dev_index, int *dev_value, int *dev_output, int *dev_result, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(dev_value[i] && i < N){
        dev_result[dev_output[i]-1] = dev_index[i];
    }
    return;
}

int compact_native(int N){
    // Calculate array size
    int size = N * sizeof(int);

    // Perform hierarchical Kogge-Stone scan on dev_value array and store result in dev_output
    hier_ks_scan(dev_value_int, dev_output_int, N, 0);

    // Copy size of compacted array from device to host and store in arrSize
    int arrSize;
    cudaMemcpy(&arrSize, &dev_output_int[N-1], sizeof(int), cudaMemcpyDeviceToHost); 

    int temp;
    cudaMemcpy(&temp, &dev_value_int[N-1], sizeof(int), cudaMemcpyDeviceToHost); 

    // Increment arrSize by 1 if needed
    if(temp){
        arrSize++;
    }

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    allocate_and_copy<int>(dev_ibead_lj_nat, dev_value_int, dev_output_int, N, arrSize, dev_ibead_neighbor_list_att);
    
    allocate_and_copy<int>(dev_jbead_lj_nat, dev_value_int, dev_output_int, N, arrSize, dev_jbead_neighbor_list_att);

    allocate_and_copy<int>(dev_itype_lj_nat, dev_value_int, dev_output_int, N, arrSize, dev_itype_neighbor_list_att);

    allocate_and_copy<int>(dev_jtype_lj_nat, dev_value_int, dev_output_int, N, arrSize, dev_jtype_neighbor_list_att);

    allocate_and_copy<double>(dev_lj_nat_pdb_dist, dev_value_int, dev_output_int, N, arrSize, dev_nl_lj_nat_pdb_dist);

    allocate_and_copy<double>(dev_lj_nat_pdb_dist2, dev_value_int, dev_output_int, N, arrSize, dev_nl_lj_nat_pdb_dist2);

    allocate_and_copy<double>(dev_lj_nat_pdb_dist6, dev_value_int, dev_output_int, N, arrSize, dev_nl_lj_nat_pdb_dist6);

    allocate_and_copy<double>(dev_lj_nat_pdb_dist12, dev_value_int, dev_output_int, N, arrSize, dev_nl_lj_nat_pdb_dist12);

    return arrSize-1;
}


int compact_non_native(int N){
    // Calculate array size
    int size = N * sizeof(int);

    // Perform hierarchical Kogge-Stone scan on dev_value array and store result in dev_output
    hier_ks_scan(dev_value_int, dev_output_int, N, 0);

    // Copy size of compacted array from device to host and store in arrSize
    int arrSize;
    cudaMemcpy(&arrSize, &dev_output_int[N-1], sizeof(int), cudaMemcpyDeviceToHost); 

    int temp;
    cudaMemcpy(&temp, &dev_value_int[N-1], sizeof(int), cudaMemcpyDeviceToHost); 

    // Increment arrSize by 1 if needed
    if(temp){
        arrSize++;
    }

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    allocate_and_copy<int>(dev_ibead_lj_non_nat, dev_value_int, dev_output_int, N, arrSize, dev_ibead_neighbor_list_rep);

    allocate_and_copy<int>(dev_jbead_lj_non_nat, dev_value_int, dev_output_int, N, arrSize, dev_jbead_neighbor_list_rep);

    allocate_and_copy<int>(dev_itype_lj_non_nat, dev_value_int, dev_output_int, N, arrSize, dev_itype_neighbor_list_rep);

    allocate_and_copy<int>(dev_jtype_lj_non_nat, dev_value_int, dev_output_int, N, arrSize, dev_jtype_neighbor_list_rep);

    return arrSize-1;
}