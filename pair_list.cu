#include "neighbor_list.h"
#include "cell_list.h"
#include "pair_list.h"
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

void update_pair_list() {

  using namespace std;

  device_to_host(1);

  // declare host variables
  double dx, dy, dz;
  double d2;
  unsigned int ibead, jbead, itype, jtype;
  double rcut, rcut2;

  nil_att = 0;
  nil_rep = 0;

  // declare device variables

  // should be native distance
  for (int i=1; i<=nnl_att; i++) {

    ibead = ibead_neighbor_list_att[i];
    jbead = jbead_neighbor_list_att[i];
    itype = itype_neighbor_list_att[i];
    jtype = jtype_neighbor_list_att[i];

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;

    rcut = 2.5*nl_lj_nat_pdb_dist[i];
    rcut2 = rcut*rcut;

    if (d2 < rcut2) {
      // add to interaction pair list
      nil_att++;
      ibead_pair_list_att[nil_att] = ibead;
      jbead_pair_list_att[nil_att] = jbead;
      itype_pair_list_att[nil_att] = itype;
      jtype_pair_list_att[nil_att] = jtype;
      pl_lj_nat_pdb_dist[nil_att] = nl_lj_nat_pdb_dist[i];
      pl_lj_nat_pdb_dist2[nil_att] = nl_lj_nat_pdb_dist2[i];
      pl_lj_nat_pdb_dist6[nil_att] = nl_lj_nat_pdb_dist6[i];
      pl_lj_nat_pdb_dist12[nil_att] = nl_lj_nat_pdb_dist12[i];
    }

  }

  assert(nil_att >= 0);

  for (int i=1; i<=nnl_rep; i++) {

    ibead = ibead_neighbor_list_rep[i];
    jbead = jbead_neighbor_list_rep[i];
    itype = itype_neighbor_list_rep[i];
    jtype = jtype_neighbor_list_rep[i];

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;

    rcut = 2.5*sigma_rep[itype][jtype];
    rcut2 = rcut*rcut;

    if (d2 < rcut2) {
      // add to interaction pair list
      nil_rep++;
      ibead_pair_list_rep[nil_rep] = ibead;
      jbead_pair_list_rep[nil_rep] = jbead;
      itype_pair_list_rep[nil_rep] = itype;
      jtype_pair_list_rep[nil_rep] = jtype;
    }
  }

  assert(nil_rep >= 0);
  
  if(debug){
    printf("nil_att: %d\nnil_rep: %d\n", nil_att, nil_rep);
    fflush(stdout);
  }
}


void update_pair_list_CL(){
    int N;

    host_to_device(1);

    N = nnl_att;

    calculate_array_native_pl(boxl, N);

    compact_native_pl_CL();

    N = nnl_rep;

    calculate_array_non_native_pl(boxl, N);

    compact_non_native_pl_CL();
}

void compact_native_pl_CL(){
    int N;
    
    N = nnl_att;

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
    thrust::device_vector<int> dev_ibead_neighbor_list_att_vec(dev_ibead_neighbor_list_att, dev_ibead_neighbor_list_att+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att_vec(dev_jbead_neighbor_list_att, dev_jbead_neighbor_list_att+N);
    thrust::device_vector<int> dev_itype_neighbor_list_att_vec(dev_itype_neighbor_list_att, dev_itype_neighbor_list_att+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att_vec(dev_jtype_neighbor_list_att, dev_jtype_neighbor_list_att+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist_vec(dev_nl_lj_nat_pdb_dist, dev_nl_lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2_vec(dev_nl_lj_nat_pdb_dist2, dev_nl_lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6_vec(dev_nl_lj_nat_pdb_dist6, dev_nl_lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12_vec(dev_nl_lj_nat_pdb_dist12, dev_nl_lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value_vec(dev_value_int, dev_value_int+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_att_vec(N);
    thrust::device_vector<int> dev_jbead_pair_list_att_vec(N);
    thrust::device_vector<int> dev_itype_pair_list_att_vec(N);
    thrust::device_vector<int> dev_jtype_pair_list_att_vec(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist_vec(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist2_vec(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist6_vec(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist12_vec(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_att_vec.begin(), dev_jbead_neighbor_list_att_vec.begin(), dev_itype_neighbor_list_att_vec.begin(), dev_jtype_neighbor_list_att_vec.begin(),
                                            dev_nl_lj_nat_pdb_dist_vec.begin(), dev_nl_lj_nat_pdb_dist2_vec.begin(), dev_nl_lj_nat_pdb_dist6_vec.begin(), dev_nl_lj_nat_pdb_dist12_vec.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_att_vec.end(), dev_jbead_neighbor_list_att_vec.end(), dev_itype_neighbor_list_att_vec.end(), dev_jtype_neighbor_list_att_vec.end(),
                                            dev_nl_lj_nat_pdb_dist_vec.end(), dev_nl_lj_nat_pdb_dist2_vec.end(), dev_nl_lj_nat_pdb_dist6_vec.end(), dev_nl_lj_nat_pdb_dist12_vec.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_att_vec.begin(), dev_jbead_pair_list_att_vec.begin(), dev_itype_pair_list_att_vec.begin(),
                                            dev_jtype_pair_list_att_vec.begin(), dev_pl_lj_nat_pdb_dist_vec.begin(), dev_pl_lj_nat_pdb_dist2_vec.begin(), 
                                            dev_pl_lj_nat_pdb_dist6_vec.begin(), dev_pl_lj_nat_pdb_dist12_vec.begin()));

    thrust::sort_by_key(dev_value_vec.begin(), dev_value_vec.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value_vec.begin(), dev_value_vec.end(), dev_value_vec.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value_vec.end()-1, dev_value_vec.end(), &arrSize);

    nil_att = arrSize;

    HostZipIterator host_result_begin(thrust::make_tuple(dev_ibead_pair_list_att, dev_jbead_pair_list_att, dev_itype_pair_list_att,
                                            dev_jtype_pair_list_att, dev_pl_lj_nat_pdb_dist, dev_pl_lj_nat_pdb_dist2, dev_pl_lj_nat_pdb_dist6, dev_pl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_att_vec.begin() + nil_att, dev_jbead_pair_list_att_vec.begin() + nil_att,
                                                dev_itype_pair_list_att_vec.begin() + nil_att, dev_jtype_pair_list_att_vec.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist_vec.begin() + nil_att, dev_pl_lj_nat_pdb_dist2_vec.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist6_vec.begin() + nil_att, dev_pl_lj_nat_pdb_dist12_vec.begin() + nil_att));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nil_att--;
}

void compact_non_native_pl_CL(){
    int N;

    N = nnl_rep;

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
    thrust::device_vector<int> dev_ibead_neighbor_list_rep_vec(dev_ibead_neighbor_list_rep, dev_ibead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep_vec(dev_jbead_neighbor_list_rep, dev_jbead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep_vec(dev_itype_neighbor_list_rep, dev_itype_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep_vec(dev_jtype_neighbor_list_rep, dev_jtype_neighbor_list_rep+N);

    // Create device value vector
    thrust::device_vector<int> dev_value_vec(dev_value_int, dev_value_int+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_rep_vec(N);
    thrust::device_vector<int> dev_jbead_pair_list_rep_vec(N);
    thrust::device_vector<int> dev_itype_pair_list_rep_vec(N);
    thrust::device_vector<int> dev_jtype_pair_list_rep_vec(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep_vec.begin(), dev_jbead_neighbor_list_rep_vec.begin(), dev_itype_neighbor_list_rep_vec.begin(), dev_jtype_neighbor_list_rep_vec.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_rep_vec.end(), dev_jbead_neighbor_list_rep_vec.end(), dev_itype_neighbor_list_rep_vec.end(), dev_jtype_neighbor_list_rep_vec.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_rep_vec.begin(), dev_jbead_pair_list_rep_vec.begin(), dev_itype_pair_list_rep_vec.begin(),
                                            dev_jtype_pair_list_rep_vec.begin()));

    thrust::sort_by_key(dev_value_vec.begin(), dev_value_vec.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value_vec.begin(), dev_value_vec.end(), dev_value_vec.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value_vec.end()-1, dev_value_vec.end(), &arrSize);

    nil_rep = arrSize;

    HostZipIterator host_result_begin(thrust::make_tuple(dev_ibead_pair_list_rep, dev_jbead_pair_list_rep, dev_itype_pair_list_rep, dev_jtype_pair_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_rep_vec.begin() + nil_rep, dev_jbead_pair_list_rep_vec.begin() + nil_rep,
                                                dev_itype_pair_list_rep_vec.begin() + nil_rep, dev_jtype_pair_list_rep_vec.begin() + nil_rep));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nil_rep--;
}

void update_pair_list_thrust(){
    int N;

    host_to_device(1);

    N = nnl_att;

    calculate_array_native_pl(boxl, N);

    compact_native_pl_thrust();

    N = nnl_rep;

    calculate_array_non_native_pl(boxl, N);

    compact_non_native_pl_thrust();
}

void compact_native_pl_thrust(){
    int N;

    N = nnl_att;

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
    thrust::device_vector<int> dev_ibead_neighbor_list_att_vec(dev_ibead_neighbor_list_att, dev_ibead_neighbor_list_att+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att_vec(dev_jbead_neighbor_list_att, dev_jbead_neighbor_list_att+N);
    thrust::device_vector<int> dev_itype_neighbor_list_att_vec(dev_itype_neighbor_list_att, dev_itype_neighbor_list_att+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att_vec(dev_jtype_neighbor_list_att, dev_jtype_neighbor_list_att+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist_vec(dev_nl_lj_nat_pdb_dist, dev_nl_lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2_vec(dev_nl_lj_nat_pdb_dist2, dev_nl_lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6_vec(dev_nl_lj_nat_pdb_dist6, dev_nl_lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12_vec(dev_nl_lj_nat_pdb_dist12, dev_nl_lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value_vec(dev_value_int, dev_value_int+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_att_vec(N);
    thrust::device_vector<int> dev_jbead_pair_list_att_vec(N);
    thrust::device_vector<int> dev_itype_pair_list_att_vec(N);
    thrust::device_vector<int> dev_jtype_pair_list_att_vec(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist_vec(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist2_vec(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist6_vec(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist12_vec(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_att_vec.begin(), dev_jbead_neighbor_list_att_vec.begin(), dev_itype_neighbor_list_att_vec.begin(), dev_jtype_neighbor_list_att_vec.begin(),
                                            dev_nl_lj_nat_pdb_dist_vec.begin(), dev_nl_lj_nat_pdb_dist2_vec.begin(), dev_nl_lj_nat_pdb_dist6_vec.begin(), dev_nl_lj_nat_pdb_dist12_vec.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_att_vec.end(), dev_jbead_neighbor_list_att_vec.end(), dev_itype_neighbor_list_att_vec.end(), dev_jtype_neighbor_list_att_vec.end(),
                                            dev_nl_lj_nat_pdb_dist_vec.end(), dev_nl_lj_nat_pdb_dist2_vec.end(), dev_nl_lj_nat_pdb_dist6_vec.end(), dev_nl_lj_nat_pdb_dist12_vec.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_att_vec.begin(), dev_jbead_pair_list_att_vec.begin(), dev_itype_pair_list_att_vec.begin(),
                                            dev_jtype_pair_list_att_vec.begin(), dev_pl_lj_nat_pdb_dist_vec.begin(), dev_pl_lj_nat_pdb_dist2_vec.begin(), 
                                            dev_pl_lj_nat_pdb_dist6_vec.begin(), dev_pl_lj_nat_pdb_dist12_vec.begin()));

    nil_att = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value_vec.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    HostZipIterator host_result_begin(thrust::make_tuple(dev_ibead_pair_list_att, dev_jbead_pair_list_att, dev_itype_pair_list_att,
                                            dev_jtype_pair_list_att, dev_pl_lj_nat_pdb_dist, dev_pl_lj_nat_pdb_dist2, dev_pl_lj_nat_pdb_dist6, dev_pl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_att_vec.begin() + nil_att, dev_jbead_pair_list_att_vec.begin() + nil_att,
                                                dev_itype_pair_list_att_vec.begin() + nil_att, dev_jtype_pair_list_att_vec.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist_vec.begin() + nil_att, dev_pl_lj_nat_pdb_dist2_vec.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist6_vec.begin() + nil_att, dev_pl_lj_nat_pdb_dist12_vec.begin() + nil_att));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nil_att--;
}

void compact_non_native_pl_thrust(){
    int N;

    N = nnl_rep;

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
    thrust::device_vector<int> dev_ibead_neighbor_list_rep_vec(dev_ibead_neighbor_list_rep, dev_ibead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep_vec(dev_jbead_neighbor_list_rep, dev_jbead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep_vec(dev_itype_neighbor_list_rep, dev_itype_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep_vec(dev_jtype_neighbor_list_rep, dev_jtype_neighbor_list_rep+N);

    // Create device value vector
    thrust::device_vector<int> dev_value_vec(dev_value_int, dev_value_int+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_rep_vec(N);
    thrust::device_vector<int> dev_jbead_pair_list_rep_vec(N);
    thrust::device_vector<int> dev_itype_pair_list_rep_vec(N);
    thrust::device_vector<int> dev_jtype_pair_list_rep_vec(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep_vec.begin(), dev_jbead_neighbor_list_rep_vec.begin(), dev_itype_neighbor_list_rep_vec.begin(), dev_jtype_neighbor_list_rep_vec.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_rep_vec.end(), dev_jbead_neighbor_list_rep_vec.end(), dev_itype_neighbor_list_rep_vec.end(), dev_jtype_neighbor_list_rep_vec.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_rep_vec.begin(), dev_jbead_pair_list_rep_vec.begin(), dev_itype_pair_list_rep_vec.begin(),
                                            dev_jtype_pair_list_rep_vec.begin()));

    nil_rep = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value_vec.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    HostZipIterator host_result_begin(thrust::make_tuple(dev_ibead_pair_list_rep, dev_jbead_pair_list_rep, dev_itype_pair_list_rep, dev_jtype_pair_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_rep_vec.begin() + nil_rep, dev_jbead_pair_list_rep_vec.begin() + nil_rep,
                                                dev_itype_pair_list_rep_vec.begin() + nil_rep, dev_jtype_pair_list_rep_vec.begin() + nil_rep));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nil_rep--;
}

void update_pair_list_RL(){
  // Declare N
	int N;

  host_to_device(1);
	
	// Set N
	N = nnl_att+1;
	
	// Calculate binary list for att
	calculate_array_native_pl(boxl, N);
  CudaCheckError();
    
  nil_att = compact_native_pl(N)-1;
  CudaCheckError();

  assert(nil_att >= 0);
	
	
	/**********************************
	 *								  *
	 * End of Attractive Calculations *
	 *								  *
	 **********************************/
	
	
	// Set N
	N = nnl_rep+1;
	
	// Calculate binary list for rep
	calculate_array_non_native_pl(boxl, N);
  CudaCheckError();
    
  nil_rep = compact_non_native_pl(N)-1;
  CudaCheckError();

  assert(nil_rep >= 0);
}

void calculate_array_native_pl(int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);

  assert(threads > 0);
  assert(blocks > 0);
	
	// Compute binary array
	array_native_pl_kernel<<<blocks, threads>>>(dev_ibead_neighbor_list_att, dev_jbead_neighbor_list_att, dev_itype_neighbor_list_att, 
                                          dev_jtype_neighbor_list_att, dev_unc_pos, dev_nl_lj_nat_pdb_dist, dev_value_int, boxl, N);

  CudaCheckError();
  // Sync device
  cudaDeviceSynchronize();
}

__global__ void array_native_pl_kernel(int *dev_ibead_neighbor_list_att, int *dev_jbead_neighbor_list_att, int *dev_itype_neighbor_list_att, 
                                      int *dev_jtype_neighbor_list_att, double3 *dev_unc_pos, double *dev_nl_lj_nat_pdb_dist, 
                                      int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i < N){
    double dx, dy, dz;
    double d2;
    int ibead, jbead, itype, jtype;
    double rcut, rcut2;

    // record sigma for ibead and jbead
    ibead = dev_ibead_neighbor_list_att[i];

    jbead = dev_jbead_neighbor_list_att[i];

    // record type of bead for ibead and jbead
    itype = dev_itype_neighbor_list_att[i];

    jtype = dev_jtype_neighbor_list_att[i];
    
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
    This is based off of nl_lj_nat_pdb_dist[i], which is the distance 
    from ibead to jbead in the resulting folded structure
    */
    rcut = 2.5*dev_nl_lj_nat_pdb_dist[i];

    // square cutoff distance, since sqrt(d2) is computationally expensive
    rcut2 = rcut*rcut;

    //printf("%f\t%f\t%f\t%d\t%d\n", d2, rcut2, dev_nl_lj_nat_pdb_dist[i], ibead, jbead);

    if(d2 < rcut2){
      dev_value[i] = 1;
    }else{
      dev_value[i] = 0;
    }
  }else if(i == 0){
      dev_value[i] = 1;
  }
}

void calculate_array_non_native_pl(int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);

  assert(threads > 0);
  assert(blocks > 0);
	
	// Compute binary array
	array_non_native_pl_kernel<<<blocks, threads>>>(dev_ibead_neighbor_list_rep, dev_jbead_neighbor_list_rep, dev_itype_neighbor_list_rep, dev_jtype_neighbor_list_rep, 
                                                dev_unc_pos, dev_value_int, boxl, N);
	CudaCheckError();
  // Sync device
  cudaDeviceSynchronize();
}

__global__ void array_non_native_pl_kernel(int *dev_ibead_neighbor_list_rep, int *dev_jbead_neighbor_list_rep, int *dev_itype_neighbor_list_rep, int *dev_jtype_neighbor_list_rep, 
                                        double3 *dev_unc_pos, int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i < N){
    double dx, dy, dz;
    double d2;
    int ibead, jbead, itype, jtype;
    double rcut, rcut2;

    // record sigma for ibead and jbead
    ibead = dev_ibead_neighbor_list_rep[i];
    jbead = dev_jbead_neighbor_list_rep[i];

    // record type of bead for ibead and jbead
    itype = dev_itype_neighbor_list_rep[i];
    jtype = dev_jtype_neighbor_list_rep[i];
    
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
    This is based off of nl_lj_nat_pdb_dist[i], which is the distance 
    from ibead to jbead in the resulting folded structure
    */
    rcut = 2.5*dev_sigma_rep[itype][jtype];

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


int compact_native_pl(int N){
    // Calculate array size
    int size = N * sizeof(int);

    // Perform hierarchical Kogge-Stone scan on dev_value array and store result in dev_output
    hier_ks_scan(dev_value_int, dev_output_int, N, 0);

    // Copy size of compacted array from device to host and store in arrSize
    int arrSize;
    cudaCheck(cudaMemcpy(&arrSize, &dev_output_int[N-1], sizeof(int), cudaMemcpyDeviceToHost)); 

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);
    
    allocate_and_copy<int>(dev_ibead_neighbor_list_att, dev_value_int, dev_output_int, N, arrSize, dev_ibead_pair_list_att);
    
    allocate_and_copy<int>(dev_jbead_neighbor_list_att, dev_value_int, dev_output_int, N, arrSize, dev_jbead_pair_list_att);

    allocate_and_copy<int>(dev_itype_neighbor_list_att, dev_value_int, dev_output_int, N, arrSize, dev_itype_pair_list_att);

    allocate_and_copy<int>(dev_jtype_neighbor_list_att, dev_value_int, dev_output_int, N, arrSize, dev_jtype_pair_list_att);

    allocate_and_copy<double>(dev_nl_lj_nat_pdb_dist, dev_value_int, dev_output_int, N, arrSize, dev_pl_lj_nat_pdb_dist);

    allocate_and_copy<double>(dev_nl_lj_nat_pdb_dist2, dev_value_int, dev_output_int, N, arrSize, dev_pl_lj_nat_pdb_dist2);

    allocate_and_copy<double>(dev_nl_lj_nat_pdb_dist6, dev_value_int, dev_output_int, N, arrSize, dev_pl_lj_nat_pdb_dist6);

    allocate_and_copy<double>(dev_nl_lj_nat_pdb_dist12, dev_value_int, dev_output_int, N, arrSize, dev_pl_lj_nat_pdb_dist12);

    return arrSize;
}

int compact_non_native_pl(int N){
    // Calculate array size
    int size = N * sizeof(int);

    // Perform hierarchical Kogge-Stone scan on dev_value array and store result in dev_output
    hier_ks_scan(dev_value_int, dev_output_int, N, 0);

    // Copy size of compacted array from device to host and store in arrSize
    int arrSize;
    cudaCheck(cudaMemcpy(&arrSize, &dev_output_int[N-1], sizeof(int), cudaMemcpyDeviceToHost)); 

    int temp;
    cudaCheck(cudaMemcpy(&temp, &dev_value_int[N-1], sizeof(int), cudaMemcpyDeviceToHost)); 

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    allocate_and_copy<int>(dev_ibead_neighbor_list_rep, dev_value_int, dev_output_int, N, arrSize, dev_ibead_pair_list_rep);

    allocate_and_copy<int>(dev_jbead_neighbor_list_rep, dev_value_int, dev_output_int, N, arrSize, dev_jbead_pair_list_rep);

    allocate_and_copy<int>(dev_itype_neighbor_list_rep, dev_value_int, dev_output_int, N, arrSize, dev_itype_pair_list_rep);

    allocate_and_copy<int>(dev_jtype_neighbor_list_rep, dev_value_int, dev_output_int, N, arrSize, dev_jtype_pair_list_rep);

    return arrSize;
}

template <typename T>
void allocate_and_copy(T *dev_index, int *dev_value, int *dev_output, int N, int arrSize, T *dev_result_index){
    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    copyElements<<<blocks, threads>>>(dev_index, dev_value, dev_output, dev_result_index, N);

    cudaDeviceSynchronize();

    CudaCheckError();
}