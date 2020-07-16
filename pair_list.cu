#include "neighbor_list.h"
#include "cell_list.h"
#include "pair_list.h"
#include "global.h"

#include <vector_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#define SECTION_SIZE 1024

__device__ __constant__ double dev_sigma_rep[3][3] = {
  {0.0, 0.0, 0.0},
	{0.0, 3.8, 5.4},
	{0.0, 5.4, 7.0}
};

void update_pair_list() {

  using namespace std;

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
}


void update_pair_list_CL(){
    int N;
    int *value;

    N = nnl_att;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_native_pl(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att, jtype_neighbor_list_att, unc_pos, nl_lj_nat_pdb_dist, value, boxl, N);

    compact_native_pl_CL(value);

    free(value);

    N = nnl_rep;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_non_native_pl(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep, unc_pos, value, boxl, N);

    compact_non_native_pl_CL(value);

    free(value);

}

void compact_native_pl_CL(int *value){
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
    thrust::device_vector<int> dev_ibead_neighbor_list_att(ibead_neighbor_list_att, ibead_neighbor_list_att+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att(jbead_neighbor_list_att, jbead_neighbor_list_att+N);
    thrust::device_vector<int> dev_itype_neighbor_list_att(itype_neighbor_list_att, itype_neighbor_list_att+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att(jtype_neighbor_list_att, jtype_neighbor_list_att+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist(nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2(nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6(nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12(nl_lj_nat_pdb_dist12, nl_lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_att(N);
    thrust::device_vector<int> dev_jbead_pair_list_att(N);
    thrust::device_vector<int> dev_itype_pair_list_att(N);
    thrust::device_vector<int> dev_jtype_pair_list_att(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist2(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist6(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist12(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_att.begin(), dev_jbead_neighbor_list_att.begin(), dev_itype_neighbor_list_att.begin(), dev_jtype_neighbor_list_att.begin(),
                                            dev_nl_lj_nat_pdb_dist.begin(), dev_nl_lj_nat_pdb_dist2.begin(), dev_nl_lj_nat_pdb_dist6.begin(), dev_nl_lj_nat_pdb_dist12.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_att.end(), dev_jbead_neighbor_list_att.end(), dev_itype_neighbor_list_att.end(), dev_jtype_neighbor_list_att.end(),
                                            dev_nl_lj_nat_pdb_dist.end(), dev_nl_lj_nat_pdb_dist2.end(), dev_nl_lj_nat_pdb_dist6.end(), dev_nl_lj_nat_pdb_dist12.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_att.begin(), dev_jbead_pair_list_att.begin(), dev_itype_pair_list_att.begin(),
                                            dev_jtype_pair_list_att.begin(), dev_pl_lj_nat_pdb_dist.begin(), dev_pl_lj_nat_pdb_dist2.begin(), 
                                            dev_pl_lj_nat_pdb_dist6.begin(), dev_pl_lj_nat_pdb_dist12.begin()));

    thrust::sort_by_key(dev_value.begin(), dev_value.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value.begin(), dev_value.end(), dev_value.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value.end()-1, dev_value.end(), &arrSize);

    nil_att = arrSize;

    free(ibead_pair_list_att);
    free(jbead_pair_list_att);
    free(itype_pair_list_att);
    free(jtype_pair_list_att);
    free(pl_lj_nat_pdb_dist);
    free(pl_lj_nat_pdb_dist2);
    free(pl_lj_nat_pdb_dist6);
    free(pl_lj_nat_pdb_dist12);

    ibead_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    jbead_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    itype_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    jtype_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    pl_lj_nat_pdb_dist = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist2 = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist6 = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist12 = (double *)malloc(nil_att*sizeof(double));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_pair_list_att, jbead_pair_list_att, itype_pair_list_att,
                                            jtype_pair_list_att, pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist2, pl_lj_nat_pdb_dist6, pl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_att.begin() + nil_att, dev_jbead_pair_list_att.begin() + nil_att,
                                                dev_itype_pair_list_att.begin() + nil_att, dev_jtype_pair_list_att.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist.begin() + nil_att, dev_pl_lj_nat_pdb_dist2.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist6.begin() + nil_att, dev_pl_lj_nat_pdb_dist12.begin() + nil_att));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nil_att--;
}

void compact_non_native_pl_CL(int *value){
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
    thrust::device_vector<int> dev_ibead_neighbor_list_rep(ibead_neighbor_list_rep, ibead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep(jbead_neighbor_list_rep, jbead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep(itype_neighbor_list_rep, itype_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep(jtype_neighbor_list_rep, jtype_neighbor_list_rep+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_rep(N);
    thrust::device_vector<int> dev_jbead_pair_list_rep(N);
    thrust::device_vector<int> dev_itype_pair_list_rep(N);
    thrust::device_vector<int> dev_jtype_pair_list_rep(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep.begin(), dev_jbead_neighbor_list_rep.begin(), dev_itype_neighbor_list_rep.begin(), dev_jtype_neighbor_list_rep.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_rep.end(), dev_jbead_neighbor_list_rep.end(), dev_itype_neighbor_list_rep.end(), dev_jtype_neighbor_list_rep.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_rep.begin(), dev_jbead_pair_list_rep.begin(), dev_itype_pair_list_rep.begin(),
                                            dev_jtype_pair_list_rep.begin()));

    thrust::sort_by_key(dev_value.begin(), dev_value.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value.begin(), dev_value.end(), dev_value.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value.end()-1, dev_value.end(), &arrSize);

    nil_rep = arrSize;

    free(ibead_pair_list_rep);
    free(jbead_pair_list_rep);
    free(itype_pair_list_rep);
    free(jtype_pair_list_rep);

    ibead_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    jbead_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    itype_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    jtype_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_pair_list_rep, jbead_pair_list_rep, itype_pair_list_rep, jtype_pair_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_rep.begin() + nil_rep, dev_jbead_pair_list_rep.begin() + nil_rep,
                                                dev_itype_pair_list_rep.begin() + nil_rep, dev_jtype_pair_list_rep.begin() + nil_rep));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nil_rep--;
}

void update_pair_list_thrust(){
    int N;
    int *value;

    N = nnl_att;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_native_pl(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att, jtype_neighbor_list_att, unc_pos, nl_lj_nat_pdb_dist, value, boxl, N);

    compact_native_pl_thrust(value);

    free(value);

    N = nnl_rep;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_non_native_pl(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep, unc_pos, value, boxl, N);

    compact_non_native_pl_thrust(value);

    free(value);

}

void compact_native_pl_thrust(int *value){
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
    thrust::device_vector<int> dev_ibead_neighbor_list_att(ibead_neighbor_list_att, ibead_neighbor_list_att+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att(jbead_neighbor_list_att, jbead_neighbor_list_att+N);
    thrust::device_vector<int> dev_itype_neighbor_list_att(itype_neighbor_list_att, itype_neighbor_list_att+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att(jtype_neighbor_list_att, jtype_neighbor_list_att+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist(nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2(nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6(nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12(nl_lj_nat_pdb_dist12, nl_lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_att(N);
    thrust::device_vector<int> dev_jbead_pair_list_att(N);
    thrust::device_vector<int> dev_itype_pair_list_att(N);
    thrust::device_vector<int> dev_jtype_pair_list_att(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist2(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist6(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist12(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_att.begin(), dev_jbead_neighbor_list_att.begin(), dev_itype_neighbor_list_att.begin(), dev_jtype_neighbor_list_att.begin(),
                                            dev_nl_lj_nat_pdb_dist.begin(), dev_nl_lj_nat_pdb_dist2.begin(), dev_nl_lj_nat_pdb_dist6.begin(), dev_nl_lj_nat_pdb_dist12.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_att.end(), dev_jbead_neighbor_list_att.end(), dev_itype_neighbor_list_att.end(), dev_jtype_neighbor_list_att.end(),
                                            dev_nl_lj_nat_pdb_dist.end(), dev_nl_lj_nat_pdb_dist2.end(), dev_nl_lj_nat_pdb_dist6.end(), dev_nl_lj_nat_pdb_dist12.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_att.begin(), dev_jbead_pair_list_att.begin(), dev_itype_pair_list_att.begin(),
                                            dev_jtype_pair_list_att.begin(), dev_pl_lj_nat_pdb_dist.begin(), dev_pl_lj_nat_pdb_dist2.begin(), 
                                            dev_pl_lj_nat_pdb_dist6.begin(), dev_pl_lj_nat_pdb_dist12.begin()));

    nil_att = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    free(ibead_pair_list_att);
    free(jbead_pair_list_att);
    free(itype_pair_list_att);
    free(jtype_pair_list_att);
    free(pl_lj_nat_pdb_dist);
    free(pl_lj_nat_pdb_dist2);
    free(pl_lj_nat_pdb_dist6);
    free(pl_lj_nat_pdb_dist12);

    ibead_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    jbead_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    itype_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    jtype_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    pl_lj_nat_pdb_dist = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist2 = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist6 = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist12 = (double *)malloc(nil_att*sizeof(double));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_pair_list_att, jbead_pair_list_att, itype_pair_list_att,
                                            jtype_pair_list_att, pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist2, pl_lj_nat_pdb_dist6, pl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_att.begin() + nil_att, dev_jbead_pair_list_att.begin() + nil_att,
                                                dev_itype_pair_list_att.begin() + nil_att, dev_jtype_pair_list_att.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist.begin() + nil_att, dev_pl_lj_nat_pdb_dist2.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist6.begin() + nil_att, dev_pl_lj_nat_pdb_dist12.begin() + nil_att));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nil_att--;
}

void compact_non_native_pl_thrust(int *value){
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
    thrust::device_vector<int> dev_ibead_neighbor_list_rep(ibead_neighbor_list_rep, ibead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep(jbead_neighbor_list_rep, jbead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep(itype_neighbor_list_rep, itype_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep(jtype_neighbor_list_rep, jtype_neighbor_list_rep+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_rep(N);
    thrust::device_vector<int> dev_jbead_pair_list_rep(N);
    thrust::device_vector<int> dev_itype_pair_list_rep(N);
    thrust::device_vector<int> dev_jtype_pair_list_rep(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep.begin(), dev_jbead_neighbor_list_rep.begin(), dev_itype_neighbor_list_rep.begin(), dev_jtype_neighbor_list_rep.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_rep.end(), dev_jbead_neighbor_list_rep.end(), dev_itype_neighbor_list_rep.end(), dev_jtype_neighbor_list_rep.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_rep.begin(), dev_jbead_pair_list_rep.begin(), dev_itype_pair_list_rep.begin(),
                                            dev_jtype_pair_list_rep.begin()));

    nil_rep = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    free(ibead_pair_list_rep);
    free(jbead_pair_list_rep);
    free(itype_pair_list_rep);
    free(jtype_pair_list_rep);

    ibead_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    jbead_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    itype_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    jtype_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_pair_list_rep, jbead_pair_list_rep, itype_pair_list_rep, jtype_pair_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_rep.begin() + nil_rep, dev_jbead_pair_list_rep.begin() + nil_rep,
                                                dev_itype_pair_list_rep.begin() + nil_rep, dev_jtype_pair_list_rep.begin() + nil_rep));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nil_rep--;
}

void update_pair_list_RL(){
  // Declare N
	int N;
	
	// Set N
	N = nnl_att+1;
	
	// Declare value array
	int *value;
	value = (int *)malloc(N*sizeof(int));
	
	// Calculate binary list for att
	calculate_array_native_pl(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att, jtype_neighbor_list_att, unc_pos, nl_lj_nat_pdb_dist, value, boxl, N);
    
  nil_att = compact_native_pl(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att, jtype_neighbor_list_att, nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist12, value, N, 
                  ibead_pair_list_att, jbead_pair_list_att, itype_pair_list_att, jtype_pair_list_att, pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist2,
                  pl_lj_nat_pdb_dist6, pl_lj_nat_pdb_dist12) - 1;
	
	// Free value memory
	free(value);
	
	
	/**********************************
	 *								  *
	 * End of Attractive Calculations *
	 *								  *
	 **********************************/
	
	
	// Set N
	N = nnl_rep+1;
	
	// Declare value array
	value = (int *)malloc(N*sizeof(int));
	
	// Calculate binary list for rep
	calculate_array_non_native_pl(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep, unc_pos, value, boxl, N);
    
  nil_rep = compact_non_native_pl(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep, value, N, 
                  ibead_pair_list_rep, jbead_pair_list_rep, itype_pair_list_rep, jtype_pair_list_rep) - 1;

  free(value);
}

void calculate_array_native_pl(int *ibead_neighbor_list_att, int *jbead_neighbor_list_att, int *itype_neighbor_list_att, int *jtype_neighbor_list_att, double3 *unc_pos, double *nl_lj_nat_pdb_dist, 
                            int *value, int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Declare device pointers
	int *dev_ibead_neighbor_list_att;
	int *dev_jbead_neighbor_list_att;
	int *dev_itype_neighbor_list_att;
	int *dev_jtype_neighbor_list_att;
	double3 *dev_unc_pos;
	double *dev_nl_lj_nat_pdb_dist; 
	int *dev_value;
	
	// Allocate device arrays
	cudaMalloc((void **)&dev_ibead_neighbor_list_att, size_int);	
	cudaMalloc((void **)&dev_jbead_neighbor_list_att, size_int);
	cudaMalloc((void **)&dev_itype_neighbor_list_att, size_int);
	cudaMalloc((void **)&dev_jtype_neighbor_list_att, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	cudaMalloc((void **)&dev_nl_lj_nat_pdb_dist, size_double);
	cudaMalloc((void **)&dev_value, size_int);
	
	// Copy host arrays to device arrays
	cudaMemcpy(dev_ibead_neighbor_list_att, ibead_neighbor_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_neighbor_list_att, jbead_neighbor_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_neighbor_list_att, itype_neighbor_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_neighbor_list_att, jtype_neighbor_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	// Compute binary array
	array_native_pl_kernel<<<blocks, threads>>>(dev_ibead_neighbor_list_att, dev_jbead_neighbor_list_att, dev_itype_neighbor_list_att, dev_jtype_neighbor_list_att, dev_unc_pos, dev_nl_lj_nat_pdb_dist, dev_value, boxl, N);

  // Sync device
  cudaDeviceSynchronize();

	// Copy device array to host array
	cudaMemcpy(value, dev_value, size_int, cudaMemcpyDeviceToHost);
	
  cudaDeviceSynchronize();

	// Free GPU memory
	cudaFree(dev_ibead_neighbor_list_att);
	cudaFree(dev_jbead_neighbor_list_att);
	cudaFree(dev_itype_neighbor_list_att);
	cudaFree(dev_jtype_neighbor_list_att);
	cudaFree(dev_unc_pos);
	cudaFree(dev_nl_lj_nat_pdb_dist);
	cudaFree(dev_value);
}

__global__ void array_native_pl_kernel(int *dev_ibead_neighbor_list_att, int *dev_jbead_neighbor_list_att, int *dev_itype_neighbor_list_att, int *dev_jtype_neighbor_list_att, double3 *dev_unc_pos, double *dev_nl_lj_nat_pdb_dist, 
                            int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i <= N){
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

    if(d2 < rcut2){
      dev_value[i] = 1;
    }else{
      dev_value[i] = 0;
    }
  }else if(i == 0){
      dev_value[i] = 1;
  }
}

void calculate_array_non_native_pl(int *ibead_neighbor_list_rep, int *jbead_neighbor_list_rep, int *itype_neighbor_list_rep, int *jtype_neighbor_list_rep, double3 *unc_pos,
                            int *value, int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Declare device pointers
	int *dev_ibead_neighbor_list_rep;
	int *dev_jbead_neighbor_list_rep;
	int *dev_itype_neighbor_list_rep;
	int *dev_jtype_neighbor_list_rep;
	double3 *dev_unc_pos; 
	int *dev_value;
	
	// Allocate device arrays
	cudaMalloc((void **)&dev_ibead_neighbor_list_rep, size_int);	
	cudaMalloc((void **)&dev_jbead_neighbor_list_rep, size_int);
	cudaMalloc((void **)&dev_itype_neighbor_list_rep, size_int);
	cudaMalloc((void **)&dev_jtype_neighbor_list_rep, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	cudaMalloc((void **)&dev_value, size_int);
	
	// Copy host arrays to device arrays
	cudaMemcpy(dev_ibead_neighbor_list_rep, ibead_neighbor_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_neighbor_list_rep, jbead_neighbor_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_neighbor_list_rep, itype_neighbor_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_neighbor_list_rep, jtype_neighbor_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_value, value, size_int, cudaMemcpyHostToDevice);
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	// Compute binary array
	array_non_native_pl_kernel<<<blocks, threads>>>(dev_ibead_neighbor_list_rep, dev_jbead_neighbor_list_rep, dev_itype_neighbor_list_rep, dev_jtype_neighbor_list_rep, 
                                                dev_unc_pos, dev_value, boxl, N);
	
  // Sync device
  cudaDeviceSynchronize();

	// Copy device array to host array
	cudaMemcpy(value, dev_value, size_int, cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(dev_ibead_neighbor_list_rep);
	cudaFree(dev_jbead_neighbor_list_rep);
	cudaFree(dev_itype_neighbor_list_rep);
	cudaFree(dev_jtype_neighbor_list_rep);
	cudaFree(dev_unc_pos);
	cudaFree(dev_value);
}

__global__ void array_non_native_pl_kernel(int *dev_ibead_neighbor_list_rep, int *dev_jbead_neighbor_list_rep, int *dev_itype_neighbor_list_rep, int *dev_jtype_neighbor_list_rep, 
                                        double3 *dev_unc_pos, int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i <= N){
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


int compact_native_pl(int *ibead_neighbor_list_att, int *jbead_neighbor_list_att, int *itype_neighbor_list_att, int *jtype_neighbor_list_att, double *nl_lj_nat_pdb_dist,
                    double *nl_lj_nat_pdb_dist2, double *nl_lj_nat_pdb_dist6, double *nl_lj_nat_pdb_dist12, int *value, int N, 
                    int *&ibead_pair_list_att, int *&jbead_pair_list_att, int *&itype_pair_list_att,
                    int *&jtype_pair_list_att, double *&pl_lj_nat_pdb_dist, double *&pl_lj_nat_pdb_dist2,
                    double *&pl_lj_nat_pdb_dist6, double *&pl_lj_nat_pdb_dist12){
    // Declare pointers for dev_output and dev_value arrays
    int *dev_output;
    int *dev_value;

    // Calculate array size
    int size = N * sizeof(int);

    // Allocate dev_value and dev_output arrays
    cudaMalloc((void**)&dev_value, size);
    cudaMalloc((void**)&dev_output, size);
 
    // Copy data from value array to device (dev_value)
    cudaMemcpy(dev_value, value, size, cudaMemcpyHostToDevice);

    // Perform hierarchical Kogge-Stone scan on dev_value array and store result in dev_output
    hier_ks_scan(dev_value, dev_output, N, 0);

    // Copy size of compacted array from device to host and store in arrSize
    int arrSize;
    cudaMemcpy(&arrSize, &dev_output[N-1], sizeof(int), cudaMemcpyDeviceToHost); 

    // Increment arrSize by 1 if needed
    if(value[N-1]){
        arrSize++;
    }

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    allocate_and_copy<int>(ibead_neighbor_list_att, dev_value, dev_output, N, arrSize, ibead_pair_list_att);
    
    allocate_and_copy<int>(jbead_neighbor_list_att, dev_value, dev_output, N, arrSize, jbead_pair_list_att);

    allocate_and_copy<int>(itype_neighbor_list_att, dev_value, dev_output, N, arrSize, itype_pair_list_att);

    allocate_and_copy<int>(jtype_neighbor_list_att, dev_value, dev_output, N, arrSize, jtype_pair_list_att);

    allocate_and_copy<double>(nl_lj_nat_pdb_dist, dev_value, dev_output, N, arrSize, pl_lj_nat_pdb_dist);

    allocate_and_copy<double>(nl_lj_nat_pdb_dist2, dev_value, dev_output, N, arrSize, pl_lj_nat_pdb_dist2);

    allocate_and_copy<double>(nl_lj_nat_pdb_dist6, dev_value, dev_output, N, arrSize, pl_lj_nat_pdb_dist6);

    allocate_and_copy<double>(nl_lj_nat_pdb_dist12, dev_value, dev_output, N, arrSize, pl_lj_nat_pdb_dist12);

    cudaFree(dev_value);
    cudaFree(dev_output);

    return arrSize;
}


int compact_non_native_pl(int *ibead_neighbor_list_rep, int *jbead_neighbor_list_rep, int *itype_neighbor_list_rep, int *jtype_neighbor_list_rep, int *value, int N, 
                    int *&ibead_pair_list_rep, int *&jbead_pair_list_rep, int *&itype_pair_list_rep, int *&jtype_pair_list_rep){
    // Declare pointers for dev_output and dev_value arrays
    int *dev_output;
    int *dev_value;

    // Calculate array size
    int size = N * sizeof(int);

    // Allocate dev_value and dev_output arrays
    cudaMalloc((void**)&dev_value, size);
    cudaMalloc((void**)&dev_output, size);
 
    // Copy data from value array to device (dev_value)
    cudaMemcpy(dev_value, value, size, cudaMemcpyHostToDevice);

    // Perform hierarchical Kogge-Stone scan on dev_value array and store result in dev_output
    hier_ks_scan(dev_value, dev_output, N, 0);

    // Copy size of compacted array from device to host and store in arrSize
    int arrSize;
    cudaMemcpy(&arrSize, &dev_output[N-1], sizeof(int), cudaMemcpyDeviceToHost); 

    // Increment arrSize by 1 if needed
    if(value[N-1]){
        arrSize++;
    }

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    allocate_and_copy<int>(ibead_neighbor_list_rep, dev_value, dev_output, N, arrSize, ibead_pair_list_rep);

    allocate_and_copy<int>(jbead_neighbor_list_rep, dev_value, dev_output, N, arrSize, jbead_pair_list_rep);

    allocate_and_copy<int>(itype_neighbor_list_rep, dev_value, dev_output, N, arrSize, itype_pair_list_rep);

    allocate_and_copy<int>(jtype_neighbor_list_rep, dev_value, dev_output, N, arrSize, jtype_pair_list_rep);

    cudaFree(dev_value);
    cudaFree(dev_output);

    return arrSize;
}