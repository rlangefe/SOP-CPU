#include <math.h>
#include <cstdlib>
#include "neighbor_list.h"
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

void update_neighbor_list_CL(){
    int N;
    int *value;

    N = ncon_att;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_native(ibead_lj_nat, jbead_lj_nat, itype_lj_nat, jtype_lj_nat, unc_pos, lj_nat_pdb_dist, value, boxl, N);

    compact_native_CL(value);

    free(value);

    N = ncon_rep;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_non_native(ibead_lj_non_nat, jbead_lj_non_nat, itype_lj_non_nat, jtype_lj_non_nat, unc_pos, value, boxl, N);

    compact_non_native_CL(value);

    free(value);

}

void compact_native_CL(int *value){
    int N;
    
    N = ncon_att;

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
    thrust::device_vector<int> dev_ibead_lj_nat(ibead_lj_nat, ibead_lj_nat+N);
    thrust::device_vector<int> dev_jbead_lj_nat(jbead_lj_nat, jbead_lj_nat+N);
    thrust::device_vector<int> dev_itype_lj_nat(itype_lj_nat, itype_lj_nat+N);
    thrust::device_vector<int> dev_jtype_lj_nat(jtype_lj_nat, jtype_lj_nat+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist(lj_nat_pdb_dist, lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist2(lj_nat_pdb_dist2, lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist6(lj_nat_pdb_dist6, lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist12(lj_nat_pdb_dist12, lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_att(N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att(N);
    thrust::device_vector<int> dev_itype_neighbor_list_att(N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_lj_nat.begin(), dev_jbead_lj_nat.begin(), dev_itype_lj_nat.begin(), dev_jtype_lj_nat.begin(),
                                            dev_lj_nat_pdb_dist.begin(), dev_lj_nat_pdb_dist2.begin(), dev_lj_nat_pdb_dist6.begin(), dev_lj_nat_pdb_dist12.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_lj_nat.end(), dev_jbead_lj_nat.end(), dev_itype_lj_nat.end(), dev_jtype_lj_nat.end(),
                                            dev_lj_nat_pdb_dist.end(), dev_lj_nat_pdb_dist2.end(), dev_lj_nat_pdb_dist6.end(), dev_lj_nat_pdb_dist12.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_att.begin(), dev_jbead_neighbor_list_att.begin(), dev_itype_neighbor_list_att.begin(),
                                            dev_jtype_neighbor_list_att.begin(), dev_nl_lj_nat_pdb_dist.begin(), dev_nl_lj_nat_pdb_dist2.begin(), 
                                            dev_nl_lj_nat_pdb_dist6.begin(), dev_nl_lj_nat_pdb_dist12.begin()));

    thrust::sort_by_key(dev_value.begin(), dev_value.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value.begin(), dev_value.end(), dev_value.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value.end()-1, dev_value.end(), &arrSize);

    nnl_att = arrSize;

    free(ibead_neighbor_list_att);
    free(jbead_neighbor_list_att);
    free(itype_neighbor_list_att);
    free(jtype_neighbor_list_att);
    free(nl_lj_nat_pdb_dist);
    free(nl_lj_nat_pdb_dist2);
    free(nl_lj_nat_pdb_dist6);
    free(nl_lj_nat_pdb_dist12);

    ibead_neighbor_list_att = (int *)malloc(nnl_att*sizeof(int));
    jbead_neighbor_list_att = (int *)malloc(nnl_att*sizeof(int));
    itype_neighbor_list_att = (int *)malloc(nnl_att*sizeof(int));
    jtype_neighbor_list_att = (int *)malloc(nnl_att*sizeof(int));
    nl_lj_nat_pdb_dist = (double *)malloc(nnl_att*sizeof(double));
    nl_lj_nat_pdb_dist2 = (double *)malloc(nnl_att*sizeof(double));
    nl_lj_nat_pdb_dist6 = (double *)malloc(nnl_att*sizeof(double));
    nl_lj_nat_pdb_dist12 = (double *)malloc(nnl_att*sizeof(double));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att,
                                            jtype_neighbor_list_att, nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_neighbor_list_att.begin() + nnl_att, dev_jbead_neighbor_list_att.begin() + nnl_att,
                                                dev_itype_neighbor_list_att.begin() + nnl_att, dev_jtype_neighbor_list_att.begin() + nnl_att, 
                                                dev_nl_lj_nat_pdb_dist.begin() + nnl_att, dev_nl_lj_nat_pdb_dist2.begin() + nnl_att, 
                                                dev_nl_lj_nat_pdb_dist6.begin() + nnl_att, dev_nl_lj_nat_pdb_dist12.begin() + nnl_att));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nnl_att--;
}

void compact_non_native_CL(int *value){
    int N;

    N = ncon_rep;

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
    thrust::device_vector<int> dev_ibead_lj_non_nat(ibead_lj_non_nat, ibead_lj_non_nat+N);
    thrust::device_vector<int> dev_jbead_lj_non_nat(jbead_lj_non_nat, jbead_lj_non_nat+N);
    thrust::device_vector<int> dev_itype_lj_non_nat(itype_lj_non_nat, itype_lj_non_nat+N);
    thrust::device_vector<int> dev_jtype_lj_non_nat(jtype_lj_non_nat, jtype_lj_non_nat+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_rep(N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep(N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep(N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_lj_non_nat.begin(), dev_jbead_lj_non_nat.begin(), dev_itype_lj_non_nat.begin(), dev_jtype_lj_non_nat.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_lj_non_nat.end(), dev_jbead_lj_non_nat.end(), dev_itype_lj_non_nat.end(), dev_jtype_lj_non_nat.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep.begin(), dev_jbead_neighbor_list_rep.begin(), dev_itype_neighbor_list_rep.begin(),
                                            dev_jtype_neighbor_list_rep.begin()));

    thrust::sort_by_key(dev_value.begin(), dev_value.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value.begin(), dev_value.end(), dev_value.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value.end()-1, dev_value.end(), &arrSize);

    nnl_rep = arrSize;

    free(ibead_neighbor_list_rep);
    free(jbead_neighbor_list_rep);
    free(itype_neighbor_list_rep);
    free(jtype_neighbor_list_rep);

    ibead_neighbor_list_rep = (int *)malloc(nnl_rep*sizeof(int));
    jbead_neighbor_list_rep = (int *)malloc(nnl_rep*sizeof(int));
    itype_neighbor_list_rep = (int *)malloc(nnl_rep*sizeof(int));
    jtype_neighbor_list_rep = (int *)malloc(nnl_rep*sizeof(int));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_neighbor_list_rep.begin() + nnl_rep, dev_jbead_neighbor_list_rep.begin() + nnl_rep,
                                                dev_itype_neighbor_list_rep.begin() + nnl_rep, dev_jtype_neighbor_list_rep.begin() + nnl_rep));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nnl_rep--;
}

void update_neighbor_list() {

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
    int *value;

    N = ncon_att;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_native(ibead_lj_nat, jbead_lj_nat, itype_lj_nat, jtype_lj_nat, unc_pos, lj_nat_pdb_dist, value, boxl, N);

    compact_native_thrust(value);

    free(value);

    N = ncon_rep;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_non_native(ibead_lj_non_nat, jbead_lj_non_nat, itype_lj_non_nat, jtype_lj_non_nat, unc_pos, value, boxl, N);

    compact_non_native_thrust(value);

    free(value);

}

void compact_native_thrust(int *value){
    int N;

    N = ncon_att;

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
    thrust::device_vector<int> dev_ibead_lj_nat(ibead_lj_nat, ibead_lj_nat+N);
    thrust::device_vector<int> dev_jbead_lj_nat(jbead_lj_nat, jbead_lj_nat+N);
    thrust::device_vector<int> dev_itype_lj_nat(itype_lj_nat, itype_lj_nat+N);
    thrust::device_vector<int> dev_jtype_lj_nat(jtype_lj_nat, jtype_lj_nat+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist(lj_nat_pdb_dist, lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist2(lj_nat_pdb_dist2, lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist6(lj_nat_pdb_dist6, lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_lj_nat_pdb_dist12(lj_nat_pdb_dist12, lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_att(N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att(N);
    thrust::device_vector<int> dev_itype_neighbor_list_att(N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6(N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_lj_nat.begin(), dev_jbead_lj_nat.begin(), dev_itype_lj_nat.begin(), dev_jtype_lj_nat.begin(),
                                            dev_lj_nat_pdb_dist.begin(), dev_lj_nat_pdb_dist2.begin(), dev_lj_nat_pdb_dist6.begin(), dev_lj_nat_pdb_dist12.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_lj_nat.end(), dev_jbead_lj_nat.end(), dev_itype_lj_nat.end(), dev_jtype_lj_nat.end(),
                                            dev_lj_nat_pdb_dist.end(), dev_lj_nat_pdb_dist2.end(), dev_lj_nat_pdb_dist6.end(), dev_lj_nat_pdb_dist12.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_att.begin(), dev_jbead_neighbor_list_att.begin(), dev_itype_neighbor_list_att.begin(),
                                            dev_jtype_neighbor_list_att.begin(), dev_nl_lj_nat_pdb_dist.begin(), dev_nl_lj_nat_pdb_dist2.begin(), 
                                            dev_nl_lj_nat_pdb_dist6.begin(), dev_nl_lj_nat_pdb_dist12.begin()));

    nnl_att = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    free(ibead_neighbor_list_att);
    free(jbead_neighbor_list_att);
    free(itype_neighbor_list_att);
    free(jtype_neighbor_list_att);
    free(nl_lj_nat_pdb_dist);
    free(nl_lj_nat_pdb_dist2);
    free(nl_lj_nat_pdb_dist6);
    free(nl_lj_nat_pdb_dist12);

    ibead_neighbor_list_att = (int *)malloc(nnl_att*sizeof(int));
    jbead_neighbor_list_att = (int *)malloc(nnl_att*sizeof(int));
    itype_neighbor_list_att = (int *)malloc(nnl_att*sizeof(int));
    jtype_neighbor_list_att = (int *)malloc(nnl_att*sizeof(int));
    nl_lj_nat_pdb_dist = (double *)malloc(nnl_att*sizeof(double));
    nl_lj_nat_pdb_dist2 = (double *)malloc(nnl_att*sizeof(double));
    nl_lj_nat_pdb_dist6 = (double *)malloc(nnl_att*sizeof(double));
    nl_lj_nat_pdb_dist12 = (double *)malloc(nnl_att*sizeof(double));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att,
                                            jtype_neighbor_list_att, nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_neighbor_list_att.begin() + nnl_att, dev_jbead_neighbor_list_att.begin() + nnl_att,
                                                dev_itype_neighbor_list_att.begin() + nnl_att, dev_jtype_neighbor_list_att.begin() + nnl_att, 
                                                dev_nl_lj_nat_pdb_dist.begin() + nnl_att, dev_nl_lj_nat_pdb_dist2.begin() + nnl_att, 
                                                dev_nl_lj_nat_pdb_dist6.begin() + nnl_att, dev_nl_lj_nat_pdb_dist12.begin() + nnl_att));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nnl_att--;
}

void compact_non_native_thrust(int *value){
    int N;

    N = ncon_rep;

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
    thrust::device_vector<int> dev_ibead_lj_non_nat(ibead_lj_non_nat, ibead_lj_non_nat+N);
    thrust::device_vector<int> dev_jbead_lj_non_nat(jbead_lj_non_nat, jbead_lj_non_nat+N);
    thrust::device_vector<int> dev_itype_lj_non_nat(itype_lj_non_nat, itype_lj_non_nat+N);
    thrust::device_vector<int> dev_jtype_lj_non_nat(jtype_lj_non_nat, jtype_lj_non_nat+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_rep(N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep(N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep(N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_lj_non_nat.begin(), dev_jbead_lj_non_nat.begin(), dev_itype_lj_non_nat.begin(), dev_jtype_lj_non_nat.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_lj_non_nat.end(), dev_jbead_lj_non_nat.end(), dev_itype_lj_non_nat.end(), dev_jtype_lj_non_nat.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep.begin(), dev_jbead_neighbor_list_rep.begin(), dev_itype_neighbor_list_rep.begin(),
                                            dev_jtype_neighbor_list_rep.begin()));

    nnl_rep = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    free(ibead_neighbor_list_rep);
    free(jbead_neighbor_list_rep);
    free(itype_neighbor_list_rep);
    free(jtype_neighbor_list_rep);

    ibead_neighbor_list_rep = (int *)malloc(nnl_rep*sizeof(int));
    jbead_neighbor_list_rep = (int *)malloc(nnl_rep*sizeof(int));
    itype_neighbor_list_rep = (int *)malloc(nnl_rep*sizeof(int));
    jtype_neighbor_list_rep = (int *)malloc(nnl_rep*sizeof(int));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_neighbor_list_rep.begin() + nnl_rep, dev_jbead_neighbor_list_rep.begin() + nnl_rep,
                                                dev_itype_neighbor_list_rep.begin() + nnl_rep, dev_jtype_neighbor_list_rep.begin() + nnl_rep));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nnl_rep--;
}

void update_neighbor_list_RL(){
    // Declare N
	int N;
	
	// Set N
	N = ncon_att;
	
	// Declare value array
	int *value;
	value = (int *)malloc(N*sizeof(int));
	
	// Calculate binary list for att
	calculate_array_native(ibead_lj_nat, jbead_lj_nat, itype_lj_nat, jtype_lj_nat, unc_pos, lj_nat_pdb_dist, value, boxl, N);
    
    nnl_att = compact_native(ibead_lj_nat, jbead_lj_nat, itype_lj_nat, jtype_lj_nat, lj_nat_pdb_dist, lj_nat_pdb_dist2, lj_nat_pdb_dist6, lj_nat_pdb_dist12, value, N, 
                    ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att, jtype_neighbor_list_att, nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist2,
                    nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist12) - 1;
	
	// Free value memory to be reallocated later
	free(value);
	
	
	/**********************************
	 *								  *
	 * End of Attractive Calculations *
	 *								  *
	 **********************************/
	
	// Set N
	N = ncon_rep;
	
	// Declare value array
	value = (int *)malloc(N*sizeof(int));
	
	// Calculate binary list for rep
	calculate_array_non_native(ibead_lj_non_nat, jbead_lj_non_nat, itype_lj_non_nat, jtype_lj_non_nat, unc_pos, value, boxl, N);
    
    nnl_rep = compact_non_native(ibead_lj_non_nat, jbead_lj_non_nat, itype_lj_non_nat, jtype_lj_non_nat, value, N, 
                    ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep) - 1;

    free(value);
}


void calculate_array_native(int *ibead_lj_nat, int *jbead_lj_nat, int *itype_lj_nat, int *jtype_lj_nat, double3 *unc_pos, double *lj_nat_pdb_dist, 
                            int *value, int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Declare device pointers
	int *dev_ibead_lj_nat;
	int *dev_jbead_lj_nat;
	int *dev_itype_lj_nat;
	int *dev_jtype_lj_nat;
	double3 *dev_unc_pos;
	double *dev_lj_nat_pdb_dist; 
	int *dev_value;
	
	// Allocate device arrays
	cudaMalloc((void **)&dev_ibead_lj_nat, size_int);	
	cudaMalloc((void **)&dev_jbead_lj_nat, size_int);
	cudaMalloc((void **)&dev_itype_lj_nat, size_int);
	cudaMalloc((void **)&dev_jtype_lj_nat, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	cudaMalloc((void **)&dev_lj_nat_pdb_dist, size_double);
	cudaMalloc((void **)&dev_value, size_int);
	
	// Copy host arrays to device arrays
	cudaMemcpy(dev_ibead_lj_nat, ibead_lj_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_lj_nat, jbead_lj_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_lj_nat, itype_lj_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_lj_nat, jtype_lj_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lj_nat_pdb_dist, lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	// Compute binary array
	array_native_kernel<<<blocks, threads>>>(dev_ibead_lj_nat, dev_jbead_lj_nat, dev_itype_lj_nat, dev_jtype_lj_nat, dev_unc_pos, dev_lj_nat_pdb_dist, dev_value, boxl, N);

    // Sync device
    cudaDeviceSynchronize();

	// Copy device array to host array
	cudaMemcpy(value, dev_value, size_int, cudaMemcpyDeviceToHost);
	
    cudaDeviceSynchronize();

	// Free GPU memory
	cudaFree(dev_ibead_lj_nat);
	cudaFree(dev_jbead_lj_nat);
	cudaFree(dev_itype_lj_nat);
	cudaFree(dev_jtype_lj_nat);
	cudaFree(dev_unc_pos);
	cudaFree(dev_lj_nat_pdb_dist);
	cudaFree(dev_value);
}

__global__ void array_native_kernel(int *dev_ibead_lj_nat, int *dev_jbead_lj_nat, int *dev_itype_lj_nat, int *dev_jtype_lj_nat, double3 *dev_unc_pos, double *dev_lj_nat_pdb_dist, 
                            int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i <= N){
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

void calculate_array_non_native(int *ibead_lj_non_nat, int *jbead_lj_non_nat, int *itype_lj_non_nat, int *jtype_lj_non_nat, double3 *unc_pos,
                            int *value, int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Declare device pointers
	int *dev_ibead_lj_non_nat;
	int *dev_jbead_lj_non_nat;
	int *dev_itype_lj_non_nat;
	int *dev_jtype_lj_non_nat;
	double3 *dev_unc_pos; 
	int *dev_value;
	
	// Allocate device arrays
	cudaMalloc((void **)&dev_ibead_lj_non_nat, size_int);	
	cudaMalloc((void **)&dev_jbead_lj_non_nat, size_int);
	cudaMalloc((void **)&dev_itype_lj_non_nat, size_int);
	cudaMalloc((void **)&dev_jtype_lj_non_nat, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	cudaMalloc((void **)&dev_value, size_int);
	
	// Copy host arrays to device arrays
	cudaMemcpy(dev_ibead_lj_non_nat, ibead_lj_non_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_lj_non_nat, jbead_lj_non_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_lj_non_nat, itype_lj_non_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_lj_non_nat, jtype_lj_non_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_value, value, size_int, cudaMemcpyHostToDevice);
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	// Compute binary array
	array_non_native_kernel<<<blocks, threads>>>(dev_ibead_lj_non_nat, dev_jbead_lj_non_nat, dev_itype_lj_non_nat, dev_jtype_lj_non_nat, dev_unc_pos, dev_value, boxl, N);
	
    // Sync device
    cudaDeviceSynchronize();

	// Copy device array to host array
	cudaMemcpy(value, dev_value, size_int, cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(dev_ibead_lj_non_nat);
	cudaFree(dev_jbead_lj_non_nat);
	cudaFree(dev_itype_lj_non_nat);
	cudaFree(dev_jtype_lj_non_nat);
	cudaFree(dev_unc_pos);
	cudaFree(dev_value);
}

__global__ void array_non_native_kernel(int *dev_ibead_lj_non_nat, int *dev_jbead_lj_non_nat, int *dev_itype_lj_non_nat, int *dev_jtype_lj_non_nat, 
                                        double3 *dev_unc_pos, int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i <= N){
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

    /* 
    Compute the cutoff distance for the given bead
    This is based off of lj_nat_pdb_dist[i], which is the distance 
    from ibead to jbead in the resulting folded structure
    */
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
 * Function: compact
 * -----------------
 *  Finds points in index with a 1 in value and stores them
 *
 *  index: array of indices to check
 *  value: binary value indicating if the corresponding index value is true (1) or false (0)
 *  N: number of elements in index and value
 *  result: pointer where compacted array is stored
 *
 *  Returns: arrSize, the size of the compacted array
 *           Note: result is modified in-place
 */

int compact(int *index, int *value, int N, int *&result){
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

    // Declare and allocate dev_result array to store compacted indices on device (on GPU)
    int *dev_result;
    cudaMalloc((void**)&dev_result, arrSize*sizeof(int));

    // Declare and allocate dev_index to store indecies (on GPU)
    int *dev_index;
    cudaMalloc((void**)&dev_index, size);

    // Copy indices from host to device
    cudaMemcpy(dev_index, index, size, cudaMemcpyHostToDevice);

    /* Calculate number of threads and blocks to use for copying
     * If N < SECTION_SIZE (max # of threads per block), use N threads per block. Else, use SECTION_SIZE threads per block
     * Divides number of elements in array by SECTION_SIZE and rounds up, ensuring it uses the minimum number of blocks required
     */
    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    // Kernel to copy elements from dev_index to dev_output if their corresponding dev_value is 1
    copyElements<<<blocks, threads>>>(dev_index, dev_value, dev_output, dev_result, N);
    
    // Sync device to ensure GPU computation is finished before proceeding
    cudaDeviceSynchronize();

    // Allocate result array on host
    free(result);
    result = (int *)malloc(arrSize*sizeof(int));

    // Copy dev_result (compacted array of indices in GPU) to result array on host
    cudaMemcpy(result, dev_result, arrSize*sizeof(int), cudaMemcpyDeviceToHost); 
    
    // Free device memory
    cudaFree(dev_result); 
    cudaFree(dev_index);
    cudaFree(dev_value);
    cudaFree(dev_output);

    return arrSize;
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
    int i = blockIdx.x * blockDim.x + threadIdx.x+1;
    if(dev_value[i] && i < N){
        dev_result[dev_output[i]-1] = dev_index[i];
    }
    return;
}

int compact(double *index, int *value, int N, double *&result){
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
    /* 
     * TODO: If the entire array has 1 as the value, an exclusive scan will have N-1 as the last value in the array.
     * However, allocating an array with N-1 entries will not store all N values from the index array.
     * Change code to determine when we need to increment arrSize and when we don't.
     * Options include:
     *  1) Changing the hierarchical scan kernel to determine if the final value in the value array is 1
     *  2) Checking to see if the final value is 1 in the value array
     * Option 2 was selected, but please double-check this approach
     */ 
    int arrSize;
    cudaMemcpy(&arrSize, &dev_output[N-1], sizeof(int), cudaMemcpyDeviceToHost); 

    // Increment arrSize by 1 if needed
    if(value[N-1]){
        arrSize++;
    }

    // Declare and allocate dev_result array to store compacted indices on device (on GPU)
    double *dev_result;
    cudaMalloc((void**)&dev_result, arrSize*sizeof(double));

    // Declare and allocate dev_index to store indecies (on GPU)
    double *dev_index;
    cudaMalloc((void**)&dev_index, N*sizeof(double));

    // Copy indices from host to device
    cudaMemcpy(dev_index, index, N*sizeof(double), cudaMemcpyHostToDevice);

    /* Calculate number of threads and blocks to use for copying
     * If N < SECTION_SIZE (max # of threads per block), use N threads per block. Else, use SECTION_SIZE threads per block
     * Divides number of elements in array by SECTION_SIZE and rounds up, ensuring it uses the minimum number of blocks required
     */
    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    // Kernel to copy elements from dev_index to dev_output if their corresponding dev_value is 1
    copyElements<<<blocks, threads>>>(dev_index, dev_value, dev_output, dev_result, N);
    
    // Sync device to ensure GPU computation is finished before proceeding
    cudaDeviceSynchronize();

    // Allocate result array on host
    free(result);
    result = (double *)malloc(arrSize*sizeof(double));

    // Copy dev_result (compacted array of indices in GPU) to result array on host
    cudaMemcpy(result, dev_result, arrSize*sizeof(double), cudaMemcpyDeviceToHost); 
    
    // Free device memory
    cudaFree(dev_result); 
    cudaFree(dev_index);
    cudaFree(dev_value);
    cudaFree(dev_output);

    return arrSize;
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
    int i = blockIdx.x * blockDim.x + threadIdx.x+1;
    if(dev_value[i] && i < N){
        dev_result[dev_output[i]-1] = dev_index[i];
    }
    return;
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

        return;
    }else{
        int threads = (int)min(N, SECTION_SIZE);
        int blocks = (int)ceil(1.0*N/SECTION_SIZE);

        int *dev_S;
        cudaMalloc((void**)&dev_S, (int)ceil(1.0*N/SECTION_SIZE) * sizeof(int));
        
        ksScanAuxInc<<<blocks, threads>>>(dev_X, dev_Y, N, dev_S);
        cudaDeviceSynchronize();

        hier_ks_scan(dev_S, dev_S, (int)ceil(1.0*N/SECTION_SIZE), 1);
        cudaDeviceSynchronize();
        
        sumIt<<<blocks, threads>>>(dev_Y, dev_S, N);
        cudaDeviceSynchronize();

        cudaFree(dev_S);

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
void allocate_and_copy(T *index, int *dev_value, int *dev_output, int N, int arrSize, T *&result_index){
    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    T *dev_index;
    cudaMalloc((void**)&dev_index, N*sizeof(T));
    cudaMemcpy(dev_index, index, N*sizeof(T), cudaMemcpyHostToDevice);
    T *dev_result_index;
    cudaMalloc((void**)&dev_result_index, arrSize*sizeof(T));

    copyElements<<<blocks, threads>>>(dev_index, dev_value, dev_output, dev_result_index, N);
    cudaDeviceSynchronize();
    free(result_index);
    result_index = (T *)malloc(arrSize*sizeof(T));
    cudaMemcpy(result_index, dev_result_index, arrSize*sizeof(T), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_index);
    cudaFree(dev_result_index);

    cudaDeviceSynchronize();
}

int compact_native(int *ibead_lj_nat, int *jbead_lj_nat, int *itype_lj_nat, int *jtype_lj_nat, double *lj_nat_pdb_dist,
                    double *lj_nat_pdb_dist2, double *lj_nat_pdb_dist6, double *lj_nat_pdb_dist12, int *value, int N, 
                    int *&ibead_neighbor_list_att, int *&jbead_neighbor_list_att, int *&itype_neighbor_list_att,
                    int *&jtype_neighbor_list_att, double *&nl_lj_nat_pdb_dist, double *&nl_lj_nat_pdb_dist2,
                    double *&nl_lj_nat_pdb_dist6, double *&nl_lj_nat_pdb_dist12){
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

    // Increment arrSize by 1 if last value is true (1)
    if(value[N-1]){
        arrSize++;
    }

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    allocate_and_copy<int>(ibead_lj_nat, dev_value, dev_output, N, arrSize, ibead_neighbor_list_att);
    
    allocate_and_copy<int>(jbead_lj_nat, dev_value, dev_output, N, arrSize, jbead_neighbor_list_att);

    allocate_and_copy<int>(itype_lj_nat, dev_value, dev_output, N, arrSize, itype_neighbor_list_att);

    allocate_and_copy<int>(jtype_lj_nat, dev_value, dev_output, N, arrSize, jtype_neighbor_list_att);

    allocate_and_copy<double>(lj_nat_pdb_dist, dev_value, dev_output, N, arrSize, nl_lj_nat_pdb_dist);

    allocate_and_copy<double>(lj_nat_pdb_dist2, dev_value, dev_output, N, arrSize, nl_lj_nat_pdb_dist2);

    allocate_and_copy<double>(lj_nat_pdb_dist6, dev_value, dev_output, N, arrSize, nl_lj_nat_pdb_dist6);

    allocate_and_copy<double>(lj_nat_pdb_dist12, dev_value, dev_output, N, arrSize, nl_lj_nat_pdb_dist12);

    cudaFree(dev_value);
    cudaFree(dev_output);

    return arrSize;
}


int compact_non_native(int *ibead_lj_non_nat, int *jbead_lj_non_nat, int *itype_lj_non_nat, int *jtype_lj_non_nat, int *value, int N, 
                    int *&ibead_neighbor_list_rep, int *&jbead_neighbor_list_rep, int *&itype_neighbor_list_rep, int *&jtype_neighbor_list_rep){
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

    // Increment arrSize by 1 if last value is true (1)
    if(value[N-1]){
        arrSize++;
    }

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    allocate_and_copy<int>(ibead_lj_non_nat, dev_value, dev_output, N, arrSize, ibead_neighbor_list_rep);

    allocate_and_copy<int>(jbead_lj_non_nat, dev_value, dev_output, N, arrSize, jbead_neighbor_list_rep);

    allocate_and_copy<int>(itype_lj_non_nat, dev_value, dev_output, N, arrSize, itype_neighbor_list_rep);

    allocate_and_copy<int>(jtype_lj_non_nat, dev_value, dev_output, N, arrSize, jtype_neighbor_list_rep);

    cudaFree(dev_value);
    cudaFree(dev_output);

    return arrSize;
}