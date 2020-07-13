#include <math.h>
#include <cstdlib>
#include "neighbor_list.h"
#include "global.h"
#include <vector_types.h>
#include <stdio.h>
#include <stdlib.h>

#define SECTION_SIZE 1024

__device__ __constant__ double dev_sigma_rep[3][3] = {
	{0.0, 0.0, 0.0},
	{0.0, 3.8, 5.4},
	{0.0, 5.4, 7.0}
};

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

void update_neighbor_list_gpu(){
 	// Declare N
	int N;
	
	// Set N
	N = ncon_att;
	
	// Declare value array
	int *value;
	value = (int *)malloc(N*sizeof(int));
	
	// Calculate binary list for att
	calculate_array_native(ibead_lj_nat, jbead_lj_nat, itype_lj_nat, jtype_lj_nat, unc_pos, lj_nat_pdb_dist, value, boxl, N);

	// Compact ibead_neighbor_list_att
	nnl_att = compact(ibead_lj_nat, value, N, ibead_neighbor_list_att)-1;
	
	// Compact jbead_neighbor_list_att
	compact(jbead_lj_nat, value, N, jbead_neighbor_list_att);
	
	// Compact itype_neighbor_list_att
	compact(itype_lj_nat, value, N, itype_neighbor_list_att);
	
	// Compact jtype_neighbor_list_att
	compact(jtype_lj_nat, value, N, jtype_neighbor_list_att);
	
	// Compact nl_lj_nat_pdb_dist
	compact(lj_nat_pdb_dist, value, N, nl_lj_nat_pdb_dist);
	
	// Compact nl_lj_nat_pdb_dist2
	compact(lj_nat_pdb_dist2, value, N, nl_lj_nat_pdb_dist2);
	
	// Compact nl_lj_nat_pdb_dist6
	compact(lj_nat_pdb_dist6, value, N, nl_lj_nat_pdb_dist6);
	
	// Compact nl_lj_nat_pdb_dist12
	compact(lj_nat_pdb_dist12, value, N, nl_lj_nat_pdb_dist12);
	
	// Free value memory
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

	// Compact ibead_neighbor_list_rep
	nnl_rep = compact(ibead_lj_non_nat, value, N, ibead_neighbor_list_rep)-1;
	
	// Compact jbead_neighbor_list_rep
	compact(jbead_lj_non_nat, value, N, jbead_neighbor_list_rep);
	
	// Compact itype_neighbor_list_rep
	compact(itype_lj_non_nat, value, N, itype_neighbor_list_rep);
	
	// Compact jtype_neighbor_list_rep
	compact(itype_lj_non_nat, value, N, itype_neighbor_list_rep);

    free(value);
}

void calculate_array_native(int *ibead_lj_nat, int *jbead_lj_nat, int *itype_lj_nat, int *jtype_lj_nat, float3 *unc_pos, double *lj_nat_pdb_dist, 
                            int *value, int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_float3 = (nbead+1)*sizeof(float3);
	
	// Declare device pointers
	int *dev_ibead_lj_nat;
	int *dev_jbead_lj_nat;
	int *dev_itype_lj_nat;
	int *dev_jtype_lj_nat;
	float3 *dev_unc_pos;
	double *dev_lj_nat_pdb_dist; 
	int *dev_value;
	
	// Allocate device arrays
	cudaMalloc((void **)&dev_ibead_lj_nat, size_int);	
	cudaMalloc((void **)&dev_jbead_lj_nat, size_int);
	cudaMalloc((void **)&dev_itype_lj_nat, size_int);
	cudaMalloc((void **)&dev_jtype_lj_nat, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_float3);
	cudaMalloc((void **)&dev_lj_nat_pdb_dist, size_double);
	cudaMalloc((void **)&dev_value, size_int);
	
	// Copy host arrays to device arrays
	cudaMemcpy(dev_ibead_lj_nat, ibead_lj_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_lj_nat, jbead_lj_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_lj_nat, itype_lj_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_lj_nat, jtype_lj_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_float3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lj_nat_pdb_dist, lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    //dummy<<<blocks, threads>>>(dev_ibead_lj_nat, dev_jbead_lj_nat, dev_itype_lj_nat, dev_jtype_lj_nat, dev_unc_pos, dev_lj_nat_pdb_dist, dev_value, boxl, N, nbead);
    
    /*
    for(int i = 0; i < N; i++){
        printf("i: %d  j: %d\n",ibead_lj_nat[i],jbead_lj_nat[i]);
        float3 a = unc_pos[ibead_lj_nat[i]];
        printf("%d: x=%f, y=%f, z=%f\n", ibead_lj_nat[i], a.x, a.y, a.z);
        
        a = unc_pos[jbead_lj_nat[i]];
        printf("%d: x=%f, y=%f, z=%f\n", jbead_lj_nat[i], a.x, a.y, a.z);
    }
    
    int *test_ibead_lj_nat = (int *)malloc(size_int);
    int *test_jbead_lj_nat = (int *)malloc(size_int);
    float3 *test_unc_pos = (float3 *)malloc(size_float3);
    
    cudaMemcpy(test_ibead_lj_nat, dev_ibead_lj_nat, size_int, cudaMemcpyDeviceToHost);
	cudaMemcpy(test_jbead_lj_nat, dev_jbead_lj_nat, size_int, cudaMemcpyDeviceToHost);
    cudaMemcpy(test_unc_pos, dev_unc_pos, size_float3, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < N; i++){
        if(i > 0 && i < N){
            if(ibead_lj_nat[i] != test_ibead_lj_nat[i] || jbead_lj_nat[i] != test_jbead_lj_nat[i] || unc_pos[ibead_lj_nat[i]].x != test_unc_pos[test_ibead_lj_nat[i]].x ||
            unc_pos[ibead_lj_nat[i]].y != test_unc_pos[test_ibead_lj_nat[i]].y || unc_pos[ibead_lj_nat[i]].z != test_unc_pos[test_ibead_lj_nat[i]].z || 
            unc_pos[jbead_lj_nat[i]].x != test_unc_pos[test_jbead_lj_nat[i]].x || unc_pos[jbead_lj_nat[i]].y != test_unc_pos[test_jbead_lj_nat[i]].y ||
            unc_pos[jbead_lj_nat[i]].z != test_unc_pos[test_jbead_lj_nat[i]].z){
                printf("i: %d  j: %d\n",ibead_lj_nat[i],jbead_lj_nat[i]);
                float3 a = unc_pos[ibead_lj_nat[i]];
                printf("%d: x=%f, y=%f, z=%f\n", ibead_lj_nat[i], a.x, a.y, a.z);
                
                a = unc_pos[jbead_lj_nat[i]];
                printf("%d: x=%f, y=%f, z=%f\n", jbead_lj_nat[i], a.x, a.y, a.z);
                
                printf("test_i: %d  test_j: %d\n",test_ibead_lj_nat[i],test_jbead_lj_nat[i]);
                a = test_unc_pos[test_ibead_lj_nat[i]];
                printf("test_%d: x=%f, y=%f, z=%f\n", test_ibead_lj_nat[i], a.x, a.y, a.z);
                
                a = test_unc_pos[test_jbead_lj_nat[i]];
                printf("test_%d: x=%f, y=%f, z=%f\n", test_jbead_lj_nat[i], a.x, a.y, a.z);
            }
        }
    }*/
	
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

__global__ void array_native_kernel(int *dev_ibead_lj_nat, int *dev_jbead_lj_nat, int *dev_itype_lj_nat, int *dev_jtype_lj_nat, float3 *dev_unc_pos, double *dev_lj_nat_pdb_dist, 
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

void calculate_array_non_native(int *ibead_lj_non_nat, int *jbead_lj_non_nat, int *itype_lj_non_nat, int *jtype_lj_non_nat, float3 *unc_pos,
                            int *value, int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_float3 = (nbead+1)*sizeof(float3);
	
	// Declare device pointers
	int *dev_ibead_lj_non_nat;
	int *dev_jbead_lj_non_nat;
	int *dev_itype_lj_non_nat;
	int *dev_jtype_lj_non_nat;
	float3 *dev_unc_pos; 
	int *dev_value;
	
	// Allocate device arrays
	cudaMalloc((void **)&dev_ibead_lj_non_nat, size_int);	
	cudaMalloc((void **)&dev_jbead_lj_non_nat, size_int);
	cudaMalloc((void **)&dev_itype_lj_non_nat, size_int);
	cudaMalloc((void **)&dev_jtype_lj_non_nat, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_float3);
	cudaMalloc((void **)&dev_value, size_int);
	
	// Copy host arrays to device arrays
	cudaMemcpy(dev_ibead_lj_non_nat, ibead_lj_non_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_lj_non_nat, jbead_lj_non_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_lj_non_nat, itype_lj_non_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_lj_non_nat, jtype_lj_non_nat, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_float3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_value, value, size_int, cudaMemcpyHostToDevice);
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	// Compute binary array
	array_non_native_kernel<<<blocks, threads>>>(dev_ibead_lj_non_nat, dev_jbead_lj_non_nat, dev_itype_lj_non_nat, dev_jtype_lj_non_nat, 
                                                dev_unc_pos, dev_value, boxl, N);
    /*
    cudaDeviceSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }else{
        printf("Success\n");
        exit(0);
    }*/
	
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
                                        float3 *dev_unc_pos, int *dev_value, int boxl, int N){
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

/*
__global__ void dummy(int *dev_ibead_lj_nat, int *dev_jbead_lj_nat, int *dev_itype_lj_nat, int *dev_jtype_lj_nat, float3 *dev_unc_pos, double *dev_lj_nat_pdb_dist, 
                            int *&dev_value, int boxl, int N, int nbead){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i < N){
    double dx, dy, dz;
    double d2;
    int ibead, jbead, itype, jtype;
    double rcut, rcut2;

    if(i >= N){
        printf("%d\n", i);
    }

    // record sigma for ibead and jbead
    ibead = dev_ibead_lj_nat[i];
    //printf("dev_ibead_lj_nat[%d]", i);
    
    jbead = dev_jbead_lj_nat[i];
    //printf("dev_jbead_lj_nat[%d]", i);
    
    // record type of bead for ibead and jbead
    itype = dev_itype_lj_nat[i];
    //printf("dev_itype_lj_nat[%d]", i);
    
    jtype = dev_jtype_lj_nat[i];
    //printf("dev_jtype_lj_nat[%d]", i);
    
    
    if(ibead > nbead+1){
        printf("ibead: %d\n", i);
    }else if(jbead > nbead+1){
        printf("jbead: %d\n", i);
    }
    
    
    // calculate distance in x, y, and z for ibead and jbead
    dx = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;
    printf("dev_unc_pos[%d].x - dev_unc_pos[%d].x", jbead, ibead);
    
    dy = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;
    printf("dev_unc_pos[%d].y - dev_unc_pos[%d].y", jbead, ibead);

    dz = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;
    printf("dev_unc_pos[%d].z - dev_unc_pos[%d].z", jbead, ibead);

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

    rcut = 3.2*dev_lj_nat_pdb_dist[i];
    printf("dev_lj_nat_pdb_dist[%d]", i);

    // square cutoff distance, since sqrt(d2) is computationally expensive
    rcut2 = rcut*rcut;

    if(d2 < rcut2){
      dev_value[i] = 1;
    }else{
      dev_value[i] = 0;
    }
    
  }
}
*/