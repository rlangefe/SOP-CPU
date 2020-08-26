#ifndef NLIST_H
#define NLIST_H

#include "global.h"
#include "GPUvars.h"


void update_neighbor_list();
void update_neighbor_list_thrust();
void update_neighbor_list_RL();
void update_neighbor_list_CL();

void compact_native_thrust();
void compact_non_native_thrust();

void compact_native_CL();
void compact_non_native_CL();

__global__ void ksScanAuxExc (int *X, int *Y, int InputSize, int *S);
__global__ void ksScanAuxInc (int *X, int *Y, int InputSize, int *S);
__global__ void ksScanExc (int *X, int *Y, int InputSize);
__global__ void ksScanInc (int *X, int *Y, int InputSize);
__global__ void sumIt (int *Y, int *S, int InputSize);

__global__ void copyElements(int *dev_index, int *dev_value, int *dev_output, int *dev_result, int N);
__global__ void copyElements(float *dev_index, int *dev_value, int *dev_output, float *dev_result, int N);

void calculate_array_native(int boxl, int N);
__global__ void array_native_kernel(int *dev_ibead_lj_nat, int *dev_jbead_lj_nat, int *dev_itype_lj_nat, int *dev_jtype_lj_nat, float3 *dev_unc_pos, float *dev_lj_nat_pdb_dist, int *dev_value, int boxl, int N);

void calculate_array_non_native(int boxl, int N);
__global__ void array_non_native_kernel(int *dev_ibead_lj_non_nat, int *dev_jbead_lj_non_nat, int *dev_itype_lj_non_nat, int *dev_jtype_lj_non_nat, float3 *dev_unc_pos, int *dev_value, int boxl, int N);

void hier_ks_scan(int *dev_X, int *dev_Y, int N, int re);
int compact(int *index, int *value, int N, int *&result);
int compact(float *index, int *value, int N, float *&result);

template <typename T>
void allocate_and_copy(T *dev_index, int *dev_value, int *dev_output, int N, int arrSize, T *dev_result_index);

int compact_non_native(int N);

int compact_native(int N);

//__global__ void dummy(int *dev_ibead_lj_nat, int *dev_jbead_lj_nat, int *dev_itype_lj_nat, int *dev_jtype_lj_nat, float3 *dev_unc_pos, float *dev_lj_nat_pdb_dist, int *&dev_value, int boxl, int N, int nbead);

#endif
