#ifndef PLIST_H
#define PList_H

#include "global.h"

void update_pair_list();

void update_pair_list_thrust();
void update_pair_list_RL();
void update_pair_list_CL();

void compact_native_pl_thrust();
void compact_non_native_pl_thrust();

void compact_native_pl_CL();
void compact_non_native_pl_CL();

void calculate_array_native_pl(int boxl, int N);
__global__ void array_native_pl_kernel(int *dev_ibead_neighbor_list_att, int *dev_jbead_neighbor_list_att, int *dev_itype_neighbor_list_att, int *dev_jtype_neighbor_list_att, float3 *dev_unc_pos, float *dev_nl_lj_nat_pdb_dist, int *dev_value, int boxl, int N);

void calculate_array_non_native_pl(int boxl, int N);
__global__ void array_non_native_pl_kernel(int *dev_ibead_neighbor_list_rep, int *dev_jbead_neighbor_list_rep, int *dev_itype_neighbor_list_rep, int *dev_jtype_neighbor_list_rep, float3 *dev_unc_pos, int *dev_value, int boxl, int N);

int compact_non_native_pl(int N);

int compact_native_pl(int N);

template <typename T>
void allocate_and_copy(T *dev_index, int *dev_value, int *dev_output, int N, int arrSize, T *dev_result_index);

#endif