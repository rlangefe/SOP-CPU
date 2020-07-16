#ifndef PAIR_LIST_H
#define PAIR_LIST_H

#include "global.h"

void update_pair_list();

void update_pair_list_thrust();
void update_pair_list_RL();
void update_pair_list_CL();

void compact_native_pl_thrust(int *value);
void compact_non_native_pl_thrust(int *value);

void compact_native_pl_CL(int *value);
void compact_non_native_pl_CL(int *value);

void calculate_array_native_pl(int *ibead_neighbor_list_att, int *jbead_neighbor_list_att, int *itype_neighbor_list_att, int *jtype_neighbor_list_att, double3 *unc_pos, double *nl_lj_nat_pdb_dist, int *value, int boxl, int N);
__global__ void array_native_pl_kernel(int *dev_ibead_neighbor_list_att, int *dev_jbead_neighbor_list_att, int *dev_itype_neighbor_list_att, int *dev_jtype_neighbor_list_att, double3 *dev_unc_pos, double *dev_nl_lj_nat_pdb_dist, int *dev_value, int boxl, int N);

void calculate_array_non_native_pl(int *ibead_neighbor_list_att, int *jbead_neighbor_list_att, int *itype_neighbor_list_att, int *jtype_neighbor_list_att, double3 *unc_pos, int *value, int boxl, int N);
__global__ void array_non_native_pl_kernel(int *dev_ibead_neighbor_list_rep, int *dev_jbead_neighbor_list_rep, int *dev_itype_neighbor_list_rep, int *dev_jtype_neighbor_list_rep, double3 *dev_unc_pos, int *dev_value, int boxl, int N);

int compact_non_native_pl(int *ibead_neighbor_list_rep, int *jbead_neighbor_list_rep, int *itype_neighbor_list_rep, int *jtype_neighbor_list_rep, int *value, int N, 
                    int *&ibead_pair_list_rep, int *&jbead_pair_list_rep, int *&itype_pair_list_rep, int *&jtype_pair_list_rep);

int compact_native_pl(int *ibead_neighbor_list_att, int *jbead_neighbor_list_att, int *itype_neighbor_list_att, int *jtype_neighbor_list_att, double *nl_lj_nat_pdb_dist,
                    double *nl_lj_nat_pdb_dist2, double *nl_lj_nat_pdb_dist6, double *nl_lj_nat_pdb_dist12, int *value, int N, 
                    int *&ibead_pair_list_att, int *&jbead_pair_list_att, int *&itype_pair_list_att,
                    int *&jtype_pair_list_att, double *&pl_lj_nat_pdb_dist, double *&pl_lj_nat_pdb_dist2,
                    double *&pl_lj_nat_pdb_dist6, double *&pl_lj_nat_pdb_dist12);

#endif