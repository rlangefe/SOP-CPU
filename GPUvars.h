#ifndef GPUVARS_H
#define GPUVARS_H

#include "global.h"

// General
extern double3 *dev_unc_pos;
extern double *dev_value_double;
extern double *dev_output_double;
extern int *dev_value_int;
extern int *dev_output_int;

// Neighbor List              
// Native
extern int *dev_ibead_lj_nat;	
extern int *dev_jbead_lj_nat;
extern int *dev_itype_lj_nat;
extern int *dev_jtype_lj_nat;
extern double *dev_lj_nat_pdb_dist;
extern double *dev_lj_nat_pdb_dist2;
extern double *dev_lj_nat_pdb_dist6;
extern double *dev_lj_nat_pdb_dist12;

// Non-Native
extern int *dev_ibead_lj_non_nat;	
extern int *dev_jbead_lj_non_nat;
extern int *dev_itype_lj_non_nat;
extern int *dev_jtype_lj_non_nat;

// Both Neighbor List and Pair List
// Native
extern int *dev_ibead_neighbor_list_att;	
extern int *dev_jbead_neighbor_list_att;
extern int *dev_itype_neighbor_list_att;
extern int *dev_jtype_neighbor_list_att;
extern double *dev_nl_lj_nat_pdb_dist;
extern double *dev_nl_lj_nat_pdb_dist2;
extern double *dev_nl_lj_nat_pdb_dist6;
extern double *dev_nl_lj_nat_pdb_dist12;

// Non-Native
extern int *dev_ibead_neighbor_list_rep;	
extern int *dev_jbead_neighbor_list_rep;
extern int *dev_itype_neighbor_list_rep;
extern int *dev_jtype_neighbor_list_rep;


// Pair List and VDW Energy and VDW Forces
// Native
extern int *dev_ibead_pair_list_att;	
extern int *dev_jbead_pair_list_att;
extern int *dev_itype_pair_list_att;
extern int *dev_jtype_pair_list_att;
extern double *dev_pl_lj_nat_pdb_dist;
extern double *dev_pl_lj_nat_pdb_dist2;
extern double *dev_pl_lj_nat_pdb_dist6;
extern double *dev_pl_lj_nat_pdb_dist12;

// Non-Native
extern int *dev_ibead_pair_list_rep;	
extern int *dev_jbead_pair_list_rep;
extern int *dev_itype_pair_list_rep;
extern int *dev_jtype_pair_list_rep;

// Fene Energy
extern int *dev_ibead_bnd;
extern int *dev_jbead_bnd;
extern double *dev_pdb_dist;

// Soft Sphere Angular Energy
extern int *dev_ibead_ang;
extern int *dev_kbead_ang;

extern double3 *dev_force;

void allocate_gpu();
void host_to_device(int op);
void device_to_host(int op);
void host_collect();
void print_op(int op, int val);

#endif