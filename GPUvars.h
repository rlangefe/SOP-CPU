#ifndef GPUVARS_H
#define GPUVARS_H

#include "global.h"

// bonded info
extern int* dev_ibead_bnd;
extern int* dev_jbead_bnd;
extern double* dev_pdb_dist;

// angular info

extern int* dev_ibead_ang;
extern int* dev_jbead_ang;
extern int* dev_kbead_ang;
extern double* dev_pdb_ang;

// pair list
extern double dev_coeff_att[][3];
extern double dev_coeff_rep[][3];
extern double dev_force_coeff_att[][3];
extern double dev_force_coeff_rep[][3];
extern double dev_sigma_rep[][3];

extern double dev_rcut_nat[][3];
extern int* dev_ibead_lj_nat;
extern int* dev_jbead_lj_nat;
extern int* dev_itype_lj_nat;
extern int* dev_jtype_lj_nat;
extern double* dev_lj_nat_pdb_dist;
extern int* dev_ibead_lj_non_nat;
extern int* dev_jbead_lj_non_nat;
extern int* dev_itype_lj_non_nat;
extern int* dev_jtype_lj_non_nat;

// neighbor / cell list
extern int* dev_ibead_neighbor_list_att;
extern int* dev_jbead_neighbor_list_att;
extern int* dev_itype_neighbor_list_att;
extern int* dev_jtype_neighbor_list_att;

extern double* dev_nl_lj_nat_pdb_dist;
extern double* dev_nl_lj_nat_pdb_dist2;
extern double* dev_nl_lj_nat_pdb_dist6;
extern double* dev_nl_lj_nat_pdb_dist12;

extern int* dev_ibead_neighbor_list_rep;
extern int* dev_jbead_neighbor_list_rep;
extern int* dev_itype_neighbor_list_rep;
extern int* dev_jtype_neighbor_list_rep;

// pair list
extern int* dev_ibead_pair_list_att;
extern int* dev_jbead_pair_list_att;
extern int* dev_itype_pair_list_att;
extern int* dev_jtype_pair_list_att;

extern double* dev_pl_lj_nat_pdb_dist;

extern int* dev_ibead_pair_list_rep;
extern int* dev_jbead_pair_list_rep;
extern int* dev_itype_pair_list_rep;
extern int* dev_jtype_pair_list_rep;

// coordinates and associated params

extern double3* dev_pos;
extern double3* dev_unc_pos;
extern double3* dev_vel;
extern double3* dev_force;
extern double3* dev_natpos; // native position vectors


// native info

extern int* dev_rna_base;
extern int* dev_rna_phosphate;

extern int* dev_value;

void allocate_gpu();

#endif