#ifndef ENERGY_H
#define ENERGY_H

#include <curand.h>
#include <curand_kernel.h>

void set_potential();
void energy_eval();
void stacking_energy();
void bond_energy();
void fene_energy();
void angular_energy();
void soft_sphere_angular_energy();
void vdw_energy();

void set_forces();
void clear_forces();
void force_eval();
void random_force();
void stacking_forces();
void bond_forces();
void fene_forces();
void angular_forces();
void soft_sphere_angular_forces();
void vdw_forces();

void vdw_energy_gpu();
void vdw_energy_att_gpu();
__global__ void vdw_energy_att_value_kernel(int *dev_ibead_pair_list_att, int *dev_jbead_pair_list_att, int *dev_itype_pair_list_att, int *dev_jtype_pair_list_att, double *dev_pl_lj_nat_pdb_dist6, double *dev_pl_lj_nat_pdb_dist12, double3 *dev_unc_pos, int N, double boxl, double *dev_result);
void vdw_energy_rep_gpu();
__global__ void vdw_energy_rep_value_kernel(int *dev_ibead_pair_list_rep, int *dev_jbead_pair_list_rep, int *dev_itype_pair_list_rep, int *dev_jtype_pair_list_rep, double3 *dev_unc_pos, int N, double boxl, double *dev_result);
void hier_ks_scan(double *dev_X, double *dev_Y, int N, int re);
__global__ void ksScanAuxExc (double *X, double *Y, int InputSize, double *S);
__global__ void ksScanAuxInc (double *X, double *Y, int InputSize, double *S);
__global__ void ksScanExc (double *X, double *Y, double InputSize);
__global__ void ksScanInc (double *X, double *Y, int InputSize);
__global__ void sumIt (double *Y, double *S, int InputSize);

void vdw_forces_gpu();
void vdw_forces_att_gpu();
__global__ void vdw_forces_att_kernel(int *dev_ibead_pair_list_att, int *dev_jbead_pair_list_att, int *dev_itype_pair_list_att, int *dev_jtype_pair_list_att, double *dev_pl_lj_nat_pdb_dist, double boxl, int N, double3 *dev_unc_pos, double3 *dev_force);
void vdw_forces_rep_gpu();
__global__ void vdw_forces_rep_kernel(int *dev_ibead_pair_list_rep, int *dev_jbead_pair_list_rep, int *dev_itype_pair_list_rep, int *dev_jtype_pair_list_rep, double boxl, int N, double3 *dev_unc_pos, double3 *dev_force);

void fene_energy_gpu();
__global__ void fene_energy_gpu_kernel(int *dev_ibead_bnd, int *dev_jbead_bnd, double3 *dev_unc_pos, double *dev_pdb_dist, int boxl, int N, double dev_R0sq, double *dev_result);

void soft_sphere_angular_energy_gpu();
__global__ void soft_sphere_angular_energy_gpu_kernel(int *dev_ibead_ang, int *dev_kbead_ang, double3 *dev_unc_pos, int boxl, int N, double coeff, double *dev_result);

void soft_sphere_angular_forces_gpu();
__global__ void soft_sphere_angular_forces_kernel(int *dev_ibead_ang, int *dev_kbead_ang, double boxl, int N, double coeff, double3 *dev_unc_pos, double3 *dev_force);

void fene_forces_gpu();
__global__ void fene_forces_kernel(int *dev_ibead_bnd, int *dev_jbead_bnd, double *dev_pdb_dist, double boxl, int N, double dev_R0sq, double dev_k_bnd, double3 *dev_unc_pos, double3 *dev_force);

void random_force_gpu();
__global__ void rand_kernel(int N, double3 *dev_force, curandState *state, double var);

void clear_forces_gpu();
__global__ void clear_forces_kernel(double3 *dev_force, int N);
#endif /* ENERGY_H */
