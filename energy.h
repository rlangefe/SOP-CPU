#ifndef ENERGY_H
#define ENERGY_H

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

#endif /* ENERGY_H */
