#include "GPUvars.h"
#include "global.h"

// bonded info
int* dev_ibead_bnd;
int* dev_jbead_bnd;
double* dev_pdb_dist; // pdb bond distances


// angular info
int* dev_ibead_ang;
int* dev_jbead_ang;
int* dev_kbead_ang;
double* dev_pdb_ang;


__device__ double dev_coeff_att[3][3] = {
    {0.0, 0.0, 0.0},
	{0.0, 0.7, 0.8},
	{0.0, 0.8, 1.0}
};

__device__ double dev_coeff_rep[3][3] = {
    {0.0, 0.0, 0.0},
	{0.0, 1.0, 1.0},
	{0.0, 1.0, 1.0}
};

__device__ double dev_force_coeff_att[3][3] = {
    {0.0,       0.0,       0.0},
	{0.0, -12.0*1.0, -12.0*0.8},
	{0.0, -12.0*0.8, -12.0*0.7}
};

__device__ double dev_force_coeff_rep[3][3] = {
    {0.0,       0.0,       0.0},
	{0.0,  -6.0*1.0,  -6.0*1.0},
	{0.0,  -6.0*1.0,  -6.0*1.0}
};

__device__ double dev_sigma_rep[3][3] = {
    {0.0, 0.0, 0.0},
	{0.0, 3.8, 5.4},
	{0.0, 5.4, 7.0}
};


__device__ double dev_rcut_nat[3][3] = {
    { 0.0,  0.0,  0.0},
    { 0.0,  8.0, 11.0},
    { 0.0, 11.0, 14.0}
};

int* dev_ibead_lj_nat;
int* dev_jbead_lj_nat;
int* dev_itype_lj_nat;
int* dev_jtype_lj_nat;

double* dev_lj_nat_pdb_dist;

int* dev_ibead_lj_non_nat;
int* dev_jbead_lj_non_nat;
int* dev_itype_lj_non_nat;
int* dev_jtype_lj_non_nat;

// neighbor / cell list
int* dev_ibead_neighbor_list_att;
int* dev_jbead_neighbor_list_att;
int* dev_itype_neighbor_list_att;
int* dev_jtype_neighbor_list_att;

double* dev_nl_lj_nat_pdb_dist;

int* dev_ibead_neighbor_list_rep;
int* dev_jbead_neighbor_list_rep;
int* dev_itype_neighbor_list_rep;
int* dev_jtype_neighbor_list_rep;

// pair list
int* dev_ibead_pair_list_att;
int* dev_jbead_pair_list_att;
int* dev_itype_pair_list_att;
int* dev_jtype_pair_list_att;

double* dev_pl_lj_nat_pdb_dist;

int* dev_ibead_pair_list_rep;
int* dev_jbead_pair_list_rep;
int* dev_itype_pair_list_rep;
int* dev_jtype_pair_list_rep;

// coordinates and associated params

int nbead;
double3* dev_pos;
double3* dev_unc_pos; // uncorrected positions
double3* dev_vel;
double3* dev_force;

// native info

int* dev_rna_base; // array which indicates whether or not a bead is a base
int* dev_rna_phosphate;

int *dev_value;

void allocate_gpu(){
    int N;
    int size_int;
    int size_double;
    int size_double3;
    
    if(usegpu_nl || usegpu_pl || usegpu_vdw_energy || usegpu_ss_ang_energy || usegpu_fene_energy || usegpu_vdw_force || usegpu_ss_ang_force || usegpu_fene_force){
        N = nbead+1;
        size_int = N*sizeof(int);

        cudaMalloc((void **)&dev_value, size_int);

        if(usegpu_nl){
            if(!strcmp(nl_algorithm,"thrust")){

            }else if(!strcmp(nl_algorithm,"RL")){

            }else if(!strcmp(nl_algorithm,"CL")){

            }
        }

        if(usegpu_pl){
            if(!strcmp(pl_algorithm,"thrust")){

            }else if(!strcmp(pl_algorithm,"RL")){

            }else if(!strcmp(pl_algorithm,"CL")){

            }
        }

        if(usegpu_vdw_energy){
            N = nbead + 1;
	
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead + 1)*sizeof(double3);

            cudaMalloc((void **)&dev_ibead_pair_list_att, size_int);
            cudaMalloc((void **)&dev_jbead_pair_list_att, size_int);
            cudaMalloc((void **)&dev_itype_pair_list_att, size_int);
            cudaMalloc((void **)&dev_jtype_pair_list_att, size_int);
            cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist6, size_double);
            cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist12, size_double);
            
            cudaMalloc((void **)&dev_unc_pos, size_double3);
            
            cudaMalloc((void **)&dev_result, size_double);
        }


        if(usegpu_fene_energy){
            N = nbnd+1;
            cudaMalloc((void **)&dev_ibead_bnd, size_int);
            cudaMalloc((void **)&dev_jbead_bnd, size_int);
            cudaMalloc((void **)&dev_unc_pos, size_double3);
            cudaMalloc((void **)&dev_pdb_dist, size_double);
            cudaMalloc((void **)&dev_result, size_double);

            
        }

        if(usegpu_soft_sphere_angular_energy){

        }

        if(usegpu_vdw_force){

        }

        if(usegpu_fene_force){

        }
        
        if(usegpu_angular_force){

        }

        if(usegpu_soft_sphere_angular_force){

        }
    }
}

void host_to_device(){
    
}

void device_to_host(){

}