#include "GPUvars.h"
#include "global.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define SECTION_SIZE 1024

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define cudaCheck(error) \
  if (error != cudaSuccess) { \
    printf("CUDA Error: %s at %s:%d\n", \
      cudaGetErrorString(error), \
      __FILE__, __LINE__); \
    exit(1); \
              }

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

// General
double3 *dev_unc_pos;
double *dev_value_double;
double *dev_output_double;
int *dev_value_int;
int *dev_output_int;

// Neighbor List              
// Native
int *dev_ibead_lj_nat;	
int *dev_jbead_lj_nat;
int *dev_itype_lj_nat;
int *dev_jtype_lj_nat;
double *dev_lj_nat_pdb_dist;
double *dev_lj_nat_pdb_dist2;
double *dev_lj_nat_pdb_dist6;
double *dev_lj_nat_pdb_dist12;


// Non-Native
int *dev_ibead_lj_non_nat;	
int *dev_jbead_lj_non_nat;
int *dev_itype_lj_non_nat;
int *dev_jtype_lj_non_nat;

// Both Neighbor List and Pair List
// Native
int *dev_ibead_neighbor_list_att;	
int *dev_jbead_neighbor_list_att;
int *dev_itype_neighbor_list_att;
int *dev_jtype_neighbor_list_att;
double *dev_nl_lj_nat_pdb_dist;
double *dev_nl_lj_nat_pdb_dist2;
double *dev_nl_lj_nat_pdb_dist6;
double *dev_nl_lj_nat_pdb_dist12;

// Non-Native
int *dev_ibead_neighbor_list_rep;	
int *dev_jbead_neighbor_list_rep;
int *dev_itype_neighbor_list_rep;
int *dev_jtype_neighbor_list_rep;


// Pair List and VDW Energy and VDW Forces
// Native
int *dev_ibead_pair_list_att;	
int *dev_jbead_pair_list_att;
int *dev_itype_pair_list_att;
int *dev_jtype_pair_list_att;
double *dev_pl_lj_nat_pdb_dist;
double *dev_pl_lj_nat_pdb_dist2;
double *dev_pl_lj_nat_pdb_dist6;
double *dev_pl_lj_nat_pdb_dist12;

// Non-Native
int *dev_ibead_pair_list_rep;	
int *dev_jbead_pair_list_rep;
int *dev_itype_pair_list_rep;
int *dev_jtype_pair_list_rep;

// Fene Energy
int *dev_ibead_bnd;
int *dev_jbead_bnd;
double *dev_pdb_dist;

// Soft Sphere Angular Energy
int *dev_ibead_ang;
int *dev_kbead_ang;

double3 *dev_force;

// Position
double3 *dev_pos;

// Velocity
double3 *dev_vel;

// Position and Velocity
double3 *dev_incr;

// cuRand
curandState *devStates;

void allocate_gpu(){
    int N;
    int size_int;
    int size_double;
    int size_double3;

    // Initial variable locations at 0 (on Host)
    for(int i = 0; i < 30; i++){
        variable_location[i] = 0;
    }

    N = nbead+1;
    size_int = N*sizeof(int);
    size_double = N*sizeof(double);
    size_double3 = (nbead+1)*sizeof(double3);

    if(usegpu_clear_force || usegpu_nl || usegpu_pl || usegpu_vdw_energy || usegpu_ss_ang_energy || usegpu_fene_energy || usegpu_vdw_force || usegpu_ss_ang_force || usegpu_fene_force || usegpu_pos || usegpu_vel || usegpu_rand_force){
        N = (nbead+1)*(nbead+1);
        size_int = N*sizeof(int);
        size_double = N*sizeof(double);
        size_double3 = (nbead+1)*sizeof(double3);
        
        cudaCheck(cudaMalloc((void **)&dev_unc_pos, size_double3));
        cudaCheck(cudaMalloc((void **)&dev_value_double, size_double));
        cudaCheck(cudaMalloc((void **)&dev_output_double, size_double));
        cudaCheck(cudaMalloc((void **)&dev_value_int, size_int));
        cudaCheck(cudaMalloc((void **)&dev_output_int, size_int));

        if(usegpu_nl){
            N = ncon_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);           
            // Native
            cudaCheck(cudaMalloc((void **)&dev_ibead_lj_nat, size_int));
            cudaCheck(cudaMalloc((void **)&dev_jbead_lj_nat, size_int));
            cudaCheck(cudaMalloc((void **)&dev_itype_lj_nat, size_int));
            cudaCheck(cudaMalloc((void **)&dev_jtype_lj_nat, size_int));
            cudaCheck(cudaMalloc((void **)&dev_lj_nat_pdb_dist, size_double));
            cudaCheck(cudaMalloc((void **)&dev_lj_nat_pdb_dist2, size_double));
            cudaCheck(cudaMalloc((void **)&dev_lj_nat_pdb_dist6, size_double));
            cudaCheck(cudaMalloc((void **)&dev_lj_nat_pdb_dist12, size_double));
            
            N = ncon_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            cudaCheck(cudaMalloc((void **)&dev_ibead_lj_non_nat, size_int));	
            cudaCheck(cudaMalloc((void **)&dev_jbead_lj_non_nat, size_int));
            cudaCheck(cudaMalloc((void **)&dev_itype_lj_non_nat, size_int));
            cudaCheck(cudaMalloc((void **)&dev_jtype_lj_non_nat, size_int));
        }

        if(usegpu_nl || usegpu_pl){
            N = ncon_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Results Native
            cudaCheck(cudaMalloc((void **)&dev_ibead_neighbor_list_att, size_int));	
            cudaCheck(cudaMalloc((void **)&dev_jbead_neighbor_list_att, size_int));
            cudaCheck(cudaMalloc((void **)&dev_itype_neighbor_list_att, size_int));
            cudaCheck(cudaMalloc((void **)&dev_jtype_neighbor_list_att, size_int));
            cudaCheck(cudaMalloc((void **)&dev_nl_lj_nat_pdb_dist, size_double));
            cudaCheck(cudaMalloc((void **)&dev_nl_lj_nat_pdb_dist2, size_double));
            cudaCheck(cudaMalloc((void **)&dev_nl_lj_nat_pdb_dist6, size_double));
            cudaCheck(cudaMalloc((void **)&dev_nl_lj_nat_pdb_dist12, size_double));
            
            N = ncon_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3); 

            // Results Non-Native
            cudaCheck(cudaMalloc((void **)&dev_ibead_neighbor_list_rep, size_int));	
            cudaCheck(cudaMalloc((void **)&dev_jbead_neighbor_list_rep, size_int));
            cudaCheck(cudaMalloc((void **)&dev_itype_neighbor_list_rep, size_int));
            cudaCheck(cudaMalloc((void **)&dev_jtype_neighbor_list_rep, size_int));
        }

        if(usegpu_pl){
            if(!strcmp(pl_algorithm,"thrust")){

            }else if(!strcmp(pl_algorithm,"RL")){
                
            }else if(!strcmp(pl_algorithm,"CL")){

            }
        }

        if(usegpu_pl || usegpu_vdw_energy || usegpu_vdw_force){
            N = ncon_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            cudaCheck(cudaMalloc((void **)&dev_ibead_pair_list_att, size_int));	
            cudaCheck(cudaMalloc((void **)&dev_jbead_pair_list_att, size_int));
            cudaCheck(cudaMalloc((void **)&dev_itype_pair_list_att, size_int));
            cudaCheck(cudaMalloc((void **)&dev_jtype_pair_list_att, size_int));
            cudaCheck(cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist, size_double));
            cudaCheck(cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist2, size_double));
            cudaCheck(cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist6, size_double));
            cudaCheck(cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist12, size_double));

            N = ncon_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3); 

            // Non-Native
            cudaCheck(cudaMalloc((void **)&dev_ibead_pair_list_rep, size_int));	
            cudaCheck(cudaMalloc((void **)&dev_jbead_pair_list_rep, size_int));
            cudaCheck(cudaMalloc((void **)&dev_itype_pair_list_rep, size_int));
            cudaCheck(cudaMalloc((void **)&dev_jtype_pair_list_rep, size_int));
        }


        if(usegpu_fene_energy || usegpu_fene_force){
            N = nbnd+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3); 

            cudaCheck(cudaMalloc((void **)&dev_ibead_bnd, size_int));
            cudaCheck(cudaMalloc((void **)&dev_jbead_bnd, size_int));
            cudaCheck(cudaMalloc((void **)&dev_pdb_dist, size_double));
        }

        if(usegpu_ss_ang_energy || usegpu_ss_ang_force){
            N = nang+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3); 

            cudaCheck(cudaMalloc((void **)&dev_ibead_ang, size_int));
            cudaCheck(cudaMalloc((void **)&dev_kbead_ang, size_int));
        }

        if(usegpu_ss_ang_force || usegpu_fene_force || usegpu_vdw_force || usegpu_vel || usegpu_pos || usegpu_rand_force || usegpu_clear_force){
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3); 

            cudaCheck(cudaMalloc((void **)&dev_force, size_double3));
        }

        if(usegpu_pos){
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3); 

            cudaCheck(cudaMalloc((void **)&dev_pos, size_double3));
        }

        if(usegpu_pos || usegpu_vel){
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            cudaCheck(cudaMalloc((void **)&dev_vel, size_double3));
            cudaCheck(cudaMalloc((void **)&dev_incr, size_double3));
        }

        if(usegpu_rand_force){
            printf("Setup Rand\n");
            setup_rng(2718, 0);
            
        }


    }

    CudaCheckError();

    cudaDeviceSynchronize();
}



void host_to_device(int op){
    int N;
    int size_int;
	int size_double;
	int size_double3;

    if(debug){
        printf("%*d: ", 2, op);
    }

    cudaDeviceSynchronize();

    switch(op){
        // Neighbor List
        case 0:
            N = ncon_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[0] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_lj_nat, ibead_lj_nat, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_lj_nat, jbead_lj_nat, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_itype_lj_nat, itype_lj_nat, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jtype_lj_nat, jtype_lj_nat, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_lj_nat_pdb_dist, lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_lj_nat_pdb_dist2, lj_nat_pdb_dist2, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_lj_nat_pdb_dist6, lj_nat_pdb_dist6, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_lj_nat_pdb_dist12, lj_nat_pdb_dist12, size_double, cudaMemcpyHostToDevice));

                variable_location[0] = 1;
            }

            if(variable_location[1] == 0){
                cudaCheck(cudaMemcpy(dev_lj_nat_pdb_dist, lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice));
                
                variable_location[1] = 1;
            }

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            N = ncon_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[3] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_lj_non_nat, ibead_lj_non_nat, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_lj_non_nat, jbead_lj_non_nat, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_itype_lj_non_nat, itype_lj_non_nat, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jtype_lj_non_nat, jtype_lj_non_nat, size_int, cudaMemcpyHostToDevice));

                variable_location[3] = 1;
            }

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            
            // Modified variables on GPU
            variable_location[4] = 1;
            variable_location[5] = 1;
            variable_location[6] = 1;

            break;

        // Pair List
        case 1:
            N = nnl_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[4] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_neighbor_list_att, ibead_neighbor_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_neighbor_list_att, jbead_neighbor_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_itype_neighbor_list_att, itype_neighbor_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jtype_neighbor_list_att, jtype_neighbor_list_att, size_int, cudaMemcpyHostToDevice));

                variable_location[4] = 1;
            }

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            if(variable_location[5] == 0){
                cudaCheck(cudaMemcpy(dev_nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist2, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist6, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_nl_lj_nat_pdb_dist12, nl_lj_nat_pdb_dist12, size_double, cudaMemcpyHostToDevice));

                variable_location[5] = 1;
            }

            N = nnl_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[6] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_neighbor_list_rep, ibead_neighbor_list_rep, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_neighbor_list_rep, jbead_neighbor_list_rep, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_itype_neighbor_list_rep, itype_neighbor_list_rep, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jtype_neighbor_list_rep, jtype_neighbor_list_rep, size_int, cudaMemcpyHostToDevice));

                variable_location[6] = 1;
            }

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            variable_location[7] = 1;
            variable_location[8] = 1;

            break;

        // VDW Energy
        case 2:
            N = nil_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[7] == 0){
                //printf("\tCopy VDW Energy");
                cudaCheck(cudaMemcpy(dev_ibead_pair_list_att, ibead_pair_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_pair_list_att, jbead_pair_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_itype_pair_list_att, itype_pair_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jtype_pair_list_att, jtype_pair_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_pl_lj_nat_pdb_dist2, pl_lj_nat_pdb_dist2, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_pl_lj_nat_pdb_dist6, pl_lj_nat_pdb_dist6, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_pl_lj_nat_pdb_dist12, pl_lj_nat_pdb_dist12, size_double, cudaMemcpyHostToDevice));

                variable_location[7] = 1;
            }
            
            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            N = nil_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[8] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_pair_list_rep, ibead_pair_list_rep, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_pair_list_rep, jbead_pair_list_rep, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_itype_pair_list_rep, itype_pair_list_rep, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jtype_pair_list_rep, jtype_pair_list_rep, size_int, cudaMemcpyHostToDevice));

                variable_location[8] = 1;
            }

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            break;

        // Fene Energy
        case 3:
            N = nbnd+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[9] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_bnd, ibead_bnd, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_bnd, jbead_bnd, size_int, cudaMemcpyHostToDevice));

                variable_location[9] = 1;
            }

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            if(variable_location[10] == 0){
                cudaCheck(cudaMemcpy(dev_pdb_dist, pdb_dist, size_double, cudaMemcpyHostToDevice));

                variable_location[10] = 1;
            }

            break;

        // Soft Sphere Angular Energy
        case 4:
            N = nang+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[11] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_ang, ibead_ang, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_kbead_ang, kbead_ang, size_int, cudaMemcpyHostToDevice));

                variable_location[11] = 1;
            }

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }
            
            break;

        // VDW Forces
        case 5:
            N = nil_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[7] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_pair_list_att, ibead_pair_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_pair_list_att, jbead_pair_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_itype_pair_list_att, itype_pair_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jtype_pair_list_att, jtype_pair_list_att, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_pl_lj_nat_pdb_dist2, pl_lj_nat_pdb_dist2, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_pl_lj_nat_pdb_dist6, pl_lj_nat_pdb_dist6, size_double, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_pl_lj_nat_pdb_dist12, pl_lj_nat_pdb_dist12, size_double, cudaMemcpyHostToDevice));

                variable_location[7] = 1;
            }

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            if(variable_location[12] == 0){
                cudaCheck(cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice));

                variable_location[12] = 1;
            }

            N = nil_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[8] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_pair_list_rep, ibead_pair_list_rep, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_pair_list_rep, jbead_pair_list_rep, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_itype_pair_list_rep, itype_pair_list_rep, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jtype_pair_list_rep, jtype_pair_list_rep, size_int, cudaMemcpyHostToDevice));

                variable_location[8] = 1;
            }

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            if(variable_location[12] == 0){
                cudaCheck(cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice));

                variable_location[12] = 1;
            }

            break;

        // Fene Forces
        case 6:
            N = nbnd+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[9] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_bnd, ibead_bnd, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_bnd, jbead_bnd, size_int, cudaMemcpyHostToDevice));

                variable_location[9] = 1;
            }
            
            if(variable_location[10] == 0){
                cudaCheck(cudaMemcpy(dev_pdb_dist, pdb_dist, size_double, cudaMemcpyHostToDevice));

                variable_location[10] = 1;
            }

            if(variable_location[9] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_bnd, ibead_bnd, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_jbead_bnd, jbead_bnd, size_int, cudaMemcpyHostToDevice));

                variable_location[9] = 1;
            }
            
            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            if(variable_location[12] == 0){
                cudaCheck(cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice));

                variable_location[12] = 1;
            }

            break;
            

        // Soft Sphere Angular Forces
        case 7:
            N = nang+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[11] == 0){
                cudaCheck(cudaMemcpy(dev_ibead_ang, ibead_ang, size_int, cudaMemcpyHostToDevice));
                cudaCheck(cudaMemcpy(dev_kbead_ang, kbead_ang, size_int, cudaMemcpyHostToDevice));

                variable_location[11] = 1;
            }
            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            if(variable_location[12] == 0){
                cudaCheck(cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice));

                variable_location[12] = 1;
            }

            break;
        
        // Random Forces
        case 8:
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[12] == 0){
                cudaCheck(cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice));

                variable_location[12] = 1;
            }

            break;

        // Pos Update
        case 9:
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[2] == 0){
                cudaCheck(cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[2] = 1;
            }

            if(variable_location[12] == 0){
                cudaCheck(cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice));

                variable_location[12] = 1;
            }

            if(variable_location[13] == 0){
                cudaCheck(cudaMemcpy(dev_incr, incr, size_double3, cudaMemcpyHostToDevice));

                variable_location[13] = 1;
            }

            if(variable_location[14] == 0){
                cudaCheck(cudaMemcpy(dev_vel, vel, size_double3, cudaMemcpyHostToDevice));

                variable_location[14] = 1;
            }

            if(variable_location[15] == 0){
                cudaCheck(cudaMemcpy(dev_pos, pos, size_double3, cudaMemcpyHostToDevice));

                variable_location[15] = 1;
            }

            break;

        // Velocity Update
        case 10:
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[12] == 0){
                cudaCheck(cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice));

                variable_location[12] = 1;
            }

            if(variable_location[13] == 0){
                cudaCheck(cudaMemcpy(dev_incr, incr, size_double3, cudaMemcpyHostToDevice));

                variable_location[13] = 1;
            }

            if(variable_location[14] == 0){
                cudaCheck(cudaMemcpy(dev_vel, vel, size_double3, cudaMemcpyHostToDevice));

                variable_location[14] = 1;
            }

            break;

        // Clear Forces
        case 11:
            variable_location[12] = 1;

            break;

        default:
            break;
    }

    if(debug){
        for(int i = 0; i < 16; i++){
            printf("%d ", variable_location[i]);
        }
        fflush(stdout);
    }

    CudaCheckError();

    cudaDeviceSynchronize();

    if(debug){
        print_op(op, 1);
    }
}

void device_to_host(int op){
    int N;
    int size_int;
	int size_double;
	int size_double3;

    if(debug){
        printf("%*d: ", 2, op);
        fflush(stdout);
    }

    if(usegpu_nl || usegpu_pl || usegpu_vdw_energy || usegpu_ss_ang_energy || usegpu_fene_energy || usegpu_vdw_force || usegpu_ss_ang_force || usegpu_fene_force || usegpu_pos || usegpu_vel || usegpu_rand_force){
        cudaDeviceSynchronize();
    }

    switch(op){
        // Neighbor List
        case 0:
            N = ncon_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[0] == 1){
                cudaCheck(cudaMemcpy(ibead_lj_nat, dev_ibead_lj_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_lj_nat, dev_jbead_lj_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_lj_nat, dev_itype_lj_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_lj_nat, dev_jtype_lj_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist, dev_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist2, dev_lj_nat_pdb_dist2, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist6, dev_lj_nat_pdb_dist6, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist12, dev_lj_nat_pdb_dist12, size_double, cudaMemcpyDeviceToHost));

                variable_location[0] = 0;
            }

            if(variable_location[1] == 1){
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist, dev_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                
                variable_location[1] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            N = ncon_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[3] == 1){
                cudaCheck(cudaMemcpy(ibead_lj_non_nat, dev_ibead_lj_non_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_lj_non_nat, dev_jbead_lj_non_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_lj_non_nat, dev_itype_lj_non_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_lj_non_nat, dev_jtype_lj_non_nat, size_int, cudaMemcpyDeviceToHost));

                variable_location[3] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            // Modified variables on host
            variable_location[4] = 0;
            variable_location[5] = 0;
            variable_location[6] = 0;

            break;

        // Pair List
        case 1:
            N = nnl_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[4] == 1){
                cudaCheck(cudaMemcpy(ibead_neighbor_list_att, dev_ibead_neighbor_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_neighbor_list_att, dev_jbead_neighbor_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_neighbor_list_att, dev_itype_neighbor_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_neighbor_list_att, dev_jtype_neighbor_list_att, size_int, cudaMemcpyDeviceToHost));

                variable_location[4] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[5] == 1){
                cudaCheck(cudaMemcpy(nl_lj_nat_pdb_dist, dev_nl_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(nl_lj_nat_pdb_dist2, dev_nl_lj_nat_pdb_dist2, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(nl_lj_nat_pdb_dist6, dev_nl_lj_nat_pdb_dist6, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(nl_lj_nat_pdb_dist12, dev_nl_lj_nat_pdb_dist12, size_double, cudaMemcpyDeviceToHost));

                variable_location[5] = 0;
            }

            N = nnl_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[6] == 1){
                cudaCheck(cudaMemcpy(ibead_neighbor_list_rep, dev_ibead_neighbor_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_neighbor_list_rep, dev_jbead_neighbor_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_neighbor_list_rep, dev_itype_neighbor_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_neighbor_list_rep, dev_jtype_neighbor_list_rep, size_int, cudaMemcpyDeviceToHost));

                variable_location[6] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            variable_location[7] = 0;
            variable_location[8] = 0;

            break;

        // VDW Energy
        case 2:
            N = nil_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[7] == 1){
                cudaCheck(cudaMemcpy(ibead_pair_list_att, dev_ibead_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_pair_list_att, dev_jbead_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_pair_list_att, dev_itype_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_pair_list_att, dev_jtype_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist, dev_pl_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist2, dev_pl_lj_nat_pdb_dist2, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist6, dev_pl_lj_nat_pdb_dist6, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist12, dev_pl_lj_nat_pdb_dist12, size_double, cudaMemcpyDeviceToHost));

                variable_location[7] = 0;
            }
            
            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            N = nil_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[8] == 1){
                cudaCheck(cudaMemcpy(ibead_pair_list_rep, dev_ibead_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_pair_list_rep, dev_jbead_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_pair_list_rep, dev_itype_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_pair_list_rep, dev_jtype_pair_list_rep, size_int, cudaMemcpyDeviceToHost));

                variable_location[8] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            break;

        // Fene Energy
        case 3:
            N = nbnd+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[9] == 1){
                cudaCheck(cudaMemcpy(ibead_bnd, dev_ibead_bnd, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_bnd, dev_jbead_bnd, size_int, cudaMemcpyDeviceToHost));

                variable_location[9] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[10] == 1){
                cudaCheck(cudaMemcpy(pdb_dist, dev_pdb_dist, size_double, cudaMemcpyDeviceToHost));

                variable_location[10] = 0;
            }

            break;

        // Soft Sphere Angular Energy
        case 4:
            N = nang+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[11] == 1){
                cudaCheck(cudaMemcpy(ibead_ang, dev_ibead_ang, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(kbead_ang, dev_kbead_ang, size_int, cudaMemcpyDeviceToHost));

                variable_location[11] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }
            
            break;

        // VDW Forces
        case 5:
            N = nil_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[7] == 1){
                cudaCheck(cudaMemcpy(ibead_pair_list_att, dev_ibead_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_pair_list_att, dev_jbead_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_pair_list_att, dev_itype_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_pair_list_att, dev_jtype_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist, dev_pl_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist2, dev_pl_lj_nat_pdb_dist2, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist6, dev_pl_lj_nat_pdb_dist6, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist12, dev_pl_lj_nat_pdb_dist12, size_double, cudaMemcpyDeviceToHost));

                variable_location[7] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            N = nil_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[8] == 1){
                cudaCheck(cudaMemcpy(ibead_pair_list_rep, dev_ibead_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_pair_list_rep, dev_jbead_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_pair_list_rep, dev_itype_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_pair_list_rep, dev_jtype_pair_list_rep, size_int, cudaMemcpyDeviceToHost));

                variable_location[8] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            break;

        // Fene Forces
        case 6:
            N = nbnd+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[9] == 1){
                cudaCheck(cudaMemcpy(ibead_bnd, dev_ibead_bnd, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_bnd, dev_jbead_bnd, size_int, cudaMemcpyDeviceToHost));

                variable_location[9] = 0;
            }
            
            if(variable_location[10] == 1){
                cudaCheck(cudaMemcpy(pdb_dist, dev_pdb_dist, size_double, cudaMemcpyDeviceToHost));

                variable_location[10] = 0;
            }

            if(variable_location[9] == 1){
                cudaCheck(cudaMemcpy(ibead_bnd, dev_ibead_bnd, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_bnd, dev_jbead_bnd, size_int, cudaMemcpyDeviceToHost));

                variable_location[9] = 0;
            }
            
            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            break;
            

        // Soft Sphere Angular Forces
        case 7:
            N = nang+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[11] == 1){
                cudaCheck(cudaMemcpy(ibead_ang, dev_ibead_ang, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(kbead_ang, dev_kbead_ang, size_int, cudaMemcpyDeviceToHost));

                variable_location[11] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            break;

        // Random Forces
        case 8:
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            break;

        // Pos Update
        case 9:
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            if(variable_location[13] == 1){
                cudaCheck(cudaMemcpy(incr, dev_incr, size_double3, cudaMemcpyDeviceToHost));

                variable_location[13] = 0;
            }

            if(variable_location[14] == 1){
                cudaCheck(cudaMemcpy(vel, dev_vel, size_double3, cudaMemcpyDeviceToHost));

                variable_location[14] = 0;
            }

            if(variable_location[15] == 1){
                cudaCheck(cudaMemcpy(pos, dev_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[15] = 0;
            }

            break;

        // Velocity Update
        case 10:
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            if(variable_location[13] == 1){
                cudaCheck(cudaMemcpy(incr, dev_incr, size_double3, cudaMemcpyDeviceToHost));

                variable_location[13] = 0;
            }

            if(variable_location[14] == 1){
                cudaCheck(cudaMemcpy(vel, dev_vel, size_double3, cudaMemcpyDeviceToHost));

                variable_location[14] = 0;
            }

            break;

        // Clear Forces
        case 11:
        
            variable_location[12] = 0;

            break;

        default:
            break;
    }

    if(debug){
        for(int i = 0; i < 16; i++){
            printf("%d ", variable_location[i]);
        }
        //printf("\n");
        fflush(stdout);
    }
    
    cudaDeviceSynchronize();

    if(debug){
        print_op(op, 0);
    }

}


void device_to_host_copy(int op){
    int N;
    int size_int;
	int size_double;
	int size_double3;

    if(debug){
        printf("%*d: ", 2, op);
    }

    if(usegpu_nl || usegpu_pl || usegpu_vdw_energy || usegpu_ss_ang_energy || usegpu_fene_energy || usegpu_vdw_force || usegpu_ss_ang_force || usegpu_fene_force || usegpu_pos || usegpu_vel || usegpu_rand_force){
        cudaDeviceSynchronize();
    }

    switch(op){
        // Neighbor List
        case 0:
            N = ncon_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[0] == 1){
                cudaCheck(cudaMemcpy(ibead_lj_nat, dev_ibead_lj_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_lj_nat, dev_jbead_lj_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_lj_nat, dev_itype_lj_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_lj_nat, dev_jtype_lj_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist, dev_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist2, dev_lj_nat_pdb_dist2, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist6, dev_lj_nat_pdb_dist6, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist12, dev_lj_nat_pdb_dist12, size_double, cudaMemcpyDeviceToHost));

                variable_location[0] = 0;
            }

            if(variable_location[1] == 1){
                cudaCheck(cudaMemcpy(lj_nat_pdb_dist, dev_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                
                variable_location[1] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            N = ncon_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[3] == 1){
                cudaCheck(cudaMemcpy(ibead_lj_non_nat, dev_ibead_lj_non_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_lj_non_nat, dev_jbead_lj_non_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_lj_non_nat, dev_itype_lj_non_nat, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_lj_non_nat, dev_jtype_lj_non_nat, size_int, cudaMemcpyDeviceToHost));

                variable_location[3] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            break;

        // Pair List
        case 1:
            N = nnl_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[4] == 1){
                cudaCheck(cudaMemcpy(ibead_neighbor_list_att, dev_ibead_neighbor_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_neighbor_list_att, dev_jbead_neighbor_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_neighbor_list_att, dev_itype_neighbor_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_neighbor_list_att, dev_jtype_neighbor_list_att, size_int, cudaMemcpyDeviceToHost));

                variable_location[4] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[5] == 1){
                cudaCheck(cudaMemcpy(nl_lj_nat_pdb_dist, dev_nl_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(nl_lj_nat_pdb_dist2, dev_nl_lj_nat_pdb_dist2, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(nl_lj_nat_pdb_dist6, dev_nl_lj_nat_pdb_dist6, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(nl_lj_nat_pdb_dist12, dev_nl_lj_nat_pdb_dist12, size_double, cudaMemcpyDeviceToHost));

                variable_location[5] = 0;
            }

            N = nnl_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[6] == 1){
                cudaCheck(cudaMemcpy(ibead_neighbor_list_rep, dev_ibead_neighbor_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_neighbor_list_rep, dev_jbead_neighbor_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_neighbor_list_rep, dev_itype_neighbor_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_neighbor_list_rep, dev_jtype_neighbor_list_rep, size_int, cudaMemcpyDeviceToHost));

                variable_location[6] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            break;

        // VDW Energy
        case 2:
            N = nil_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[7] == 1){
                cudaCheck(cudaMemcpy(ibead_pair_list_att, dev_ibead_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_pair_list_att, dev_jbead_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_pair_list_att, dev_itype_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_pair_list_att, dev_jtype_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist, dev_pl_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist2, dev_pl_lj_nat_pdb_dist2, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist6, dev_pl_lj_nat_pdb_dist6, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist12, dev_pl_lj_nat_pdb_dist12, size_double, cudaMemcpyDeviceToHost));

                variable_location[7] = 0;
            }
            
            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            N = nil_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[8] == 1){
                cudaCheck(cudaMemcpy(ibead_pair_list_rep, dev_ibead_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_pair_list_rep, dev_jbead_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_pair_list_rep, dev_itype_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_pair_list_rep, dev_jtype_pair_list_rep, size_int, cudaMemcpyDeviceToHost));

                variable_location[8] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            break;

        // Fene Energy
        case 3:
            N = nbnd+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[9] == 1){
                cudaCheck(cudaMemcpy(ibead_bnd, dev_ibead_bnd, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_bnd, dev_jbead_bnd, size_int, cudaMemcpyDeviceToHost));

                variable_location[9] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[10] == 1){
                cudaCheck(cudaMemcpy(pdb_dist, dev_pdb_dist, size_double, cudaMemcpyDeviceToHost));

                variable_location[10] = 0;
            }

            break;

        // Soft Sphere Angular Energy
        case 4:
            N = nang+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[11] == 1){
                cudaCheck(cudaMemcpy(ibead_ang, dev_ibead_ang, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(kbead_ang, dev_kbead_ang, size_int, cudaMemcpyDeviceToHost));

                variable_location[11] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }
            
            break;

        // VDW Forces
        case 5:
            N = nil_att+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Native
            if(variable_location[7] == 1){
                cudaCheck(cudaMemcpy(ibead_pair_list_att, dev_ibead_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_pair_list_att, dev_jbead_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_pair_list_att, dev_itype_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_pair_list_att, dev_jtype_pair_list_att, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist, dev_pl_lj_nat_pdb_dist, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist2, dev_pl_lj_nat_pdb_dist2, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist6, dev_pl_lj_nat_pdb_dist6, size_double, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(pl_lj_nat_pdb_dist12, dev_pl_lj_nat_pdb_dist12, size_double, cudaMemcpyDeviceToHost));

                variable_location[7] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            N = nil_rep+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            // Non-Native
            if(variable_location[8] == 1){
                cudaCheck(cudaMemcpy(ibead_pair_list_rep, dev_ibead_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_pair_list_rep, dev_jbead_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(itype_pair_list_rep, dev_itype_pair_list_rep, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jtype_pair_list_rep, dev_jtype_pair_list_rep, size_int, cudaMemcpyDeviceToHost));

                variable_location[8] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            break;

        // Fene Forces
        case 6:
            N = nbnd+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[9] == 1){
                cudaCheck(cudaMemcpy(ibead_bnd, dev_ibead_bnd, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_bnd, dev_jbead_bnd, size_int, cudaMemcpyDeviceToHost));

                variable_location[9] = 0;
            }
            
            if(variable_location[10] == 1){
                cudaCheck(cudaMemcpy(pdb_dist, dev_pdb_dist, size_double, cudaMemcpyDeviceToHost));

                variable_location[10] = 0;
            }

            if(variable_location[9] == 1){
                cudaCheck(cudaMemcpy(ibead_bnd, dev_ibead_bnd, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(jbead_bnd, dev_jbead_bnd, size_int, cudaMemcpyDeviceToHost));

                variable_location[9] = 0;
            }
            
            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            break;
            

        // Soft Sphere Angular Forces
        case 7:
            N = nang+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[11] == 1){
                cudaCheck(cudaMemcpy(ibead_ang, dev_ibead_ang, size_int, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(kbead_ang, dev_kbead_ang, size_int, cudaMemcpyDeviceToHost));

                variable_location[11] = 0;
            }

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            break;

        // Random Forces
        case 8:
            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            break;

        // Pos Update
        case 9:
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[2] == 1){
                cudaCheck(cudaMemcpy(unc_pos, dev_unc_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[2] = 0;
            }

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            if(variable_location[13] == 1){
                cudaCheck(cudaMemcpy(incr, dev_incr, size_double3, cudaMemcpyDeviceToHost));

                variable_location[13] = 0;
            }

            if(variable_location[14] == 1){
                cudaCheck(cudaMemcpy(vel, dev_vel, size_double3, cudaMemcpyDeviceToHost));

                variable_location[14] = 0;
            }

            if(variable_location[15] == 1){
                cudaCheck(cudaMemcpy(pos, dev_pos, size_double3, cudaMemcpyDeviceToHost));

                variable_location[15] = 0;
            }

            break;

        // Velocity Update
        case 10:
            N = nbead+1;
            size_int = N*sizeof(int);
            size_double = N*sizeof(double);
            size_double3 = (nbead+1)*sizeof(double3);

            if(variable_location[12] == 1){
                cudaCheck(cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost));

                variable_location[12] = 0;
            }

            if(variable_location[13] == 1){
                cudaCheck(cudaMemcpy(incr, dev_incr, size_double3, cudaMemcpyDeviceToHost));

                variable_location[13] = 0;
            }

            if(variable_location[14] == 1){
                cudaCheck(cudaMemcpy(vel, dev_vel, size_double3, cudaMemcpyDeviceToHost));

                variable_location[14] = 0;
            }

            break;

        // Clear Forces
        case 11:
            
            break;

        default:
            break;
    }

    if(debug){
        for(int i = 0; i < 16; i++){
            printf("%d ", variable_location[i]);
        }
        //printf("\n");
        fflush(stdout);
    }

    CudaCheckError();

    cudaDeviceSynchronize();

    if(debug){
        print_op(op, 0);
    }
}

void host_collect(){
        if(debug)
            printf("host_collect\n");
        for(int i = 0; i < 11; i++){
            if(debug)
                printf("\t");
            device_to_host_copy(i);
            fflush(stdout);
        }
        if(debug)
            printf("host_collect\n");
}

void print_op(int op, int val){
    int N;
    int size_int;
	int size_double;
	int size_double3;

    char *direction;

    char d1[] = "GPU";
    char d2[] = "CPU";

    if(val == 1){
        direction = d1;
    }else{
        direction = d2;
    }

    //printf("%d: ", op);

    cudaDeviceSynchronize();

    switch(op){
        // Neighbor List
        case 0:
            printf("Neighbor List %s", direction);

            break;

        // Pair List
        case 1:
            printf("Pair List %s", direction);

            break;

        // VDW Energy
        case 2:
            printf("VDW Energy %s", direction);

            break;

        // Fene Energy
        case 3:
            printf("Fene Energy %s", direction);

            break;

        // Soft Sphere Angular Energy
        case 4:
            printf("Soft Sphere Angular Energy %s", direction);
            
            break;

        // VDW Forces
        case 5:
            printf("VDW Forces %s", direction);

            break;

        // Fene Forces
        case 6:
            printf("Fene Forces %s", direction);

            break;
            

        // Soft Sphere Angular Forces
        case 7:
            printf("Soft Sphere Angular Forces %s", direction);

            break;

        // Random Forces
        case 8:
            printf("Random Forces %s", direction);

            break;

        case 9:
            printf("Position Update %s", direction);

            break;

        case 10:
            printf("Velocity Update %s", direction);

            break;

        case 11:
            printf("Clearing Forces %s", direction);
            break;

        default:
            break;
    }
    printf("\n");
    fflush(stdout);
}

void setup_rng(unsigned long long seed, unsigned long long offset)
{

  int N = nbead + 1;
  
  cudaCheck(cudaMalloc((void **)&devStates, N * sizeof(curandState)));
  cudaDeviceSynchronize();
  CudaCheckError();
  
  int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);
  
  cudaDeviceSynchronize();
  setup_rng_kernel<<<blocks, threads>>>(devStates, seed, offset, N);
  cudaDeviceSynchronize();
  CudaCheckError();
}

__global__ void setup_rng_kernel(curandState *state, unsigned long long seed, unsigned long long offset, int N){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id > 0 && id < N)
  {
    curand_init(seed + id-1, 0, offset, &state[id]);
  }
}