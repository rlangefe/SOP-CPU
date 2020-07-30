#include <cstdlib>
#include <math.h>
#include <cstdio>
#include <cuda_runtime.h>
#include "global.h"
#include "energy.h"
#include "GPUvars.h"
#include "utils.h"
#include <stdlib.h> 
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

__device__ __constant__ double dev_coeff_att[3][3] = {
    {0.0, 0.0, 0.0},
	{0.0, 0.7, 0.8},
	{0.0, 0.8, 1.0}
};

__device__ __constant__ double dev_coeff_rep[3][3] = {
    {0.0, 0.0, 0.0},
	{0.0, 1.0, 1.0},
	{0.0, 1.0, 1.0}
};

__device__ __constant__ double dev_force_coeff_att[3][3] = {
    {0.0,       0.0,       0.0},
	{0.0, -12.0*1.0, -12.0*0.8},
	{0.0, -12.0*0.8, -12.0*0.7}
};

__device__ __constant__ double dev_force_coeff_rep[3][3] = {
    {0.0,       0.0,       0.0},
	{0.0,  -6.0*1.0,  -6.0*1.0},
	{0.0,  -6.0*1.0,  -6.0*1.0}
};

__device__ __constant__ double dev_sigma_rep[3][3] = {
    {0.0, 0.0, 0.0},
	{0.0, 3.8, 5.4},
	{0.0, 5.4, 7.0}
};

__device__ __constant__ double dev_rcut_nat[3][3] = {
    { 0.0,  0.0,  0.0},
    { 0.0,  8.0, 11.0},
    { 0.0, 11.0, 14.0}
};

void energy_eval()
{

  using namespace std;
  char oline[1024];

  for( int i=1; i<=npot_term; i++ ) {
    pot_term[i]();
  }

  rna_etot = e_bnd + e_ang_ss + e_vdw_rr;
  system_etot = rna_etot;

}

void force_eval()
{

  using namespace std;
  char oline[1024];

  clear_forces();

  for( int i=1; i<=nforce_term; i++ ) {
    force_term[i]();
  }

}

void clear_forces() {

  using namespace std;
  
  printf("Before Collection\n");
  fflush(stdout);
  
  device_to_host(11);

  printf("After Collection\n");
  fflush(stdout);

  printf("%f\n", force[10].x);
  fflush(stdout);

  for( int i=1; i<=nbead; i++ ) {
    force[i].x = 0.0;
    force[i].y = 0.0;
    force[i].z = 0.0;
  }

  printf("After Clearing\n");
  fflush(stdout);

  

}

void set_potential() {

  using namespace std;

  int iterm;

  iterm = 0;
  for( int i=1; i<=mpot_term; i++ ) {
    switch(i) {
    case 1:
      if( pot_term_on[i] ) {
        if(usegpu_fene_energy == 0){
	        pot_term[++iterm] = &fene_energy;
        }else{
          pot_term[++iterm] = &fene_energy_gpu;
        }
      }
      break;
    case 2:
      if( pot_term_on[i] ) {
        if(usegpu_ss_ang_energy == 0){
	        pot_term[++iterm] = &soft_sphere_angular_energy;
        }else{
          pot_term[++iterm] = &soft_sphere_angular_energy_gpu;
        }
      }
      break;
    case 5:
      if( pot_term_on[i] ) {
        if(usegpu_vdw_energy == 0){
	        pot_term[++iterm] = &vdw_energy;
        }else{
          pot_term[++iterm] = &vdw_energy_gpu;
        }
      }
      break;
    default:
      break;
    }
  }

}

void set_forces()
{

  using namespace std;

  int iterm;

  iterm = 0;
  for( int i=1; i<=mforce_term; i++ ) {
    switch(i) {
    case 1:
      if( force_term_on[i] ) {
        if(usegpu_rand_force == 0){
	        force_term[++iterm] = &random_force;
        }else{
          force_term[++iterm] = &random_force_gpu;
        }
      }
      break;
    case 2:
      if( force_term_on[i] ) {
        if(usegpu_fene_force == 0){
	        force_term[++iterm] = &fene_forces;
        }else{
          force_term[++iterm] = &fene_forces_gpu;
        }
      }
      break;
    case 3:
      if( force_term_on[i] ) {
        if(usegpu_ss_ang_force == 0){
	        force_term[++iterm] = &soft_sphere_angular_forces;
        }else{
          force_term[++iterm] = &soft_sphere_angular_forces_gpu;
        }
      }
      break;
    case 6:
      if( force_term_on[i] ) {
        if(usegpu_vdw_force == 0){
	        force_term[++iterm] = &vdw_forces;
        }else{
          force_term[++iterm] = &vdw_forces_gpu;
        }
      }
      break;
    default:
      break;
    }
  }

}

void fene_energy()
{

  using namespace std;

  device_to_host(3);

  int ibead, jbead;
  double dx, dy, dz, d,dev;
  char line[2048];

  e_bnd = 0.0;
  for( int i=1; i<=nbnd; i++ ) {

    ibead = ibead_bnd[i];
    jbead = jbead_bnd[i];

    dx = unc_pos[jbead].x-unc_pos[ibead].x;
    dy = unc_pos[jbead].y-unc_pos[ibead].y;
    dz = unc_pos[jbead].z-unc_pos[ibead].z;


    // min images

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d = sqrt(dx*dx+dy*dy+dz*dz);
    dev = d-pdb_dist[i];

    e_bnd += log1p(-dev*dev/R0sq); // log1p(x) = log(1-x)

  }

  e_bnd *= -e_bnd_coeff;

  return;

}

void soft_sphere_angular_energy()
{

  using namespace std;

  device_to_host(4);

  e_ang_ss = 0.0;
  int ibead, kbead;
  coord r_ik;
  double d,d6;

  for( int i=1; i<=nang; i++ ) {

    ibead = ibead_ang[i];
    kbead = kbead_ang[i];

    r_ik.x = unc_pos[kbead].x - unc_pos[ibead].x;
    r_ik.y = unc_pos[kbead].y - unc_pos[ibead].y;
    r_ik.z = unc_pos[kbead].z - unc_pos[ibead].z;

    // min images

    r_ik.x -= boxl*rnd(r_ik.x/boxl);
    r_ik.y -= boxl*rnd(r_ik.y/boxl);
    r_ik.z -= boxl*rnd(r_ik.z/boxl);

    d = sqrt(r_ik.x*r_ik.x + r_ik.y*r_ik.y + r_ik.z*r_ik.z);
    d6 = pow(d,6.0);

    e_ang_ss += e_ang_ss_coeff/d6;
  }

  return;

}

void vdw_energy()
{

  using namespace std;

  device_to_host(2);

  int ibead,jbead;
  int itype,jtype;
  double dx,dy,dz,d,d2,d6,d12;
  char line[2048];

  e_vdw_rr = 0.0;
  e_vdw_rr_att = 0.0;
  e_vdw_rr_rep = 0.0;

  for( int i=1; i<=nil_att; i++ ) {

    ibead = ibead_pair_list_att[i];
    jbead = jbead_pair_list_att[i];
    itype = itype_pair_list_att[i];
    jtype = jtype_pair_list_att[i];

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    // min images

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;
    d6 = d2*d2*d2;
    d12 = d6*d6;

    e_vdw_rr_att += coeff_att[itype][jtype] * (pl_lj_nat_pdb_dist12[i]/d12)-2.0*(pl_lj_nat_pdb_dist6[i]/d6);

  }

  for( int i=1; i<=nil_rep; i++ ) {

    ibead = ibead_pair_list_rep[i];
    jbead = jbead_pair_list_rep[i];
    itype = itype_pair_list_rep[i];
    jtype = jtype_pair_list_rep[i];

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    // min images

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;
    d6 = d2*d2*d2;
    d12 = d6*d6;

    e_vdw_rr_rep += coeff_rep[itype][jtype] * (sigma_rep12[itype][jtype]/d12+sigma_rep6[itype][jtype]/d6);

  }

  e_vdw_rr = e_vdw_rr_att + e_vdw_rr_rep;

  return;

}

void vdw_forces()
{

  using namespace std;

  device_to_host(5);

  char line[2048];

  int ibead,jbead;
  int itype,jtype;
  double dx,dy,dz,d,d2,d6,d12;
  double fx,fy,fz;
  double co1;
  const static double tol = 1.0e-7;
  double rep_tol;

  for( int i=1; i<=nil_att; i++ ) {

    ibead = ibead_pair_list_att[i];
    jbead = jbead_pair_list_att[i];
    itype = itype_pair_list_att[i];
    jtype = jtype_pair_list_att[i];

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    // min images

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;
    
    if( d2 < tol*pl_lj_nat_pdb_dist2[i] ) continue;
    d6 = d2*d2*d2;
    d12 = d6*d6;

    co1 = force_coeff_att[itype][jtype]/d2*((pl_lj_nat_pdb_dist12[i]/d12)-(pl_lj_nat_pdb_dist6[i]/d6));

    fx = co1*dx;
    fy = co1*dy;
    fz = co1*dz;

    force[ibead].x += fx;
    force[ibead].y += fy;
    force[ibead].z += fz;

    force[jbead].x -= fx;
    force[jbead].y -= fy;
    force[jbead].z -= fz;

  }

  for( int i=1; i<=nil_rep; i++ ) {

    ibead = ibead_pair_list_rep[i];
    jbead = jbead_pair_list_rep[i];
    itype = itype_pair_list_rep[i];
    jtype = jtype_pair_list_rep[i];

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    // min images

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;
    rep_tol = sigma_rep2[itype][jtype]*tol;
    if( d2 <  rep_tol ) continue;
    d6 = d2*d2*d2;
    d12 = d6*d6;

    co1 = force_coeff_rep[itype][jtype]/d2*
      (2.0*sigma_rep12[itype][jtype]/d12+sigma_rep6[itype][jtype]/d6);

    fx = co1*dx;
    fy = co1*dy;
    fz = co1*dz;

    force[ibead].x += fx;
    force[ibead].y += fy;
    force[ibead].z += fz;

    force[jbead].x -= fx;
    force[jbead].y -= fy;
    force[jbead].z -= fz;

  }

}

void soft_sphere_angular_forces()
{

  using namespace std;

  device_to_host(7);

  char line[2048];

  int ibead,kbead;
  double dx,dy,dz,d,d8;
  double fx,fy,fz;
  double co1;

  for( int i=1; i<=nang; i++ ) {

      ibead = ibead_ang[i];
      kbead = kbead_ang[i];

      dx = unc_pos[kbead].x - unc_pos[ibead].x;
      dy = unc_pos[kbead].y - unc_pos[ibead].y;
      dz = unc_pos[kbead].z - unc_pos[ibead].z;

      // min images

      dx -= boxl*rnd(dx/boxl);
      dy -= boxl*rnd(dy/boxl);
      dz -= boxl*rnd(dz/boxl);

      d = sqrt(dx*dx+dy*dy+dz*dz);
      d8 = pow(d,8.0);

      co1 = f_ang_ss_coeff/d8;

      fx = co1*dx;
      fy = co1*dy;
      fz = co1*dz;

      force[ibead].x -= fx;
      force[ibead].y -= fy;
      force[ibead].z -= fz;

      force[kbead].x += fx;
      force[kbead].y += fy;
      force[kbead].z += fz;

  }

}

void fene_forces()
{

  using namespace std;

  printf("Before\n");
  fflush(stdout);
  device_to_host(6);
  printf("After\n");
  fflush(stdout);

  int ibead, jbead;
  double dx, dy, dz, d, dev, dev2;
  double fx, fy, fz;
  double temp;

  char line[2048];

  for( int i=1; i<=nbnd; i++ ) {

    ibead = ibead_bnd[i];
    jbead = jbead_bnd[i];

    dx = unc_pos[jbead].x-unc_pos[ibead].x;
    dy = unc_pos[jbead].y-unc_pos[ibead].y;
    dz = unc_pos[jbead].z-unc_pos[ibead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d = sqrt(dx*dx+dy*dy+dz*dz);
    dev = d - pdb_dist[i];
    dev2 = dev*dev;
    temp = -k_bnd*dev/d/(1.0-dev2/R0sq);

    fx = temp*dx;
    fy = temp*dy;
    fz = temp*dz;

    force[ibead].x -= fx;
    force[ibead].y -= fy;
    force[ibead].z -= fz;

    force[jbead].x += fx;
    force[jbead].y += fy;
    force[jbead].z += fz;

  }

}

void random_force() {

  using namespace std;

  device_to_host(8);

  double var;
  int problem;

  var = sqrt(2.0*T*zeta/h);

  for( int i=1; i<=nbead; i++ ) {
    force[i].x += var*generator.gasdev();
    force[i].y += var*generator.gasdev();
    force[i].z += var*generator.gasdev();

  }

}

/**********************
* Start GPU Functions *
**********************/

void random_force_gpu(){
  using namespace std;

  host_to_device(8);

  double var = sqrt(2.0*T*zeta/h);

  int N = nbead+1;

  int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);
  
  rand_kernel<<<blocks, threads>>>(N, dev_force, devStates, var);
  cudaDeviceSynchronize();
 
  CudaCheckError();
  
}

__global__ void rand_kernel(int N, double3 *dev_force, curandState *dev_state, double var){
  unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i > 0 && i < N){
    // Copy state to local memory for efficiency
    curandState localState = dev_state[i];
    
    dev_force[i].x +=  curand_normal(&localState) * var;
    dev_force[i].y +=  curand_normal(&localState) * var;
    dev_force[i].z +=  curand_normal(&localState) * var;
    
    // Copy state back to global memory
    dev_state[i] = localState;
  }
}

void vdw_energy_gpu(){
  e_vdw_rr = 0.0;
  e_vdw_rr_att = 0.0;
  e_vdw_rr_rep = 0.0;

  host_to_device(2);

  vdw_energy_att_gpu();
  
  CudaCheckError();

  vdw_energy_rep_gpu();

  CudaCheckError();

  e_vdw_rr = e_vdw_rr_att + e_vdw_rr_rep;

  return;
}

void vdw_energy_att_gpu(){
  e_vdw_rr_att = 0.0;

  int N = nil_att+1;

	int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	vdw_energy_att_value_kernel<<<blocks, threads>>>(dev_ibead_pair_list_att, dev_jbead_pair_list_att, dev_itype_pair_list_att, dev_jtype_pair_list_att, 
														dev_pl_lj_nat_pdb_dist6, dev_pl_lj_nat_pdb_dist12, dev_unc_pos, N, boxl, dev_value_double);

  CudaCheckError();
	
	hier_ks_scan(dev_value_double, dev_value_double, N, 0);
	
	cudaCheck(cudaMemcpy(&e_vdw_rr_att, &dev_value_double[N-1], sizeof(double), cudaMemcpyDeviceToHost));

  CudaCheckError();
}

__global__ void vdw_energy_att_value_kernel(int *dev_ibead_pair_list_att, int *dev_jbead_pair_list_att, int *dev_itype_pair_list_att, int *dev_jtype_pair_list_att, 
											double *dev_pl_lj_nat_pdb_dist6, double *dev_pl_lj_nat_pdb_dist12, double3 *dev_unc_pos, int N, double boxl, double *dev_result){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 0 && i < N){
		int ibead,jbead;
		int itype,jtype;
		double dx,dy,dz,d,d2,d6,d12;
		
		ibead = dev_ibead_pair_list_att[i];
		jbead = dev_jbead_pair_list_att[i];
		itype = dev_itype_pair_list_att[i];
		jtype = dev_jtype_pair_list_att[i];

		dx = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;
		dy = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;
		dz = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;

		// min images

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

		d2 = dx*dx+dy*dy+dz*dz;
		d6 = d2*d2*d2;
		d12 = d6*d6;

		dev_result[i] = dev_coeff_att[itype][jtype] * (dev_pl_lj_nat_pdb_dist12[i]/d12)-2.0*(dev_pl_lj_nat_pdb_dist6[i]/d6);
	}else if(i == 0){
		dev_result[i] = 0;
	}
}

void vdw_energy_rep_gpu(){
	e_vdw_rr_rep = 0.0;
	
	int N = nil_rep+1;
	
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead + 1)*sizeof(double3);
	
	int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	vdw_energy_rep_value_kernel<<<blocks, threads>>>(dev_ibead_pair_list_rep, dev_jbead_pair_list_rep, dev_itype_pair_list_rep, dev_jtype_pair_list_rep, 
													                        dev_unc_pos, N, boxl, dev_value_double);

  CudaCheckError();
	
	hier_ks_scan(dev_value_double, dev_value_double, N, 0);

	cudaCheck(cudaMemcpy(&e_vdw_rr_rep, &dev_value_double[N-1], sizeof(double), cudaMemcpyDeviceToHost));

  CudaCheckError();
}

__global__ void vdw_energy_rep_value_kernel(int *dev_ibead_pair_list_rep, int *dev_jbead_pair_list_rep, int *dev_itype_pair_list_rep, int *dev_jtype_pair_list_rep, 
											double3 *dev_unc_pos, int N, double boxl, double *dev_result){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 0 && i < N){
		int ibead,jbead;
		int itype,jtype;
		double dx,dy,dz,d,d2,d6,d12;
		
		ibead = dev_ibead_pair_list_rep[i];
		jbead = dev_jbead_pair_list_rep[i];
		itype = dev_itype_pair_list_rep[i];
		jtype = dev_jtype_pair_list_rep[i];

		dx = dev_unc_pos[jbead].x - dev_unc_pos[ibead].x;
		dy = dev_unc_pos[jbead].y - dev_unc_pos[ibead].y;
		dz = dev_unc_pos[jbead].z - dev_unc_pos[ibead].z;

		// min images

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

		d2 = dx*dx+dy*dy+dz*dz;
		d6 = d2*d2*d2;
		d12 = d6*d6;

    double s = dev_sigma_rep[itype][jtype];
    double s2 = s*s;
    double s6 = s2*s2*s2;
    double s12 = s6*s6;
		
    dev_result[i] = dev_coeff_rep[itype][jtype] * (s12/d12 + s6/d6);
	}else if(i == 0){
		dev_result[i] = 0;
	}
}

void hier_ks_scan(double *dev_X, double *dev_Y, int N, int re){
    if(N <= SECTION_SIZE){
        ksScanInc<<<1, N>>>(dev_X, dev_Y, N);

        CudaCheckError();

        cudaDeviceSynchronize();

        return;
    }else{
        int threads = (int)min(N, SECTION_SIZE);
        int blocks = (int)ceil(1.0*N/SECTION_SIZE);

        double *dev_S;
        cudaCheck(cudaMalloc((void**)&dev_S, (int)ceil(1.0*N/SECTION_SIZE) * sizeof(double)));
        
        ksScanAuxInc<<<blocks, threads>>>(dev_X, dev_Y, N, dev_S);
        cudaDeviceSynchronize();

        hier_ks_scan(dev_S, dev_S, (int)ceil(1.0*N/SECTION_SIZE), 1);
        cudaDeviceSynchronize();
        
        sumIt<<<blocks, threads>>>(dev_Y, dev_S, N);
        cudaDeviceSynchronize();

        CudaCheckError();

        cudaFree(dev_S);

        return;
    }
}

__global__ void ksScanAuxExc (double *X, double *Y, int InputSize, double *S) {
    double val;
    
    __shared__ double XY[SECTION_SIZE];
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

__global__ void ksScanAuxInc (double *X, double *Y, int InputSize, double *S) {
    double val;
    
    __shared__ double XY[SECTION_SIZE];
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

__global__ void ksScanExc (double *X, double *Y, double InputSize) {
    double val;
    
    __shared__ double XY[SECTION_SIZE];
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

__global__ void ksScanInc (double *X, double *Y, int InputSize) {
    double val;
    
    __shared__ double XY[SECTION_SIZE];
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

__global__ void sumIt (double *Y, double *S, int InputSize) {
    if(blockIdx.x > 0){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < InputSize){
            Y[i] += S[blockIdx.x-1];
        }
    }
}

void vdw_forces_gpu()
{
  using namespace std;

  host_to_device(5);

  vdw_forces_att_gpu();

  CudaCheckError();

  vdw_forces_rep_gpu();

  CudaCheckError();
}

void vdw_forces_att_gpu(){	
	int N = nil_att + 1;
	
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	int threads = (int)min(N, SECTION_SIZE);
	int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	vdw_forces_att_kernel<<<blocks, threads>>>(dev_ibead_pair_list_att, dev_jbead_pair_list_att, dev_itype_pair_list_att, dev_jtype_pair_list_att, 
												dev_pl_lj_nat_pdb_dist, boxl, N, dev_unc_pos, dev_force);

  CudaCheckError();
}

__global__ void vdw_forces_att_kernel(int *dev_ibead_pair_list_att, int *dev_jbead_pair_list_att, int *dev_itype_pair_list_att, int *dev_jtype_pair_list_att, 
								double *dev_pl_lj_nat_pdb_dist, double boxl, int N, double3 *dev_unc_pos, double3 *dev_force){
									
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i > 0 && i < N){
		int ibead,jbead;
		int itype,jtype;
		double dx,dy,dz,d,d2,d6,d12;
		double fx,fy,fz;
		double co1;
		const static double tol = 1.0e-7;

		ibead = dev_ibead_pair_list_att[i];
		jbead = dev_jbead_pair_list_att[i];
		itype = dev_itype_pair_list_att[i];
		jtype = dev_jtype_pair_list_att[i];

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
		
		double pl_dist2 = dev_pl_lj_nat_pdb_dist[i] * dev_pl_lj_nat_pdb_dist[i];
		
		if( d2 < tol*pl_dist2 ) return;
		d6 = d2*d2*d2;
		d12 = d6*d6;

		double pl_dist6 = pl_dist2 * pl_dist2 * pl_dist2;

		double pl_dist12 = pl_dist6 * pl_dist6;

		co1 = dev_force_coeff_att[itype][jtype]/d2*((pl_dist12/d12)-(pl_dist6/d6));

		fx = co1*dx;
		fy = co1*dy;
		fz = co1*dz;

		//dev_force[ibead].x += fx;
		atomicAdd(&dev_force[ibead].x, fx);
		
		//dev_force[ibead].y += fy;
		atomicAdd(&dev_force[ibead].y, fy);
		
		//dev_force[ibead].z += fz;
		atomicAdd(&dev_force[ibead].z, fz);

		//dev_force[jbead].x -= fx;
		atomicAdd(&dev_force[jbead].x, -1.0*fx);
		
		//dev_force[jbead].y -= fy;
		atomicAdd(&dev_force[jbead].y, -1.0*fy);
		
		//dev_force[jbead].z -= fz;
		atomicAdd(&dev_force[jbead].z, -1.0*fz);
	}
}

void vdw_forces_rep_gpu(){	
	int N = nil_rep + 1;
	
	int size_int = N*sizeof(int);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	int threads = (int)min(N, SECTION_SIZE);
	int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	vdw_forces_rep_kernel<<<blocks, threads>>>(dev_ibead_pair_list_rep, dev_jbead_pair_list_rep, dev_itype_pair_list_rep, dev_jtype_pair_list_rep, 
												boxl, N, dev_unc_pos, dev_force);
  CudaCheckError();
}

__global__ void vdw_forces_rep_kernel(int *dev_ibead_pair_list_rep, int *dev_jbead_pair_list_rep, int *dev_itype_pair_list_rep, int *dev_jtype_pair_list_rep, double boxl, int N,
								double3 *dev_unc_pos, double3 *dev_force){
									
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i > 0 && i < N) {
		int ibead,jbead;
		int itype,jtype;
		double dx,dy,dz,d,d2,d6,d12;
		double fx,fy,fz;
		double co1;
		const static double tol = 1.0e-7;
		double rep_tol;

		ibead = dev_ibead_pair_list_rep[i];
		jbead = dev_jbead_pair_list_rep[i];
		itype = dev_itype_pair_list_rep[i];
		jtype = dev_jtype_pair_list_rep[i];

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

		double s2 = dev_sigma_rep[itype][jtype] * dev_sigma_rep[itype][jtype];
			
		rep_tol = s2*tol;

		if( d2 <  rep_tol ) return;
		d6 = d2*d2*d2;
		d12 = d6*d6;

		double s6 = s2*s2*s2;
		double s12 = s6*s6;

		co1 = dev_force_coeff_rep[itype][jtype]/d2 * (2.0*s12/d12+s6/d6);

		fx = co1*dx;
		fy = co1*dy;
		fz = co1*dz;

		//dev_force[ibead].x += fx;
		atomicAdd(&dev_force[ibead].x, fx);
		
		//dev_force[ibead].y += fy;
		atomicAdd(&dev_force[ibead].y, fy);
		
		//dev_force[ibead].z += fz;
		atomicAdd(&dev_force[ibead].z, fz);

		//dev_force[jbead].x -= fx;
		atomicAdd(&dev_force[jbead].x, -1.0*fx);
		
		//dev_force[jbead].y -= fy;
		atomicAdd(&dev_force[jbead].y, -1.0*fy);
		
		//dev_force[jbead].z -= fz;
		atomicAdd(&dev_force[jbead].z, -1.0*fz);

	}
}

void fene_energy_gpu(){	
  host_to_device(3);

	int N = nbnd+1;
	int size_int = N*sizeof(int);
	int size_double3 = (nbead+1)*sizeof(double3);
	int size_double = N*sizeof(double);
	
	e_bnd = 0.0;
	
	int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	fene_energy_gpu_kernel<<<blocks, threads>>>(dev_ibead_bnd, dev_jbead_bnd, dev_unc_pos, dev_pdb_dist, boxl, N, R0sq, dev_value_double);

  CudaCheckError();
	
	hier_ks_scan(dev_value_double, dev_value_double, N, 0);
	
	cudaMemcpy(&e_bnd, &dev_value_double[N-1], sizeof(double), cudaMemcpyDeviceToHost);

  CudaCheckError();
	
	e_bnd *= -e_bnd_coeff;

	return;
}

__global__ void fene_energy_gpu_kernel(int *dev_ibead_bnd, int *dev_jbead_bnd, double3 *dev_unc_pos, double *dev_pdb_dist, int boxl, int N, double dev_R0sq, double *dev_result){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 0 && i < N){
		int ibead, jbead;
		double dx, dy, dz, d,dev;
		
		ibead = dev_ibead_bnd[i];
		jbead = dev_jbead_bnd[i];

		dx = dev_unc_pos[jbead].x-dev_unc_pos[ibead].x;
		dy = dev_unc_pos[jbead].y-dev_unc_pos[ibead].y;
		dz = dev_unc_pos[jbead].z-dev_unc_pos[ibead].z;

		// min images

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

		d = std::sqrt(dx*dx+dy*dy+dz*dz);
		dev = d-dev_pdb_dist[i];

		dev_result[i] = std::log1p(-dev*dev/dev_R0sq);
	}else if(i == 0){
		dev_result[i] = 0.0;
	}
}


void soft_sphere_angular_energy_gpu(){
  host_to_device(4);

	e_ang_ss = 0.0;

	int N = nang+1;
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	int threads = (int)min(N, SECTION_SIZE);
	int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	soft_sphere_angular_energy_gpu_kernel<<<blocks, threads>>>(dev_ibead_ang, dev_kbead_ang, dev_unc_pos, boxl, N, e_ang_ss_coeff, dev_value_double);

  CudaCheckError();

	hier_ks_scan(dev_value_double, dev_value_double, N, 0);
	
  CudaCheckError();

	cudaMemcpy(&e_ang_ss, &dev_value_double[N-1], sizeof(double), cudaMemcpyDeviceToHost);

  CudaCheckError();
}

__global__ void soft_sphere_angular_energy_gpu_kernel(int *dev_ibead_ang, int *dev_kbead_ang, double3 *dev_unc_pos, int boxl, int N, double coeff, double *dev_result){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i > 0 && i < N){
		int ibead, kbead;
	  double3 r_ik;
		double d,d6;

    ibead = dev_ibead_ang[i];
    kbead = dev_kbead_ang[i];
		
		r_ik.x = dev_unc_pos[kbead].x - dev_unc_pos[ibead].x;
		r_ik.y = dev_unc_pos[kbead].y - dev_unc_pos[ibead].y;
		r_ik.z = dev_unc_pos[kbead].z - dev_unc_pos[ibead].z;

		// min images

    
		// apply periodic boundary conditions to dx, dy, and dz
		//r_ik.x -= boxl*rnd(r_ik.x/boxl);
		double rnd_value;

		rnd_value = ( ((r_ik.x/boxl)>0) ? std::floor((r_ik.x/boxl)+0.5) : std::ceil((r_ik.x/boxl)-0.5) );
		r_ik.x -= boxl*rnd_value;

		//r_ik.y -= boxl*rnd(r_ik.y/boxl);
		rnd_value = ( ((r_ik.y/boxl)>0) ? std::floor((r_ik.y/boxl)+0.5) : std::ceil((r_ik.y/boxl)-0.5) );
		r_ik.y -= boxl*rnd_value;

		//r_ik.z -= boxl*rnd(r_ik.z/boxl);
		rnd_value = ( ((r_ik.z/boxl)>0) ? std::floor((r_ik.z/boxl)+0.5) : std::ceil((r_ik.z/boxl)-0.5) );
		r_ik.z -= boxl*rnd_value;
    
		d = std::sqrt(r_ik.x*r_ik.x + r_ik.y*r_ik.y + r_ik.z*r_ik.z);
		d6 = std::pow(d,6.0);
    
		dev_result[i] = coeff/d6;
	}else if(i == 0){
		dev_result[i] = 0;
	}
	
}

void soft_sphere_angular_forces_gpu(){	
  host_to_device(7);
	
	int N = nang + 1;
	
	int size_int = N*sizeof(int);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	int threads = (int)min(N, SECTION_SIZE);
	int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	soft_sphere_angular_forces_kernel<<<blocks, threads>>>(dev_ibead_ang, dev_kbead_ang, boxl, N, f_ang_ss_coeff, dev_unc_pos, dev_force);

  CudaCheckError();
}

__global__ void soft_sphere_angular_forces_kernel(int *dev_ibead_ang, int *dev_kbead_ang, double boxl, int N, double f_ang_ss_coeff, double3 *dev_unc_pos, double3 *dev_force){
									
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i > 0 && i < N) {
		int ibead,kbead;
		double dx,dy,dz,d,d8;
		double fx,fy,fz;
		double co1;

		ibead = dev_ibead_ang[i];
		kbead = dev_kbead_ang[i];

		dx = dev_unc_pos[kbead].x - dev_unc_pos[ibead].x;
		dy = dev_unc_pos[kbead].y - dev_unc_pos[ibead].y;
		dz = dev_unc_pos[kbead].z - dev_unc_pos[ibead].z;
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

		d = std::sqrt(dx*dx+dy*dy+dz*dz);
		d8 = std::pow(d,8.0);

		co1 = f_ang_ss_coeff/d8;

		fx = co1*dx;
		fy = co1*dy;
		fz = co1*dz;

		
		atomicAdd(&dev_force[kbead].x, fx);
		
		atomicAdd(&dev_force[kbead].y, fy);
		
		atomicAdd(&dev_force[kbead].z, fz);

		
		atomicAdd(&dev_force[ibead].x, -1.0*fx);
		
		atomicAdd(&dev_force[ibead].y, -1.0*fy);
		
		atomicAdd(&dev_force[ibead].z, -1.0*fz);

	}
}

void fene_forces_gpu(){
  host_to_device(6);

	int N = nbnd + 1;
	
	int size_int = N*sizeof(int);
	int size_double3 = (nbead+1)*sizeof(double3);
	int size_double = N*sizeof(double);
	
	int threads = (int)min(N, SECTION_SIZE);
	int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	fene_forces_kernel<<<blocks, threads>>>(dev_ibead_bnd, dev_jbead_bnd, dev_pdb_dist, boxl, N, R0sq, k_bnd, dev_unc_pos, dev_force);

  CudaCheckError();
}

__global__ void fene_forces_kernel(int *dev_ibead_bnd, int *dev_jbead_bnd, double *dev_pdb_dist, double boxl, int N, double dev_R0sq, double dev_k_bnd, double3 *dev_unc_pos, double3 *dev_force){
									
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i > 0 && i < N) {
		int ibead, jbead;
		double dx, dy, dz, d, dev, dev2;
		double fx, fy, fz;
		double temp;

		ibead = dev_ibead_bnd[i];
		jbead = dev_jbead_bnd[i];
    
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
    
		d = std::sqrt(dx*dx+dy*dy+dz*dz);
		dev = d - dev_pdb_dist[i];
		dev2 = dev*dev;
		temp = -dev_k_bnd*dev/d/(1.0-dev2/dev_R0sq);


		fx = temp*dx;
		fy = temp*dy;
		fz = temp*dz;

		
		atomicAdd(&dev_force[jbead].x, fx);
		
		atomicAdd(&dev_force[jbead].y, fy);
		
		atomicAdd(&dev_force[jbead].z, fz);

		
		atomicAdd(&dev_force[ibead].x, -1.0*fx);
		
		atomicAdd(&dev_force[ibead].y, -1.0*fy);
		
		atomicAdd(&dev_force[ibead].z, -1.0*fz);

	}
}