#include <cstdlib>
#include <math.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "global.h"
#include "energy.h"
#include <stdlib.h> 

#define SECTION_SIZE 1024

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

__device__ __constant__ double dev_sigma_rep[3][3] = {
	{0.0, 0.0, 0.0},
	{0.0, 3.8, 5.4},
	{0.0, 5.4, 7.0}
};

__device__ __constant__ double dev_force_coeff_att[3][3] = { 
  {0.0,       0.0,       0.0},
  {0.0, -12.0*1.0, -12.0*0.8},
	{0.0, -12.0*0.8, -12.0*0.7}
};

__device__ double dev_force_coeff_rep[3][3] = {
  {0.0,       0.0,       0.0},
	{0.0,  -6.0*1.0,  -6.0*1.0},
	{0.0,  -6.0*1.0,  -6.0*1.0}
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

  for( int i=1; i<=nbead; i++ ) {
    force[i].x = 0.0;
    force[i].y = 0.0;
    force[i].z = 0.0;
  }

}

void set_potential() {

  using namespace std;

  int iterm;

  iterm = 0;
  for( int i=1; i<=mpot_term; i++ ) {
    switch(i) {
    case 1:
      if( pot_term_on[i] ) {
	      pot_term[++iterm] = &fene_energy;
      }
      break;
    case 2:
      if( pot_term_on[i] ) {
	      pot_term[++iterm] = &soft_sphere_angular_energy;
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
	      force_term[++iterm] = &random_force;
      }
      break;
    case 2:
      if( force_term_on[i] ) {
	      force_term[++iterm] = &fene_forces;
      }
      break;
    case 3:
      if( force_term_on[i] ) {
	      force_term[++iterm] = &soft_sphere_angular_forces;
      }
      break;
    case 6:
      if( force_term_on[i] ) {
        if(usegpu_vdw_force == 0){
	        force_term[++iterm] = &vdw_forces;
        }else{
          force_term[++iterm] = &vdw_forces_matrix_gpu;
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

void vdw_energy_gpu()
{
  e_vdw_rr = 0.0;
  e_vdw_rr_att = 0.0;
  e_vdw_rr_rep = 0.0;

  vdw_energy_att_gpu();

  vdw_energy_rep_gpu();

  e_vdw_rr = e_vdw_rr_att + e_vdw_rr_rep;

  return;
}

void vdw_energy_att_gpu(){	
	int N = nil_att + 1;
	
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead + 1)*sizeof(double3);

	int *dev_ibead_pair_list_att;
	int *dev_jbead_pair_list_att;
	int *dev_itype_pair_list_att;
	int *dev_jtype_pair_list_att;
	double *dev_pl_lj_nat_pdb_dist6;
	double *dev_pl_lj_nat_pdb_dist12;
	
	double3 *dev_unc_pos;
	
	double *dev_result;
	
	cudaMalloc((void **)&dev_ibead_pair_list_att, size_int);
	cudaMalloc((void **)&dev_jbead_pair_list_att, size_int);
	cudaMalloc((void **)&dev_itype_pair_list_att, size_int);
	cudaMalloc((void **)&dev_jtype_pair_list_att, size_int);
	cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist6, size_double);
	cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist12, size_double);
	
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	
	cudaMalloc((void **)&dev_result, size_double);
	
	cudaMemcpy(dev_ibead_pair_list_att, ibead_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_pair_list_att, jbead_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_pair_list_att, itype_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_pair_list_att, jtype_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pl_lj_nat_pdb_dist6, pl_lj_nat_pdb_dist6, size_double, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pl_lj_nat_pdb_dist12, pl_lj_nat_pdb_dist12, size_double, cudaMemcpyHostToDevice);
	
	
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	
	int threads = (int)min(N, SECTION_SIZE);
  	int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	vdw_energy_att_value_kernel<<<blocks, threads>>>(dev_ibead_pair_list_att, dev_jbead_pair_list_att, dev_itype_pair_list_att, dev_jtype_pair_list_att, 
														dev_pl_lj_nat_pdb_dist6, dev_pl_lj_nat_pdb_dist12, dev_unc_pos, N, boxl, dev_result);
	
	hier_ks_scan(dev_result, dev_result, N, 0);
	
	cudaMemcpy(&e_vdw_rr_att, &dev_result[N-1], sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_ibead_pair_list_att);
	cudaFree(dev_jbead_pair_list_att);
	cudaFree(dev_itype_pair_list_att);
	cudaFree(dev_jtype_pair_list_att);
	cudaFree(dev_pl_lj_nat_pdb_dist6);
	cudaFree(dev_pl_lj_nat_pdb_dist12);
	
	cudaFree(dev_unc_pos);
	
	cudaFree(dev_result);
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
	
	int N = nil_rep + 1;
	
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead + 1)*sizeof(double3);

	int *dev_ibead_pair_list_rep;
  int *dev_jbead_pair_list_rep;
  int *dev_itype_pair_list_rep;
  int *dev_jtype_pair_list_rep;
	
	double3 *dev_unc_pos;
	
	double *dev_result;
	
	cudaMalloc((void **)&dev_ibead_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_jbead_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_itype_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_jtype_pair_list_rep, size_int);
	
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	
	cudaMalloc((void **)&dev_result, size_double);
	
	cudaMemcpy(dev_ibead_pair_list_rep, ibead_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_pair_list_rep, jbead_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_pair_list_rep, itype_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_pair_list_rep, jtype_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	
	
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	
	int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	vdw_energy_rep_value_kernel<<<blocks, threads>>>(dev_ibead_pair_list_rep, dev_jbead_pair_list_rep, dev_itype_pair_list_rep, dev_jtype_pair_list_rep, 
													                        dev_unc_pos, N, boxl, dev_result);
	
	hier_ks_scan(dev_result, dev_result, N, 0);
	
	cudaMemcpy(&e_vdw_rr_rep, &dev_result[N-1], sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_ibead_pair_list_rep);
  cudaFree(dev_jbead_pair_list_rep);
  cudaFree(dev_itype_pair_list_rep);
  cudaFree(dev_jtype_pair_list_rep);
	
	cudaFree(dev_unc_pos);
	
	cudaFree(dev_result);
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

        cudaDeviceSynchronize();

        return;
    }else{
        int threads = (int)min(N, SECTION_SIZE);
        int blocks = (int)ceil(1.0*N/SECTION_SIZE);

        double *dev_S;
        cudaMalloc((void**)&dev_S, (int)ceil(1.0*N/SECTION_SIZE) * sizeof(double));
        
        ksScanAuxInc<<<blocks, threads>>>(dev_X, dev_Y, N, dev_S);
        cudaDeviceSynchronize();

        hier_ks_scan(dev_S, dev_S, (int)ceil(1.0*N/SECTION_SIZE), 1);
        cudaDeviceSynchronize();
        
        sumIt<<<blocks, threads>>>(dev_Y, dev_S, N);
        cudaDeviceSynchronize();

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

  vdw_forces_att_gpu();

  vdw_forces_rep_gpu();
}

void vdw_forces_att_gpu(){
	int *dev_ibead_pair_list_att;
	int *dev_jbead_pair_list_att;
	int *dev_itype_pair_list_att;
	int *dev_jtype_pair_list_att;
	double *dev_pl_lj_nat_pdb_dist;
	double3 *dev_unc_pos;
	double3 *dev_force;
	
	int N = nil_att + 1;
	
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = nbead*sizeof(double3);
	
	cudaMalloc((void **)&dev_ibead_pair_list_att, size_int);
	cudaMalloc((void **)&dev_jbead_pair_list_att, size_int);
	cudaMalloc((void **)&dev_itype_pair_list_att, size_int);
	cudaMalloc((void **)&dev_jtype_pair_list_att, size_int);
	cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist, size_double);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	cudaMalloc((void **)&dev_force, size_double3);
	
	cudaMemcpy(dev_ibead_pair_list_att, ibead_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_pair_list_att, jbead_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_pair_list_att, itype_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_pair_list_att, jtype_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice);
	
	int threads = (int)min(N, SECTION_SIZE);
	int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	vdw_forces_att_kernel<<<blocks, threads>>>(dev_ibead_pair_list_att, dev_jbead_pair_list_att, dev_itype_pair_list_att, dev_jtype_pair_list_att, 
												dev_pl_lj_nat_pdb_dist, boxl, N, dev_unc_pos, dev_force);
												
	cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost);
										
	cudaFree(dev_ibead_pair_list_att);
	cudaFree(dev_jbead_pair_list_att);
	cudaFree(dev_itype_pair_list_att);
	cudaFree(dev_jtype_pair_list_att);
	cudaFree(dev_pl_lj_nat_pdb_dist);
	cudaFree(dev_unc_pos);
	cudaFree(dev_force);
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
	int *dev_ibead_pair_list_rep;
	int *dev_jbead_pair_list_rep;
	int *dev_itype_pair_list_rep;
	int *dev_jtype_pair_list_rep;
	double3 *dev_unc_pos;
	double3 *dev_force;
	
	int N = nil_rep + 1;
	
	int size_int = N*sizeof(int);
	int size_double3 = nbead*sizeof(double3);
	
	cudaMalloc((void **)&dev_ibead_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_jbead_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_itype_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_jtype_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	cudaMalloc((void **)&dev_force, size_double3);
	
	cudaMemcpy(dev_ibead_pair_list_rep, ibead_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_pair_list_rep, jbead_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_pair_list_rep, itype_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_pair_list_rep, jtype_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice);
	
	int threads = (int)min(N, SECTION_SIZE);
	int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	vdw_forces_rep_kernel<<<blocks, threads>>>(dev_ibead_pair_list_rep, dev_jbead_pair_list_rep, dev_itype_pair_list_rep, dev_jtype_pair_list_rep, 
												boxl, N, dev_unc_pos, dev_force);
										
										
	cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost);
	
	cudaFree(dev_ibead_pair_list_rep);
	cudaFree(dev_jbead_pair_list_rep);
	cudaFree(dev_itype_pair_list_rep);
	cudaFree(dev_jtype_pair_list_rep);
	cudaFree(dev_unc_pos);
	cudaFree(dev_force);
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

void vdw_forces_matrix_gpu()
{
	using namespace std;

	double *values_x = (double *)malloc(nil_att+1);
	double *values_y = (double *)malloc(nil_att+1);
	double *values_z = (double *)malloc(nil_att+1);

	printf("Allocate Values\n");
	fflush(stdout);

	vdw_forces_att_values_gpu(values_x, values_y, values_z);

	printf("Start Summing Forces\n");
	fflush(stdout);

	int N = nil_att+1;

	vdw_sum_forces(values_x, ibead_pair_list_att, jbead_pair_list_att, 1, N);
	vdw_sum_forces(values_y, ibead_pair_list_att, jbead_pair_list_att, 2, N);
	vdw_sum_forces(values_z, ibead_pair_list_att, jbead_pair_list_att, 3, N);

	free(values_x);
	free(values_y);
	free(values_z);


	values_x = (double *)malloc(nil_rep+1);
	values_y = (double *)malloc(nil_rep+1);
	values_z = (double *)malloc(nil_rep+1);

	vdw_forces_rep_values_gpu(values_x, values_y, values_z);

	N = nil_rep+1;

	vdw_sum_forces(values_x, ibead_pair_list_rep, jbead_pair_list_rep, 1, N);
	vdw_sum_forces(values_y, ibead_pair_list_rep, jbead_pair_list_rep, 2, N);
	vdw_sum_forces(values_z, ibead_pair_list_rep, jbead_pair_list_rep, 3, N);

	free(values_x);
	free(values_y);
	free(values_z);
}

void vdw_forces_att_values_gpu(double *values_x, double *values_y, double *values_z){
	printf("Attractive Values\n");
	fflush(stdout);
	int *dev_ibead_pair_list_att;
	int *dev_jbead_pair_list_att;
	int *dev_itype_pair_list_att;
	int *dev_jtype_pair_list_att;
	double *dev_pl_lj_nat_pdb_dist;
	double3 *dev_unc_pos;
	
	double *dev_values_x;
	double *dev_values_y;
	double *dev_values_z;
	
	int N = nil_att + 1;
	
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);

	printf("\tCUDA Malloc\n");
	fflush(stdout);
	
	cudaMalloc((void **)&dev_ibead_pair_list_att, size_int);
	cudaMalloc((void **)&dev_jbead_pair_list_att, size_int);
	cudaMalloc((void **)&dev_itype_pair_list_att, size_int);
	cudaMalloc((void **)&dev_jtype_pair_list_att, size_int);
	cudaMalloc((void **)&dev_pl_lj_nat_pdb_dist, size_double);
	cudaMalloc((void **)&dev_values_x, size_double);
	cudaMalloc((void **)&dev_values_y, size_double);
	cudaMalloc((void **)&dev_values_z, size_double);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	
	printf("\tCUDA Memcpy\n");
	fflush(stdout);

	cudaMemcpy(dev_ibead_pair_list_att, ibead_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_pair_list_att, jbead_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_pair_list_att, itype_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_pair_list_att, jtype_pair_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	
	int threads = (int)min(N, SECTION_SIZE);
	int blocks = (int)ceil(1.0*N/SECTION_SIZE);

	printf("\tStart Attractive Value Kernel\n");
	fflush(stdout);
	
	vdw_forces_att_values_kernel<<<blocks, threads>>>(dev_ibead_pair_list_att, dev_jbead_pair_list_att, dev_itype_pair_list_att, dev_jtype_pair_list_att, 
												dev_pl_lj_nat_pdb_dist, boxl, N, dev_unc_pos, dev_values_x, dev_values_y, dev_values_z);

	printf("\tEnd Attractive Value Kernel\n");
	fflush(stdout);

	cudaMemcpy(values_x, dev_values_x, size_double, cudaMemcpyDeviceToHost);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	cudaMemcpy(values_y, dev_values_y, size_double, cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	cudaMemcpy(values_z, dev_values_z, size_double, cudaMemcpyDeviceToHost);
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
										
	cudaFree(dev_ibead_pair_list_att);
	cudaFree(dev_jbead_pair_list_att);
	cudaFree(dev_itype_pair_list_att);
	cudaFree(dev_jtype_pair_list_att);
	cudaFree(dev_pl_lj_nat_pdb_dist);
	cudaFree(dev_unc_pos);
	cudaFree(dev_values_x);
	cudaFree(dev_values_y);
	cudaFree(dev_values_z);
}

__global__ void vdw_forces_att_values_kernel(int *dev_ibead_pair_list_att, int *dev_jbead_pair_list_att, int *dev_itype_pair_list_att, int *dev_jtype_pair_list_att, 
								double *dev_pl_lj_nat_pdb_dist, double boxl, int N, double3 *dev_unc_pos, double *dev_values_x, double *dev_values_y, double *dev_values_z){
									
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
		
		dev_values_x[i] = fx;
		dev_values_y[i] = fy;
		dev_values_z[i] = fz;
		
	}else if(i == 0){
		dev_values_x[i] = 0;
		dev_values_y[i] = 0;
		dev_values_z[i] = 0;
	}
}

void vdw_forces_rep_values_gpu(double *values_x, double *values_y, double *values_z){
	int *dev_ibead_pair_list_rep;
	int *dev_jbead_pair_list_rep;
	int *dev_itype_pair_list_rep;
	int *dev_jtype_pair_list_rep;
	double3 *dev_unc_pos;
	double3 *dev_force;
	
	double *dev_values_x;
	double *dev_values_y;
	double *dev_values_z;
	
	int N = nil_rep + 1;
	
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double3);
	int size_double3 = (nbead+1)*sizeof(double);
	
	cudaMalloc((void **)&dev_ibead_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_jbead_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_itype_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_jtype_pair_list_rep, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	cudaMalloc((void **)&dev_force, size_double3);
	
	cudaMalloc((void **)&dev_values_x, size_double);
	cudaMalloc((void **)&dev_values_y, size_double);
	cudaMalloc((void **)&dev_values_z, size_double);
	
	cudaMemcpy(dev_ibead_pair_list_rep, ibead_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_pair_list_rep, jbead_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_pair_list_rep, itype_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_pair_list_rep, jtype_pair_list_rep, size_int, cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice);

	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	
	int threads = (int)min(N, SECTION_SIZE);
	int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	vdw_forces_rep_values_kernel<<<blocks, threads>>>(dev_ibead_pair_list_rep, dev_jbead_pair_list_rep, dev_itype_pair_list_rep, dev_jtype_pair_list_rep, 
												boxl, N, dev_unc_pos, dev_values_x, dev_values_y, dev_values_z);
										
	
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	cudaMemcpy(values_x, dev_values_x, size_double, cudaMemcpyDeviceToHost);
	cudaMemcpy(values_y, dev_values_y, size_double, cudaMemcpyDeviceToHost);
	cudaMemcpy(values_z, dev_values_z, size_double, cudaMemcpyDeviceToHost);
	
	cudaFree(dev_ibead_pair_list_rep);
	cudaFree(dev_jbead_pair_list_rep);
	cudaFree(dev_itype_pair_list_rep);
	cudaFree(dev_jtype_pair_list_rep);
	cudaFree(dev_unc_pos);
	cudaFree(dev_values_x);
	cudaFree(dev_values_y);
	cudaFree(dev_values_z);
}

__global__ void vdw_forces_rep_values_kernel(int *dev_ibead_pair_list_rep, int *dev_jbead_pair_list_rep, int *dev_itype_pair_list_rep, int *dev_jtype_pair_list_rep, double boxl, int N,
								double3 *dev_unc_pos, double *dev_values_x, double *dev_values_y, double *dev_values_z){
									
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

		dev_values_x[i] = fx;
		dev_values_y[i] = fy;
		dev_values_z[i] = fz;

	}else if(i == 0){
		dev_values_x[i] = 0;
		dev_values_y[i] = 0;
		dev_values_z[i] = 0;
	}
}

void vdw_sum_forces(double *values, int *ibead, int *jbead, int direction, int N){	
	int A_num_rows = nbead+1;
	int A_num_cols = nbead+1;
	int A_num_nnz  = N;
	double alpha = 1.0;
	double beta  = 0.0;
		
	int   *dev_ibead, *dev_jbead;
	double *dev_values, *dX, *dY;

	printf("\tAllocate Device Arrays\n");
	fflush(stdout);

	cudaMalloc((void**) &dev_ibead,  A_num_nnz*sizeof(int));
	cudaMalloc((void**) &dev_jbead,  A_num_nnz * sizeof(int));
	cudaMalloc((void**) &dev_values, A_num_nnz * sizeof(double));
	cudaMalloc((void**) &dX,         A_num_cols * sizeof(double));
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	cudaMalloc((void**) &dY,         A_num_rows * sizeof(double));

	printf("\tCUDA Memcpy Device Arrays\n");
	fflush(stdout);

	cudaMemcpy(dev_ibead, ibead, A_num_nnz * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_jbead, jbead, A_num_nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, A_num_nnz * sizeof(double), cudaMemcpyHostToDevice);
    
	
	
	int threads = (int)min(A_num_cols, SECTION_SIZE);
	int blocks = (int)ceil(1.0*A_num_cols/SECTION_SIZE);

	printf("\tFill dX Kernel\n");
	fflush(stdout);
	
	//fill_with<<<blocks, threads>>>(dX, A_num_cols, 1.0);
	cudaMemset(dX, 1.0, A_num_cols * sizeof(double));
	
	// check for error
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	
	printf("\tBegin Matrix Operations\n");
	fflush(stdout);

	cusparseHandle_t     handle = 0;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	void*  dBuffer    = NULL;
	size_t bufferSize = 0;
	cusparseCreate(&handle);
	// Create sparse matrix A in Coo format
	printf("\tCreate sparse matrix A in Coo format\n");
	fflush(stdout);
	cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_num_nnz, dev_ibead, dev_jbead, dev_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	// Create dense vector X
	printf("\tCreate dense vector X\n");
	fflush(stdout);
	cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F);
	// Create dense vector y
	printf("\tCreate dense vector y\n");
	fflush(stdout);
	cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F);
	// Allocate an external buffer if needed
	printf("\tAllocate an external buffer if needed\n");
	fflush(stdout);
	cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);

	printf("\tAllocate buffer\n");
	fflush(stdout);

	cudaMalloc(&dBuffer, bufferSize);

	// Execute SpMV

	printf("\tExecute SpMV\n");
	fflush(stdout);
	cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);

	// destroy matrix/vector descriptors
	cusparseDestroySpMat(matA);
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
	cusparseDestroy(handle);
	
	
	int size_double3 = nbead*sizeof(double3);
	
	double3 *dev_force;
	
	cudaMalloc((void **)&dev_force, size_double3);
	
	cudaMemcpy(dev_force, force, size_double3, cudaMemcpyHostToDevice);
	
	threads = (int)min(A_num_rows, SECTION_SIZE);
	blocks = (int)ceil(1.0*A_num_rows/SECTION_SIZE);
	
	vdw_forces_kernel<<<blocks, threads>>>(dY, A_num_rows, dev_force, direction);
	
	
	
	
	threads = (int)min(A_num_cols, SECTION_SIZE);
	blocks = (int)ceil(1.0*A_num_cols/SECTION_SIZE);
	
	//fill_with<<<blocks, threads>>>(dX, A_num_cols, -1.0);
	
	cudaMemset(dX, -1.0, A_num_cols * sizeof(double));
	
	handle = 0;
	dBuffer    = NULL;
	bufferSize = 0;
	cusparseCreate(&handle);
	// Create sparse matrix A in CSR format
	cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_num_nnz, dev_jbead, dev_ibead, dev_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	// Create dense vector X
	cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F);
	// Create dense vector y
	cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F);
	// allocate an external buffer if needed
	cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
	cudaMalloc(&dBuffer, bufferSize);

	// execute SpMV
	cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);

	// destroy matrix/vector descriptors
	cusparseDestroySpMat(matA);
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
	cusparseDestroy(handle);
	
	
	size_double3 = (nbead+1)*sizeof(double3);
	
	threads = (int)min(A_num_rows, SECTION_SIZE);
	blocks = (int)ceil(1.0*A_num_rows/SECTION_SIZE);
	
	vdw_forces_kernel<<<blocks, threads>>>(dY, A_num_rows, dev_force, direction);
	
	
	
	
	cudaMemcpy(force, dev_force, size_double3, cudaMemcpyDeviceToHost);	
	
	cudaFree(dX);
	cudaFree(dY);
	cudaFree(dev_ibead);
	cudaFree(dev_jbead);
	cudaFree(dev_force);
	cudaFree(dev_values);
}

__global__ void vdw_forces_kernel(double *dY, int size, double3 *dev_force, int direction){
	// direction=1 -> x
	// direction=2 -> y
	// direction=3 -> z
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i > 0 && i < size){
		if(dY[i] != 0){
			if(direction == 1){
				dev_force[i].x+= dY[i];
			}else if(direction == 2){
				dev_force[i].y+= dY[i];
			}else if(direction == 3){
				dev_force[i].z+= dY[i];
			}
		}
	}
}

__global__ void fill_with(double *dX, int size, double val){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i < size){
		dX[i] = val;
	}
}