
#include <cstdlib>
#include <cstdio>
#include "global.h"
#include <math.h>

coord::coord() {
  x = 0.0;
  y = 0.0;
  z = 0.0;
}

coord::~coord() {
}
int debug = 0;

// Benchmarking
float nl_time = 0.0;
float pl_time = 0.0;
float vdw_energy_time = 0.0;
float vdw_forces_time = 0.0;
float fene_energy_time = 0.0;
float fene_forces_time = 0.0;
float ss_ang_energy_time = 0.0;
float ss_ang_forces_time = 0.0;
float update_pos_time = 0.0;
float update_vel_time = 0.0;
float clear_forces_time = 0.0;
float rng_time = 0.0;

int ncmd;
char cmd[mcmd+1][mwdsize];
char opt[mopt_tot+1][mwdsize];
int opt_ptr[mcmd+1];
char pathname[MAXPATHLEN];
char nl_algorithm[30];
char pl_algorithm[30];
int prec = 0;

// bonded info
float k_bnd; // bond spring constant
int nbnd; // number of bonds
int* ibead_bnd;
int* jbead_bnd;
float* pdb_dist; // pdb bond distances
int bnds_allocated = 0;
float R0;
float R0sq;
float e_bnd_coeff;

// angular info
float k_ang;
int nang;
int* ibead_ang;
int* jbead_ang;
int* kbead_ang;
float* pdb_ang;
int angs_allocated = 0;
float e_ang_coeff;
float e_ang_ss_coeff;
float f_ang_ss_coeff;

// rna-rna vdw

int ncon_att; // number of native contacts
int ncon_rep; // repulisve non-native contact

// neighbor list
int nnl_att;
int nnl_rep;

// pair list
int nil_att;
int nil_rep;


float coeff_att[3][3] = { {0.0, 0.0, 0.0},
			   {0.0, 0.7, 0.8},
			   {0.0, 0.8, 1.0} };

float coeff_rep[3][3] = { {0.0, 0.0, 0.0},
			   {0.0, 1.0, 1.0},
			   {0.0, 1.0, 1.0} };

float force_coeff_att[3][3] = { {0.0,       0.0,       0.0},
				 {0.0, -12.0*1.0, -12.0*0.8},
				 {0.0, -12.0*0.8, -12.0*0.7} };

float force_coeff_rep[3][3] = { {0.0,       0.0,       0.0},
				 {0.0,  -6.0*1.0,  -6.0*1.0},
				 {0.0,  -6.0*1.0,  -6.0*1.0} };

float sigma_rep[3][3] = { {0.0, 0.0, 0.0},
			   {0.0, 3.8, 5.4},
			   {0.0, 5.4, 7.0} };

float sigma_rep2[3][3];
float sigma_rep6[3][3];
float sigma_rep12[3][3];

//float sigma_rep;
//float sigma_rep2;
float sigma_ss; // for angular soft-sphere repulsion
float sigma_ss6; // for angular soft-sphere repulsion
float epsilon_ss; // for angular soft-sphere repulsion
//float force_coeff_rep;
float rcut_nat[3][3] = { { 0.0,  0.0,  0.0},
                          { 0.0,  8.0, 11.0},
                          { 0.0, 11.0, 14.0} };
int* ibead_lj_nat;
int* jbead_lj_nat;
int* itype_lj_nat;
int* jtype_lj_nat;

float* lj_nat_pdb_dist;
float* lj_nat_pdb_dist2;
float* lj_nat_pdb_dist6;
float* lj_nat_pdb_dist12;

int* ibead_lj_non_nat;
int* jbead_lj_non_nat;
int* itype_lj_non_nat;
int* jtype_lj_non_nat;

// neighbor / cell list
int* ibead_neighbor_list_att;
int* jbead_neighbor_list_att;
int* itype_neighbor_list_att;
int* jtype_neighbor_list_att;

float* nl_lj_nat_pdb_dist;
float* nl_lj_nat_pdb_dist2;
float* nl_lj_nat_pdb_dist6;
float* nl_lj_nat_pdb_dist12;

int* ibead_neighbor_list_rep;
int* jbead_neighbor_list_rep;
int* itype_neighbor_list_rep;
int* jtype_neighbor_list_rep;

// pair list
int* ibead_pair_list_att;
int* jbead_pair_list_att;
int* itype_pair_list_att;
int* jtype_pair_list_att;

float* pl_lj_nat_pdb_dist;
float* pl_lj_nat_pdb_dist2;
float* pl_lj_nat_pdb_dist6;
float* pl_lj_nat_pdb_dist12;

int* ibead_pair_list_rep;
int* jbead_pair_list_rep;
int* itype_pair_list_rep;
int* jtype_pair_list_rep;

int lj_rna_rna_allocated = 0;

// coordinates and associated params

int nbead;
float3* pos;
float3* unc_pos; // uncorrected positions
float3* vel;
float3* force;
int pos_allocated = 0;
int vel_allocated = 0;
int force_allocated = 0;
int unc_pos_allocated = 0;

// native info

int* rna_base; // array which indicates whether or not a bead is a base
int rna_base_allocated;
int* rna_phosphate;
int rna_phosphate_allocated;

// miscellaneous run paramaters;

Ran_Gen generator; // random number generator
int run;
int restart = 0; // default is to start a new simulation
int rgen_restart = 0; // default don't restart random number generator
int sim_type = 1; // integration scheme; default is underdamped
float T; // temperature
int usegpu_nl = 0;
int usegpu_pl = 0;
int usegpu_vdw_energy = 0;
int usegpu_vdw_force = 0;
int usegpu_ss_ang_energy = 0;
int usegpu_ss_ang_force = 0;
int usegpu_fene_energy = 0;
int usegpu_fene_force = 0;
int usegpu_pos = 0;
int usegpu_vel = 0;
int usegpu_rand_force = 0;
int usegpu_clear_force = 0;
int neighborlist = 0; // neighbor list cutoff method?
int celllist = 0; // cell list cutoff method?
float boxl; // Length of an edge of the simulation box
float ncell;
float lcell;
float zeta; // friction coefficient
float nstep; // number of steps to take
float istep_restart = 0.0;
int nup;
int inlup;
int nnlup;
float h; // time step
float halfh;
float a1; // a1,a2,a3,a4,a5 are used for integration
float a2;
float a3;
float a4;
float a5;
char ufname[mwdsize+1];
char rcfname[mwdsize+1];
char cfname[mwdsize+1];
char unccfname[mwdsize+1];
char vfname[mwdsize+1];
char binfname[mwdsize+1];
char uncbinfname[mwdsize+1];
char iccnfigfname[mwdsize+1];
int binsave = 1; // default will save trajectory
const float pi = acos(-1);

// force and pot stuff

int nforce_term = 4; // ran,bnds,angs,vdw -- default is that tension is off
int force_term_on[mforce_term+1] = { 0, 1, 1, 1, 0,
				     0, 1, 0, 0, 0, 0 };
force_term_Ptr force_term[mforce_term+1];

int npot_term = 3; // bnds,angs,vdw
int pot_term_on[mpot_term+1] = { 0, 1, 1, 0, 0,
				 1, 0, 0, 0, 0, 0 };
pot_term_Ptr pot_term[mpot_term+1];

//observables

float e_bnd,e_ang,e_tor,e_stack,e_elec,e_ang_ss;
float e_vdw_rr,e_vdw_rr_att,e_vdw_rr_rep;
float rna_etot,system_etot;
float chi;
float Q;
int contct_nat;
int contct_tot;
float end2endsq;
float rgsq;
float kinT;

// Increment
float3 *incr;

float rnd(float x)
{
  using namespace std;
  return ( (x>0) ? floor(x+0.5) : ceil(x-0.5) );
}

/* To whoever is maintaining this, I appologize for using variable_location.
	It is the best way to keep track of where each variable lives.
	If new GPU variables need to be added, please update this documentation
	to indicate what number matches what variable groups

	0: 	ibead_lj_nat
		jbead_lj_nat
		itype_lj_nat
		jtype_lj_nat

	1:	lj_nat_pdb_dist

	2:	unc_pos

	3:	ibead_lj_non_nat
		jbead_lj_non_nat
		itype_lj_non_nat
		jtype_lj_non_nat

	4:	ibead_neighbor_list_att
		jbead_neighbor_list_att
		itype_neighbor_list_att
		jtype_neighbor_list_att

	5:	nl_lj_nat_pdb_dist
		nl_lj_nat_pdb_dist2
		nl_lj_nat_pdb_dist6
		nl_lj_nat_pdb_dist12

	6:	ibead_neighbor_list_rep
		jbead_neighbor_list_rep
		itype_neighbor_list_rep
		jtype_neighbor_list_rep

	7:	ibead_pair_list_att
		jbead_pair_list_att
		itype_pair_list_att
		jtype_pair_list_att
		pl_lj_nat_pdb_dist
		pl_lj_nat_pdb_dist2
		pl_lj_nat_pdb_dist6
		pl_lj_nat_pdb_dist12

	8: 	ibead_pair_list_rep
		jbead_pair_list_rep
		itype_pair_list_rep
		jtype_pair_list_rep

	9:	ibead_bnd
		jbead_bnd

	10:	pdb_dist

	11:	ibead_ang
		kbead_ang

	12:	force

	13: incr

	14: vel

	15: pos
	*/

int variable_location[30] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};