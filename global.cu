
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
const int debug = 0;

int ncmd;
char cmd[mcmd+1][mwdsize];
char opt[mopt_tot+1][mwdsize];
int opt_ptr[mcmd+1];
char pathname[MAXPATHLEN];
char nl_algorithm[30];
char pl_algorithm[30];

// bonded info
double k_bnd; // bond spring constant
int nbnd; // number of bonds
int* ibead_bnd;
int* jbead_bnd;
double* pdb_dist; // pdb bond distances
int bnds_allocated = 0;
double R0;
double R0sq;
double e_bnd_coeff;

// angular info
double k_ang;
int nang;
int* ibead_ang;
int* jbead_ang;
int* kbead_ang;
double* pdb_ang;
int angs_allocated = 0;
double e_ang_coeff;
double e_ang_ss_coeff;
double f_ang_ss_coeff;

// rna-rna vdw

int ncon_att; // number of native contacts
int ncon_rep; // repulisve non-native contact

// neighbor list
int nnl_att;
int nnl_rep;

// pair list
int nil_att;
int nil_rep;


double coeff_att[3][3] = { {0.0, 0.0, 0.0},
			   {0.0, 0.7, 0.8},
			   {0.0, 0.8, 1.0} };

double coeff_rep[3][3] = { {0.0, 0.0, 0.0},
			   {0.0, 1.0, 1.0},
			   {0.0, 1.0, 1.0} };

double force_coeff_att[3][3] = { {0.0,       0.0,       0.0},
				 {0.0, -12.0*1.0, -12.0*0.8},
				 {0.0, -12.0*0.8, -12.0*0.7} };

double force_coeff_rep[3][3] = { {0.0,       0.0,       0.0},
				 {0.0,  -6.0*1.0,  -6.0*1.0},
				 {0.0,  -6.0*1.0,  -6.0*1.0} };

double sigma_rep[3][3] = { {0.0, 0.0, 0.0},
			   {0.0, 3.8, 5.4},
			   {0.0, 5.4, 7.0} };

double sigma_rep2[3][3];
double sigma_rep6[3][3];
double sigma_rep12[3][3];

//double sigma_rep;
//double sigma_rep2;
double sigma_ss; // for angular soft-sphere repulsion
double sigma_ss6; // for angular soft-sphere repulsion
double epsilon_ss; // for angular soft-sphere repulsion
//double force_coeff_rep;
double rcut_nat[3][3] = { { 0.0,  0.0,  0.0},
                          { 0.0,  8.0, 11.0},
                          { 0.0, 11.0, 14.0} };
int* ibead_lj_nat;
int* jbead_lj_nat;
int* itype_lj_nat;
int* jtype_lj_nat;

double* lj_nat_pdb_dist;
double* lj_nat_pdb_dist2;
double* lj_nat_pdb_dist6;
double* lj_nat_pdb_dist12;

int* ibead_lj_non_nat;
int* jbead_lj_non_nat;
int* itype_lj_non_nat;
int* jtype_lj_non_nat;

// neighbor / cell list
int* ibead_neighbor_list_att;
int* jbead_neighbor_list_att;
int* itype_neighbor_list_att;
int* jtype_neighbor_list_att;

double* nl_lj_nat_pdb_dist;
double* nl_lj_nat_pdb_dist2;
double* nl_lj_nat_pdb_dist6;
double* nl_lj_nat_pdb_dist12;

int* ibead_neighbor_list_rep;
int* jbead_neighbor_list_rep;
int* itype_neighbor_list_rep;
int* jtype_neighbor_list_rep;

// pair list
int* ibead_pair_list_att;
int* jbead_pair_list_att;
int* itype_pair_list_att;
int* jtype_pair_list_att;

double* pl_lj_nat_pdb_dist;
double* pl_lj_nat_pdb_dist2;
double* pl_lj_nat_pdb_dist6;
double* pl_lj_nat_pdb_dist12;

int* ibead_pair_list_rep;
int* jbead_pair_list_rep;
int* itype_pair_list_rep;
int* jtype_pair_list_rep;

int lj_rna_rna_allocated = 0;

// coordinates and associated params

int nbead;
double3* pos;
double3* unc_pos; // uncorrected positions
double3* vel;
double3* force;
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
double T; // temperature
int usegpu_nl = 0;
int usegpu_pl = 0;
int usegpu_vdw_energy = 0;
int usegpu_vdw_force = 0;
int usegpu_ss_ang_energy = 0;
int usegpu_ss_ang_force = 0;
int usegpu_fene_energy = 0;
int usegpu_fene_force = 0;
int neighborlist = 0; // neighbor list cutoff method?
int celllist = 0; // cell list cutoff method?
double boxl; // Length of an edge of the simulation box
double ncell;
double lcell;
double zeta; // friction coefficient
double nstep; // number of steps to take
double istep_restart = 0.0;
int nup;
int inlup;
int nnlup;
double h; // time step
double halfh;
double a1; // a1,a2,a3,a4,a5 are used for integration
double a2;
double a3;
double a4;
double a5;
char ufname[mwdsize+1];
char rcfname[mwdsize+1];
char cfname[mwdsize+1];
char unccfname[mwdsize+1];
char vfname[mwdsize+1];
char binfname[mwdsize+1];
char uncbinfname[mwdsize+1];
char iccnfigfname[mwdsize+1];
int binsave = 1; // default will save trajectory
const double pi = acos(-1);

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

double e_bnd,e_ang,e_tor,e_stack,e_elec,e_ang_ss;
double e_vdw_rr,e_vdw_rr_att,e_vdw_rr_rep;
double rna_etot,system_etot;
double chi;
double Q;
int contct_nat;
int contct_tot;
double end2endsq;
double rgsq;
double kinT;

double rnd(double x)
{
  using namespace std;
  return ( (x>0) ? floor(x+0.5) : ceil(x-0.5) );
}
