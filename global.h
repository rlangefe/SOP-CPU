#ifndef GLOBAL_H
#define GLOBAL_H

#include "random_generator.h"

class coord {
 public:
  coord();
  ~coord();
  float x;
  float y;
  float z;
};

extern int debug;

// Benchmarking
extern float nl_time;
extern float pl_time;
extern float vdw_energy_time;
extern float vdw_forces_time;
extern float fene_energy_time;
extern float fene_forces_time;
extern float ss_ang_energy_time;
extern float ss_ang_forces_time;
extern float update_pos_time;
extern float update_vel_time;
extern float clear_forces_time;
extern float rng_time;

const int mcmd = 100; // maximum number of input commands
const int mopt = 10; // maximum number of options associated with a command
const int mopt_tot = mcmd*mopt; // max total number of options
const int mwdsize = 1024; // maximum number of characters in a word
const size_t MAXPATHLEN = 2048;

extern char cmd[][mwdsize];
extern char opt[][mwdsize]; // holds the options
extern int opt_ptr[]; // holds index of first option correspond to given cmd
extern int ncmd; // number of input commands
extern int nopt_tot; // total # of options
extern char pathname[];
extern char nl_algorithm[];
extern char pl_algorithm[];
extern int prec;

extern int variable_location[];

// bonded info
extern float k_bnd;
extern int nbnd;
extern int* ibead_bnd;
extern int* jbead_bnd;
extern float* pdb_dist;
extern int bnds_allocated;
extern float e_bnd;
extern float e_bnd_coeff;
extern float R0;
extern float R0sq;

// angular info

extern float k_ang;
extern int nang;
extern int* ibead_ang;
extern int* jbead_ang;
extern int* kbead_ang;
extern float* pdb_ang;
extern int angs_allocated;
extern float e_ang;
extern float e_ang_coeff;
extern float e_ang_ss;
extern float e_ang_ss_coeff;
extern float f_ang_ss_coeff;

// rna-rna vdw info

extern int ncon_att; // number of native contacts
extern int ncon_rep; // repulisve non-native contact

// neighbor list
extern int nnl_att;
extern int nnl_rep;

// pair list
extern int nil_att;
extern int nil_rep;
extern float coeff_att[][3];
extern float coeff_rep[][3];
extern float force_coeff_att[][3];
extern float force_coeff_rep[][3];
extern float sigma_rep[][3];
extern float sigma_rep2[][3];
extern float sigma_rep6[][3];
extern float sigma_rep12[][3];

extern float rcut_nat[][3];
extern int* ibead_lj_nat;
extern int* jbead_lj_nat;
extern int* itype_lj_nat;
extern int* jtype_lj_nat;
extern float* lj_nat_pdb_dist;
extern float* lj_nat_pdb_dist2; // 2nd power of the pdb distance
extern float* lj_nat_pdb_dist6; // 6th power of the pdb distance
extern float* lj_nat_pdb_dist12; // 12th power of the pdb distance
extern int* ibead_lj_non_nat;
extern int* jbead_lj_non_nat;
extern int* itype_lj_non_nat;
extern int* jtype_lj_non_nat;

// neighbor / cell list
extern int* ibead_neighbor_list_att;
extern int* jbead_neighbor_list_att;
extern int* itype_neighbor_list_att;
extern int* jtype_neighbor_list_att;

extern float* nl_lj_nat_pdb_dist;
extern float* nl_lj_nat_pdb_dist2;
extern float* nl_lj_nat_pdb_dist6;
extern float* nl_lj_nat_pdb_dist12;

extern int* ibead_neighbor_list_rep;
extern int* jbead_neighbor_list_rep;
extern int* itype_neighbor_list_rep;
extern int* jtype_neighbor_list_rep;

// pair list
extern int* ibead_pair_list_att;
extern int* jbead_pair_list_att;
extern int* itype_pair_list_att;
extern int* jtype_pair_list_att;

extern float* pl_lj_nat_pdb_dist;
extern float* pl_lj_nat_pdb_dist2;
extern float* pl_lj_nat_pdb_dist6;
extern float* pl_lj_nat_pdb_dist12;

extern int* ibead_pair_list_rep;
extern int* jbead_pair_list_rep;
extern int* itype_pair_list_rep;
extern int* jtype_pair_list_rep;

extern int lj_rna_rna_allocated;
extern int* switch_fnb;
extern float e_vdw_rr;
extern float e_vdw_rr_att;
extern float e_vdw_rr_rep;

// coordinates and associated params

extern int nbead;
extern int ncrowder;
extern int nbead_tot;
extern float3* pos;
extern float3* unc_pos;
extern float3* vel;
extern float3* force;
extern float3* natpos; // native position vectors
extern int pos_allocated;
extern int unc_pos_allocated;
extern int vel_allocated;
extern int force_allocated;
extern int natpos_allocated;

// miscellaneous run paramaters;

extern int run;
extern Ran_Gen generator; // the random number generator
extern int restart; // are we restarting an old simulation?
extern int rgen_restart; // should we restart the random number generator?
extern int sim_type; // integration scheme 1 = underdamped; 2 = overdamped
extern float T; // temperature (kcal/mol)
extern int usegpu_nl;
extern int usegpu_pl;
extern int usegpu_vdw_energy;
extern int usegpu_vdw_force;
extern int usegpu_ss_ang_energy;
extern int usegpu_ss_ang_force;
extern int usegpu_fene_energy;
extern int usegpu_fene_force;
extern int usegpu_pos;
extern int usegpu_vel;
extern int usegpu_rand_force;
extern int usegpu_clear_force;
extern int neighborlist; // neighbor list cutoff method?
extern int celllist; // cell list cutoff method?
extern float minT; // minimum temperature determines crowder cutoffs
extern float boxl; // Length of an edge of the simulation box
extern float ncell;
extern float lcell;
extern float zeta; // friction coefficient
extern float nstep; // number of steps to take
extern float istep_restart; // which step to we restart from?
extern int nup;
extern int inlup;
extern int nnlup;
extern float h; // time step
extern float halfh;
extern float a1; // a1,a2,a3,a4 are used for integration
extern float a2;
extern float a3;
extern float a4;
extern float a5;
extern char ufname[];
extern char rcfname[];
extern char cfname[];
extern char unccfname[];
extern char vfname[];
extern char binfname[];
extern char uncbinfname[];
extern char iccnfigfname[];
extern int binsave;

/* potential stuff */
extern int npot_term;// number of terms in the potential
const int mpot_term = 10;// max number of terms in potential
extern int pot_term_on[];// is a particular term in the potential on?
typedef void (*pot_term_Ptr) ();
/* array of pointers to functions;
   each element is for evaluating a
   particular term in the potential */
extern pot_term_Ptr pot_term[];
extern float rna_etot;
extern float system_etot;

/* force stuff */
extern int nforce_term; // number of force terms
const int mforce_term = 10; // max number of force terms
extern int force_term_on[]; // is a particular force term on?
typedef void (*force_term_Ptr) ();
extern force_term_Ptr force_term[]; // array of pointers to functions -- each elements is for evaluating a particular type of force

// observables
extern float chi;
extern float Q;
extern int contct_nat;
extern int contct_tot;
extern float end2endsq;
extern float rgsq;
extern float kinT;

extern float sigma_ss;
extern float sigma_ss6;
extern float epsilon_ss;
extern float e_bnd,e_ang,e_tor,e_stack,e_elec,e_ang_ss;
extern float e_vdw_rr,e_vdw_rr_att,e_vdw_rr_rep;
extern float rna_etot,system_etot;

// Increment
extern float3 *incr;


// native info

extern int* rna_base;
extern int rna_base_allocated;
extern int* rna_phosphate;
extern int rna_phosphate_allocated;

// conversion factors;
const float kcalpmol2K = 503.15;

float rnd(float);

#endif /* GLOBAL_H */
