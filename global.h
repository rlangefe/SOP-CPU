#ifndef GLOBAL_H
#define GLOBAL_H

#include "random_generator.h"

class coord {
 public:
  coord();
  ~coord();
  double x;
  double y;
  double z;
};

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

// bonded info
extern double k_bnd;
extern int nbnd;
extern int* ibead_bnd;
extern int* jbead_bnd;
extern double* pdb_dist;
extern int bnds_allocated;
extern double e_bnd;
extern double e_bnd_coeff;
extern double R0;
extern double R0sq;

// angular info

extern double k_ang;
extern int nang;
extern int* ibead_ang;
extern int* jbead_ang;
extern int* kbead_ang;
extern double* pdb_ang;
extern int angs_allocated;
extern double e_ang;
extern double e_ang_coeff;
extern double e_ang_ss;
extern double e_ang_ss_coeff;
extern double f_ang_ss_coeff;

// rna-rna vdw info

extern int ncon_att; // number of native contacts
extern int ncon_rep; // repulisve non-native contact

// neighbor list
extern int nnl_att;
extern int nnl_rep;

// pair list
extern int nil_att;
extern int nil_rep;
extern double coeff_att[][3];
extern double coeff_rep[][3];
extern double force_coeff_att[][3];
extern double force_coeff_rep[][3];
extern double sigma_rep[][3];
extern double sigma_rep2[][3];
extern double sigma_rep6[][3];
extern double sigma_rep12[][3];

extern double rcut_nat[][3];
extern int* ibead_lj_nat;
extern int* jbead_lj_nat;
extern int* itype_lj_nat;
extern int* jtype_lj_nat;
extern double* lj_nat_pdb_dist;
extern double* lj_nat_pdb_dist2; // 2nd power of the pdb distance
extern double* lj_nat_pdb_dist6; // 6th power of the pdb distance
extern double* lj_nat_pdb_dist12; // 12th power of the pdb distance
extern int* ibead_lj_non_nat;
extern int* jbead_lj_non_nat;
extern int* itype_lj_non_nat;
extern int* jtype_lj_non_nat;

// neighbor / cell list
extern int* ibead_neighbor_list_att;
extern int* jbead_neighbor_list_att;
extern int* itype_neighbor_list_att;
extern int* jtype_neighbor_list_att;

extern double* nl_lj_nat_pdb_dist;
extern double* nl_lj_nat_pdb_dist2;
extern double* nl_lj_nat_pdb_dist6;
extern double* nl_lj_nat_pdb_dist12;

extern int* ibead_neighbor_list_rep;
extern int* jbead_neighbor_list_rep;
extern int* itype_neighbor_list_rep;
extern int* jtype_neighbor_list_rep;

// pair list
extern int* ibead_pair_list_att;
extern int* jbead_pair_list_att;
extern int* itype_pair_list_att;
extern int* jtype_pair_list_att;

extern double* pl_lj_nat_pdb_dist;
extern double* pl_lj_nat_pdb_dist2;
extern double* pl_lj_nat_pdb_dist6;
extern double* pl_lj_nat_pdb_dist12;

extern int* ibead_pair_list_rep;
extern int* jbead_pair_list_rep;
extern int* itype_pair_list_rep;
extern int* jtype_pair_list_rep;

extern int lj_rna_rna_allocated;
extern int* switch_fnb;
extern double e_vdw_rr;
extern double e_vdw_rr_att;
extern double e_vdw_rr_rep;

// coordinates and associated params

extern int nbead;
extern int ncrowder;
extern int nbead_tot;
extern double3* pos;
extern double3* unc_pos;
extern double3* vel;
extern double3* force;
extern double3* natpos; // native position vectors
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
extern double T; // temperature (kcal/mol)
extern int usegpu_nl;
extern int usegpu_pl;
extern int usegpu_vdw_energy;
extern int usegpu_vdw_force;
extern int usegpu_ss_ang_energy;
extern int usegpu_ss_ang_force;
extern int usegpu_fene_energy;
extern int usegpu_fene_force;
extern int neighborlist; // neighbor list cutoff method?
extern int celllist; // cell list cutoff method?
extern double minT; // minimum temperature determines crowder cutoffs
extern double boxl; // Length of an edge of the simulation box
extern double ncell;
extern double lcell;
extern double zeta; // friction coefficient
extern double nstep; // number of steps to take
extern double istep_restart; // which step to we restart from?
extern int nup;
extern int inlup;
extern int nnlup;
extern double h; // time step
extern double halfh;
extern double a1; // a1,a2,a3,a4 are used for integration
extern double a2;
extern double a3;
extern double a4;
extern double a5;
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
extern double rna_etot;
extern double system_etot;

/* force stuff */
extern int nforce_term; // number of force terms
const int mforce_term = 10; // max number of force terms
extern int force_term_on[]; // is a particular force term on?
typedef void (*force_term_Ptr) ();
extern force_term_Ptr force_term[]; // array of pointers to functions -- each elements is for evaluating a particular type of force

// observables
extern double chi;
extern double Q;
extern int contct_nat;
extern int contct_tot;
extern double end2endsq;
extern double rgsq;
extern double kinT;

extern double sigma_ss;
extern double sigma_ss6;
extern double epsilon_ss;
extern double e_bnd,e_ang,e_tor,e_stack,e_elec,e_ang_ss;
extern double e_vdw_rr,e_vdw_rr_att,e_vdw_rr_rep;
extern double rna_etot,system_etot;


// native info

extern int* rna_base;
extern int rna_base_allocated;
extern int* rna_phosphate;
extern int rna_phosphate_allocated;

// conversion factors;
const double kcalpmol2K = 503.15;

double rnd(double);

#endif /* GLOBAL_H */
