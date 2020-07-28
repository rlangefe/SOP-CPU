#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <iostream>
#include <fstream>
#include "params.h"
#include "io.h"
#include "global.h"

void set_params(int icmd)
{

  using namespace std;
  char oline[1024];
  int iopt;

  if( !strcmp(opt[opt_ptr[icmd]],"dynamics") ) { // set the type of simulation
    if( !strcmp(opt[opt_ptr[icmd]+1],"underdamped") ) {
      sim_type = 1; // low-friction limit for sampling
      h = 2.5e-3;
      halfh = h/2.0;
      a1 = h*(1.0-zeta*halfh);
      a2 = h*halfh;
      a3 = (1.0-h*zeta/2.0+(h*zeta)*(h*zeta)/4.0)/h;
      a4 = halfh*(1.0-h*zeta/2.0);
    } else if( !strcmp(opt[opt_ptr[icmd]+1],"overdamped") ) {
      sim_type = 2; // hi-friction limit for kinetics
      h = 0.02;
      a5 = h/zeta;
    }
  } else if( !strcmp(opt[opt_ptr[icmd]],"temp") ) { // set the temperature
    set_temp(atof(opt[opt_ptr[icmd]+1]));

  } else if( !strcmp(opt[opt_ptr[icmd]],"nstep") ) { // # of steps
    nstep = atof(opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"istep_restart") ) { // where to restart from
    istep_restart = atof(opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"nup") ) { // # of steps before an update
    nup = atoi(opt[opt_ptr[icmd]+1]);

  } else if( !strcmp(opt[opt_ptr[icmd]],"run") ) { // set current run
    run = atoi((opt[opt_ptr[icmd]+1]));
    generator.set_seed(-100-run);

  } else if( !strcmp(opt[opt_ptr[icmd]],"ufname") ) { // set update file name
    strcpy(ufname,opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"rcfname") ) { // set restart coordinate file name
    strcpy(rcfname,opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"cfname") ) { // set save coordinate file name
    strcpy(cfname,opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"rgenfname") ) { // set random generator file name
    generator.set_fname(opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"unccfname") ) { // set save coordinate file name
    strcpy(unccfname,opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"vfname") ) { // set save velocity file name
    strcpy(vfname,opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"binfname") ) { // set save trajectory file name
    strcpy(binfname,opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"uncbinfname") ) { // set save trajectory file name
    strcpy(uncbinfname,opt[opt_ptr[icmd]+1]);

  } else if( !strcmp(opt[opt_ptr[icmd]],"cutofftype") ) { // neighbor list on or off?
    if( !strcmp(opt[opt_ptr[icmd]+1],"neighborlist" ) ) { neighborlist = 1; }
    else if( !strcmp(opt[opt_ptr[icmd]+1],"celllist" ) ) { celllist = 1; }
    else { }

  } else if( !strcmp(opt[opt_ptr[icmd]],"nnlup") ) { // neighbor / cell list update frequency
    nnlup = atoi(opt[opt_ptr[icmd]+1]);

  }else if(!strcmp(opt[opt_ptr[icmd]],"debug")){
    debug = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"prec")){
    prec = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_nl")){
    usegpu_nl = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_pl")){
    usegpu_pl = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_vdw_energy")){
    usegpu_vdw_energy = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_ss_ang_energy")){
    usegpu_ss_ang_energy = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_fene_energy")){
    usegpu_fene_energy = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_vdw_force")){
    usegpu_vdw_force = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_ss_ang_force")){
    usegpu_ss_ang_force = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_fene_force")){
    usegpu_fene_force = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_pos")){
    usegpu_pos = atoi((opt[opt_ptr[icmd]+1]));

  }else if(!strcmp(opt[opt_ptr[icmd]],"usegpu_vel")){
    usegpu_vel = atoi((opt[opt_ptr[icmd]+1]));

  } else if(!strcmp(opt[opt_ptr[icmd]],"nl_algorithm")){
    strcpy(nl_algorithm, opt[opt_ptr[icmd]+1]);

  } else if(!strcmp(opt[opt_ptr[icmd]],"pl_algorithm")){
    strcpy(pl_algorithm, opt[opt_ptr[icmd]+1]);

  } else if( !strcmp(opt[opt_ptr[icmd]],"boxl") ) { // box length for pbc
    boxl = atof(opt[opt_ptr[icmd]+1]);
  } else if( !strcmp(opt[opt_ptr[icmd]],"ncell") ) { // number of cells along box length
    ncell = atof(opt[opt_ptr[icmd]+1]);
    lcell = boxl / ncell;

  } else if( !strcmp(opt[opt_ptr[icmd]],"restart") ) { // restart on or off?
    if( !strcmp(opt[opt_ptr[icmd]+1],"on" ) ) { restart = 1; }
    else { restart = 0; }
  } else if( !strcmp(opt[opt_ptr[icmd]],"rgen_restart") ) { // restart the generator?
    if( !strcmp(opt[opt_ptr[icmd]+1],"on" ) ) { rgen_restart = 1; }
    else { rgen_restart = 0; }
  } else if( !strcmp(opt[opt_ptr[icmd]],"t_step") ) {
    h = atof((opt[opt_ptr[icmd]+1]));
    halfh = h/2.0;
    a1 = h*(1.0-zeta*halfh);
    a2 = h*halfh;
    a3 = (1.0-h*zeta/2.0+(h*zeta)*(h*zeta)/4.0)/h;
    a4 = halfh*(1.0-h*zeta/2.0);
    a5 = h/zeta;
  } else if( !strcmp(opt[opt_ptr[icmd]],"zeta") ) { // friction coefficient
    if (sim_type == 1) h = 2.5e-3; else if (sim_type == 2) h = 0.02;
    zeta = atof((opt[opt_ptr[icmd]+1]));
    a1 = h*(1.0-zeta*halfh);
    a3 = (1.0-h*zeta/2.0+(h*zeta)*(h*zeta)/4.0)/h;
    a4 = halfh*(1.0-h*zeta/2.0);
    a5 = h/zeta;
  } else {};

}

void set_temp(double temp)
{
  using namespace std;

  T = temp;
}

void load(int icmd)
{
  using namespace std;

  ifstream in;
  char line[2048];
  char* tokPtr;
  int test;
  int test1,test2;
  int ncon_tot;
  int icon_att, icon_rep;
  int i,j,k,l;
  int ibead,jbead;
  int itype,jtype;
  double real_phi, ideal_phi;
  double r_ij;
  int istate;

  if( !strcmp(opt[opt_ptr[icmd]],"bonds") ) { // load bonds
    cout << "[Reading in bonds...]" << endl;
    in.clear();
    in.open(opt[opt_ptr[icmd]+1],ios::in); // open file
    in.getline(line,2048);
    tokPtr = strtok(line," ");
    tokPtr = strtok(NULL," ");
    nbnd = atoi(tokPtr); // read in number of bonds
    init_bonds(nbnd);
    for( int i=1; i<=nbnd; i++ ) {
      in.getline(line,2048);
      tokPtr = strtok(line," ");
      ibead_bnd[i] = atoi(tokPtr); // first bead index
      tokPtr = strtok(NULL," ");
      jbead_bnd[i] = atoi(tokPtr); // second bead index
      tokPtr = strtok(NULL," ");
      pdb_dist[i] = atof(tokPtr); // equilibrium distance (angstrom)
    }
    in.close(); // close file
    cout << "[Finished reading bonds (" << nbnd <<")]" << endl;
  } else if(!strcmp(opt[opt_ptr[icmd]],"angles")) { // load angles
    cout << "[Reading in angles...]" << endl;
    in.clear();
    in.open(opt[opt_ptr[icmd]+1],ios::in);
    in.getline(line,2048);
    tokPtr = strtok(line," ");
    tokPtr = strtok(NULL," ");
    nang = atoi(tokPtr); // read in number of angles
    init_angles(nang);
    for( int i=1; i<=nang; i++ ) {
      in.getline(line,2048);
      tokPtr = strtok(line," ");
      ibead_ang[i] = atoi(tokPtr); // first bead index
      tokPtr = strtok(NULL," ");
      jbead_ang[i] = atoi(tokPtr); // second bead index
      tokPtr = strtok(NULL," ");
      kbead_ang[i] = atoi(tokPtr); // third bead index
      tokPtr = strtok(NULL," ");
      pdb_ang[i] = atof(tokPtr); // equilibrium angle (radians) ; SOP -> dist between bead i,i+2
    }
    in.close();
    cout << "[Finished reading angles (" << nang <<")]" << endl;
  } else if(!strcmp(opt[opt_ptr[icmd]],"vdw")) { // load rna-rna vdw
    cout << "[Reading in VDW interactions...]" << endl;
    in.clear();
    in.open(opt[opt_ptr[icmd]+1],ios::in);
    in.getline(line,2048);
    tokPtr = strtok(line," ");
    tokPtr = strtok(NULL," ");
    ncon_att = atoi(tokPtr);
    tokPtr = strtok(NULL," ");
    tokPtr = strtok(NULL," ");
    ncon_rep = atoi(tokPtr);
    init_lj(ncon_att,ncon_rep);
    ncon_tot = ncon_att + ncon_rep;
    icon_att = 0;
    icon_rep = 0;
    for( int i=1; i<=ncon_tot; i++ ) {
      in.getline(line,2048);
      tokPtr = strtok(line," ");
      ibead = atoi(tokPtr);
      tokPtr = strtok(NULL," ");
      jbead = atoi(tokPtr);
      tokPtr = strtok(NULL," ");
      r_ij = atof(tokPtr);
      tokPtr = strtok(NULL," ");
      itype = atoi(tokPtr);
      tokPtr = strtok(NULL," ");
      jtype = atoi(tokPtr);
      if (r_ij < rcut_nat[itype][jtype]) {
	icon_att++;
	ibead_lj_nat[icon_att] = ibead;
	jbead_lj_nat[icon_att] = jbead;
	itype_lj_nat[icon_att] = itype;
	jtype_lj_nat[icon_att] = jtype;
	lj_nat_pdb_dist[icon_att] = r_ij;
	lj_nat_pdb_dist2[icon_att] = r_ij*r_ij;
	lj_nat_pdb_dist6[icon_att] = lj_nat_pdb_dist2[icon_att]*
	  lj_nat_pdb_dist2[icon_att]*lj_nat_pdb_dist2[icon_att];
	lj_nat_pdb_dist12[icon_att] = lj_nat_pdb_dist6[icon_att]*
	  lj_nat_pdb_dist6[icon_att];

	nil_att++;
	ibead_pair_list_att[nil_att] = ibead;
	jbead_pair_list_att[nil_att] = jbead;
	itype_pair_list_att[nil_att] = itype;
	jtype_pair_list_att[nil_att] = jtype;
	pl_lj_nat_pdb_dist[nil_att] = r_ij;
	pl_lj_nat_pdb_dist2[nil_att] = lj_nat_pdb_dist2[icon_att];
	pl_lj_nat_pdb_dist6[nil_att] = lj_nat_pdb_dist6[icon_att];
	pl_lj_nat_pdb_dist12[nil_att] = lj_nat_pdb_dist12[icon_att];
      } else {
	icon_rep++;
	ibead_lj_non_nat[icon_rep] = ibead;
	jbead_lj_non_nat[icon_rep] = jbead;
	itype_lj_non_nat[icon_rep] = itype;
	jtype_lj_non_nat[icon_rep] = jtype;

	nil_rep++;
	ibead_pair_list_rep[nil_rep] = ibead;
	jbead_pair_list_rep[nil_rep] = jbead;
	itype_pair_list_rep[nil_rep] = itype;
	jtype_pair_list_rep[nil_rep] = jtype;
      }
    }
    in.close();
    cout << "[Finished reading VDW interactions (" << icon_att << "/" << icon_rep <<")]" << endl;
  } else if(!strcmp(opt[opt_ptr[icmd]],"init")) { // load init coordinates
    cout << "[Reading in initial coordinates...]" << endl;
    in.clear();
    in.open(opt[opt_ptr[icmd]+1],ios::in);
    in.getline(line,2048);
    tokPtr = strtok(line," ");
    tokPtr = strtok(NULL," ");
    nbead = atoi(tokPtr); // read in number of beads
    init_pos(nbead);
    for( int i=1; i<=nbead; i++ ) {
      in.getline(line,2048);
      tokPtr = strtok(line," ");
      tokPtr = strtok(NULL," ");
      pos[i].x = atof(tokPtr);
      unc_pos[i].x = pos[i].x;
      tokPtr = strtok(NULL," ");
      pos[i].y = atof(tokPtr);
      unc_pos[i].y = pos[i].y;
      tokPtr = strtok(NULL," ");
      pos[i].z = atof(tokPtr);
      unc_pos[i].z = pos[i].z;
    }
    in.close();
    cout << "[Finished reading initial coordinates (" << nbead << ")]" << endl;
  }
}

void alloc_arrays()
{
  using namespace std;

  // bonds

  k_bnd = 20.0;
  R0 = 2.0; // = 0.4*a
  R0sq = R0*R0;
  e_bnd_coeff = k_bnd*R0sq/2.0; // SOP model
  nbnd = 1529;
  ibead_bnd = new int[nbnd+1];
  jbead_bnd = new int[nbnd+1];
  pdb_dist = new double[nbnd+1];
  bnds_allocated = 1;

  // angles

  k_ang = 20.0;
  e_ang_coeff = k_ang/2.0;
  nang = 1528;
  ibead_ang = new int[nang+1];
  jbead_ang = new int[nang+1];
  kbead_ang = new int[nang+1];
  pdb_ang = new double[nang+1];
  angs_allocated = 1;

  sigma_ss = 3.5; // = 0.76*a
  sigma_ss6 = pow(sigma_ss,6.0);
  epsilon_ss = 1.0;
  e_ang_ss_coeff = epsilon_ss*sigma_ss6;
  f_ang_ss_coeff = 6.0*e_ang_ss_coeff;

  // rna-rna vdw

  ncon_att = 8996;
  ncon_rep = 1157632;
  // neighbor list
  nnl_att = 0;
  nnl_rep = 0;
  // pair list
  nil_att = 0;
  nil_rep = 0;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      sigma_rep2[i][j] = sigma_rep[i][j] * sigma_rep[i][j];
      sigma_rep6[i][j] = sigma_rep2[i][j] * sigma_rep2[i][j] * sigma_rep2[i][j];
      sigma_rep12[i][j] = sigma_rep6[i][j] * sigma_rep6[i][j];
    }
  }

  ibead_lj_nat = new int[ncon_att+1];
  jbead_lj_nat = new int[ncon_att+1];
  itype_lj_nat = new int[ncon_att+1];
  jtype_lj_nat = new int[ncon_att+1];
  lj_nat_pdb_dist = new double[ncon_att+1];
  lj_nat_pdb_dist2 = new double[ncon_att+1];
  lj_nat_pdb_dist6 = new double[ncon_att+1];
  lj_nat_pdb_dist12 = new double[ncon_att+1];
  ibead_lj_non_nat = new int[ncon_rep+1];
  jbead_lj_non_nat = new int[ncon_rep+1];
  itype_lj_non_nat = new int[ncon_rep+1];
  jtype_lj_non_nat = new int[ncon_rep+1];

  ibead_neighbor_list_att = new int[ncon_att+1];
  jbead_neighbor_list_att = new int[ncon_att+1];
  itype_neighbor_list_att = new int[ncon_att+1];
  jtype_neighbor_list_att = new int[ncon_att+1];
  nl_lj_nat_pdb_dist = new double[ncon_att+1];
  nl_lj_nat_pdb_dist2 = new double[ncon_att+1];
  nl_lj_nat_pdb_dist6 = new double[ncon_att+1];
  nl_lj_nat_pdb_dist12 = new double[ncon_att+1];
  ibead_neighbor_list_rep = new int[ncon_rep+1];
  jbead_neighbor_list_rep = new int[ncon_rep+1];
  itype_neighbor_list_rep = new int[ncon_rep+1];
  jtype_neighbor_list_rep = new int[ncon_rep+1];

  ibead_pair_list_att = new int[ncon_att+1];
  jbead_pair_list_att = new int[ncon_att+1];
  itype_pair_list_att = new int[ncon_att+1];
  jtype_pair_list_att = new int[ncon_att+1];
  pl_lj_nat_pdb_dist = new double[ncon_att+1];
  pl_lj_nat_pdb_dist2 = new double[ncon_att+1];
  pl_lj_nat_pdb_dist6 = new double[ncon_att+1];
  pl_lj_nat_pdb_dist12 = new double[ncon_att+1];
  ibead_pair_list_rep = new int[ncon_rep+1];
  jbead_pair_list_rep = new int[ncon_rep+1];
  itype_pair_list_rep = new int[ncon_rep+1];
  jtype_pair_list_rep = new int[ncon_rep+1];

  lj_rna_rna_allocated = 1;

  // coordinates

  nbead = 1530;
  pos = new double3[nbead+1];
  unc_pos = new double3[nbead+1];
  vel = new double3[nbead+1];
  force = new double3[nbead+1];
  rna_base = new int [nbead+1];
  rna_phosphate = new int [nbead+1];
  pos_allocated = 1;
  unc_pos_allocated = 1;
  vel_allocated = 1;
  force_allocated = 1;
  rna_base_allocated = 1;
  rna_phosphate_allocated = 1;

  // increment
  incr = new double3[nbead+1];

  // miscellaneous run parameters

  run = 1;
  generator.set_seed(-100-run);
  T = 0.6; // kcal/mol

  neighborlist = 0; // neighbor list cutoff method?
  celllist = 0; // cell list cutoff method?
  boxl = 500.0;
  ncell = 55.0;
  lcell = boxl / ncell;
  zeta = 5.0e-2; // 0.05*tau^{-1} = friction coeff
  nstep = 5e7;
  nup = 1000;
  nnlup = 50; // neighbor list update frequency
  h = 2.5e-3;
  halfh = h/2.0;
  a1 = h*(1.0-zeta*halfh);
  a2 = h*halfh;
  a3 = (1.0-h*zeta/2.0+(h*zeta)*(h*zeta)/4.0)/h;
  a4 = halfh*(1.0-h*zeta/2.0);
  a5 = h/zeta;
  strcpy(ufname,"update.out");
  strcpy(rcfname,"restart_c.dat");
  strcpy(cfname,"coord.out");
  strcpy(unccfname,"unccoord.out");
  strcpy(vfname,"veloc.out");
  strcpy(binfname,"traj.bin");
  strcpy(uncbinfname,"traj_uncorrected.bin");
}

void release_bonds()
{
  using namespace std;

  delete [] ibead_bnd;
  delete [] jbead_bnd;
  delete [] pdb_dist;
  bnds_allocated = 0;
}

void init_bonds(int numbonds)
{
  using namespace std;

  nbnd = numbonds;
  ibead_bnd = new int[numbonds+1];
  jbead_bnd = new int[numbonds+1];
  pdb_dist = new double[numbonds+1];
  bnds_allocated = 1;
}

void release_angles()
{
  using namespace std;

  delete [] ibead_ang;
  delete [] jbead_ang;
  delete [] kbead_ang;
  delete [] pdb_ang;
  angs_allocated = 0;

}

void init_angles(int numangs)
{
  using namespace std;

  nang = numangs;
  ibead_ang = new int[numangs+1];
  jbead_ang = new int[numangs+1];
  kbead_ang = new int[numangs+1];
  pdb_ang = new double[numangs+1];
  angs_allocated = 1;

}

void release_lj()
{
  using namespace std;

  delete [] ibead_lj_nat;
  delete [] jbead_lj_nat;
  delete [] itype_lj_nat;
  delete [] jtype_lj_nat;
  delete [] lj_nat_pdb_dist;
  delete [] lj_nat_pdb_dist2;
  delete [] lj_nat_pdb_dist6;
  delete [] lj_nat_pdb_dist12;
  delete [] ibead_lj_non_nat;
  delete [] jbead_lj_non_nat;
  delete [] itype_lj_non_nat;
  delete [] jtype_lj_non_nat;

  delete [] ibead_neighbor_list_att;
  delete [] jbead_neighbor_list_att;
  delete [] itype_neighbor_list_att;
  delete [] jtype_neighbor_list_att;
  delete [] nl_lj_nat_pdb_dist;
  delete [] nl_lj_nat_pdb_dist2;
  delete [] nl_lj_nat_pdb_dist6;
  delete [] nl_lj_nat_pdb_dist12;
  delete [] ibead_neighbor_list_rep;
  delete [] jbead_neighbor_list_rep;
  delete [] itype_neighbor_list_rep;
  delete [] jtype_neighbor_list_rep;

  // pair list
  delete [] ibead_pair_list_att;
  delete [] jbead_pair_list_att;
  delete [] itype_pair_list_att;
  delete [] jtype_pair_list_att;
  delete [] pl_lj_nat_pdb_dist;
  delete [] pl_lj_nat_pdb_dist2;
  delete [] pl_lj_nat_pdb_dist6;
  delete [] pl_lj_nat_pdb_dist12;
  delete [] ibead_pair_list_rep;
  delete [] jbead_pair_list_rep;
  delete [] itype_pair_list_rep;
  delete [] jtype_pair_list_rep;

  lj_rna_rna_allocated = 0;

}

void init_lj(int numatt, int numrep )
{
  using namespace std;

  ncon_att = numatt;
  ncon_rep = numrep;
  ibead_lj_nat = new int[numatt+1];
  jbead_lj_nat = new int[numatt+1];
  itype_lj_nat = new int[numatt+1];
  jtype_lj_nat = new int[numatt+1];
  lj_nat_pdb_dist = new double[numatt+1];
  lj_nat_pdb_dist2 = new double[numatt+1];
  lj_nat_pdb_dist6 = new double[numatt+1];
  lj_nat_pdb_dist12 = new double[numatt+1];
  ibead_lj_non_nat = new int[numrep+1];
  jbead_lj_non_nat = new int[numrep+1];
  itype_lj_non_nat = new int[numrep+1];
  jtype_lj_non_nat = new int[numrep+1];

  ibead_neighbor_list_att = new int[numatt+1];
  jbead_neighbor_list_att = new int[numatt+1];
  itype_neighbor_list_att = new int[numatt+1];
  jtype_neighbor_list_att = new int[numatt+1];
  nl_lj_nat_pdb_dist = new double[numatt+1];
  nl_lj_nat_pdb_dist2 = new double[numatt+1];
  nl_lj_nat_pdb_dist6 = new double[numatt+1];
  nl_lj_nat_pdb_dist12 = new double[numatt+1];
  ibead_neighbor_list_rep = new int[numrep+1];
  jbead_neighbor_list_rep = new int[numrep+1];
  itype_neighbor_list_rep = new int[numrep+1];
  jtype_neighbor_list_rep = new int[numrep+1];

  ibead_pair_list_att = new int[numatt+1];
  jbead_pair_list_att = new int[numatt+1];
  itype_pair_list_att = new int[numatt+1];
  jtype_pair_list_att = new int[numatt+1];
  pl_lj_nat_pdb_dist = new double[numatt+1];
  pl_lj_nat_pdb_dist2 = new double[numatt+1];
  pl_lj_nat_pdb_dist6 = new double[numatt+1];
  pl_lj_nat_pdb_dist12 = new double[numatt+1];
  ibead_pair_list_rep = new int[numrep+1];
  jbead_pair_list_rep = new int[numrep+1];
  itype_pair_list_rep = new int[numrep+1];
  jtype_pair_list_rep = new int[numrep+1];

  lj_rna_rna_allocated = 1;

}

void init_pos(int nbead)
{
  using namespace std;

  unc_pos = new double3[nbead+1];
  pos = new double3[nbead+1];

  vel = new double3[nbead+1];
  force = new double3[nbead+1];

  pos_allocated = 1;
  unc_pos_allocated = 1;
  vel_allocated = 1;
  force_allocated = 1;
}

void release_pos()
{
  using namespace std;

  delete [] unc_pos;
  delete [] pos;

  delete [] vel;
  delete [] force;

  pos_allocated = 0;
  unc_pos_allocated = 0;
  vel_allocated = 0;
  force_allocated = 0;
}
