#include <iostream>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <unistd.h>
#include "sop.h"
#include "random_generator.h"
#include "energy.h"
#include "io.h"
#include "params.h"
#include "neighbor_list.h"
#include "cell_list.h"
#include "pair_list.h"

#include <vector_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#define SECTION_SIZE 1024

__device__ __constant__ double dev_sigma_rep[3][3] = {
  {0.0, 0.0, 0.0},
	{0.0, 3.8, 5.4},
	{0.0, 5.4, 7.0}
};

int main(int argc,char* argv[])
{

  using namespace std;

  if( argc<2 ) {
    cerr << "Usage: " << argv[0] <<  " < input_file >" << endl;
    exit(-1);
  }
  time_t tm0 = time(0); // wall time at this point
  cout << "CURRENT TIME IS: " << ctime(&tm0);
  if( getcwd(pathname,MAXPATHLEN)==NULL ) {
    cerr << "PROBLEM GETTING PATH" << endl;
  } else {
    cout << "CURRENT WORKING DIRECTORY: " << pathname << endl;
  }

  alloc_arrays(); // allocates certain arrays and initializes some variables
  read_input(argv[1]); // read input file

  clock_t ck0 = clock(); // clock ticks at this point
  ex_cmds(); // perform commands (simulation)

  // time stats
  time_t tm1 = time(0);
  clock_t ck1 = clock();
  cout << "+-------------------+" << endl;
  cout << "| Simulation Stats: |" << endl;
  cout << "+-------------------+" << endl;
  cout << "Wall Time              : " << difftime(tm1,tm0) << " sec" << endl;
  cout << "Total Computation Time : " << float(ck1-ck0)/CLOCKS_PER_SEC << " sec" << endl;
  cout << "Computation Rate       : " << float(ck1-ck0)/CLOCKS_PER_SEC/nstep << " sec / timestep" << endl;
  cout << "CURRENT TIME IS        : " << ctime(&tm1);

  return 0;

}

void ex_cmds()
{

  using namespace std;

  char oline[1024];
  int iopt;

  for( int i=1; i<=ncmd; i++ ) {
     // read data
     if( !strcmp(cmd[i],"load") ) { load(i); }
     // set parameters
     else if( !strcmp(cmd[i],"set") ) {
       set_params(i);
     }
     // run simulation
     else if( !strcmp(cmd[i],"run") ) { 
       simulation_ctrl();
      }
     // ???
     else {};
  }

}

void simulation_ctrl()
{
  using namespace std;

  switch( sim_type ) {
  case 1:
    underdamped_ctrl();
    break;
  case 2:
    overdamped_ctrl();
    break;
  default:
    cerr << "UNRECOGNIZED SIM_TYPE!" << endl;
    exit(-1);
  }
}

void underdamped_ctrl()
{
  using namespace std;

  char oline[2048];
  double istep = 1.0;
  int iup = 1;
  int inlup = 1;
  ofstream out(ufname,ios::out|ios::app);
  static int first_time = 1;

  coord* incr = new coord[nbead+1];

  if( (!restart)&&first_time ) { // zero out the velocities and forces
    for( int i=1; i<=nbead; i++ ) {
      vel[i].x = 0.0;
      vel[i].y = 0.0;
      vel[i].z = 0.0;
      force[i].x = 0.0;
      force[i].y = 0.0;
      force[i].z = 0.0;
    }
  }

  print_sim_params();

  if(neighborlist == 1){
    run_neighbor_list_update();
    run_pair_list_update();
  }else if(celllist == 1){
    run_cell_list_update();
    run_pair_list_update();
  }

  set_potential();
  set_forces();

  char line[2048];

  if( restart ) {
    load_coords(cfname,unccfname);
    load_vels(vfname);
    istep = istep_restart + 1.0;
  }

  if( rgen_restart ) {
    generator.restart();
  }

  if( first_time ) {
    energy_eval();
    force_eval();
  }

  if( binsave ) {
    if( (first_time)&&(!rgen_restart) ) {
      record_traj(binfname,uncbinfname);
    }
    while( istep <= nstep ) {
      // compute pair separation list
      if ((inlup % nnlup) == 0) {
        if(neighborlist == 1){
          run_neighbor_list_update();
        }else if(celllist == 1){
          run_cell_list_update();
        }
        inlup = 0;
      }
      inlup++;
      
      if(celllist == 1 || neighborlist == 1){
        run_pair_list_update();
      }

      underdamped_iteration(incr);
      if( !(iup%nup) ) { // updates
	      energy_eval();
	      calculate_observables(incr);
        sprintf(oline,"%.0lf %f %f %f %f %f %f %f %d %f", istep,T,kinT,e_bnd,e_ang_ss,e_vdw_rr,rna_etot, Q,contct_nat,rgsq);
	      out << oline << endl;
	      iup = 0;
	      record_traj(binfname,uncbinfname);
	      save_coords(cfname,unccfname);
	      save_vels(vfname);
	      generator.save_state();
      }

      istep += 1.0;
      iup++;
    }
    out.close();
  }

  if(first_time) {
    first_time = 0;
  }

  delete [] incr;

  return;
}

void calculate_observables(coord* increment)
{

  using namespace std;

  char oline[1024];
  double dx,dy,dz,d;
  static const double tol = 1.0; // tolerance for chi distances
  static const double chinorm = (double(nbead*nbead)-5.0*double(nbead)+6.0)/2.0;
  double sumvsq;
  int nchi;
  int ibead, jbead;
  int itype, jtype;
  float r_ij;
  char line[2048];

  // chi, contct_nat, contct_tot, Q

  contct_nat = 0;
  for( int i=1; i<=ncon_att; i++ ) {
    ibead = ibead_lj_nat[i];
    jbead = jbead_lj_nat[i];
    r_ij = lj_nat_pdb_dist[i];
    itype = itype_lj_nat[i];
    jtype = jtype_lj_nat[i];

    dx = unc_pos[ibead].x-unc_pos[jbead].x;
    dy = unc_pos[ibead].y-unc_pos[jbead].y;
    dz = unc_pos[ibead].z-unc_pos[jbead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d = sqrt( dx*dx+dy*dy+dz*dz );
    if(d/r_ij < 1.25) {
      contct_nat++;
    }
  }
  Q = double(contct_nat)/ncon_att;


  // rgsq

  rgsq = 0.0;
  for( int i=1; i<=nbead-1; i++ ) {
    for( int j=i+1; j<=nbead; j++ ) {
      dx = unc_pos[i].x-unc_pos[j].x;
      dy = unc_pos[i].y-unc_pos[j].y;
      dz = unc_pos[i].z-unc_pos[j].z;
      dx -= boxl*rnd(dx/boxl);
      dy -= boxl*rnd(dy/boxl);
      dz -= boxl*rnd(dz/boxl);

      rgsq += (dx*dx+dy*dy+dz*dz);
    }
  }
  rgsq /= double(nbead*nbead);

  // kinT

  if( sim_type == 1 ) {
    sumvsq = 0.0;

    for( int i=1; i<=nbead; i++ ) {
      sumvsq += vel[i].x*vel[i].x	+ vel[i].y*vel[i].y	+ vel[i].z*vel[i].z;
    }
    kinT = sumvsq/(3.0*double(nbead));
  } else if( sim_type == 2 ) {
    sumvsq = 0.0;
    for( int i=1; i<=nbead; i++ ) {
      sumvsq += increment[i].x*increment[i].x +
	increment[i].y*increment[i].y +
	increment[i].z*increment[i].z;
    }
    sumvsq *= zeta/(2.0*h);
    kinT = sumvsq/(3.0*double(nbead));
  } else {}
}

void underdamped_iteration(coord* incr)
{
  using namespace std;

  static const double eps = 1.0e-5;

  for( int i=1; i<=nbead; i++ ) {

    // compute position increments

    incr[i].x = a1*vel[i].x + a2*force[i].x;
    incr[i].y = a1*vel[i].y + a2*force[i].y;
    incr[i].z = a1*vel[i].z + a2*force[i].z;

    // update bead positions

    pos[i].x += incr[i].x;
    pos[i].y += incr[i].y;
    pos[i].z += incr[i].z;

    pos[i].x -= boxl*rnd(pos[i].x/boxl);
    pos[i].y -= boxl*rnd(pos[i].y/boxl);
    pos[i].z -= boxl*rnd(pos[i].z/boxl);

    unc_pos[i].x += incr[i].x;
    unc_pos[i].y += incr[i].y;
    unc_pos[i].z += incr[i].z;

  }

  // force_update

  force_eval();

  if( T < eps ) return; // don't update velocities for steepest descent

  // update_velocities

  for( int i=1; i<=nbead; i++ ) {
    // compute velocity increments
    vel[i].x = a3*incr[i].x + a4*force[i].x;
    vel[i].y = a3*incr[i].y + a4*force[i].y;
    vel[i].z = a3*incr[i].z + a4*force[i].z;
  }
}

void overdamped_iteration(coord* incr)
{
   using namespace std;

   for( int i=1; i<=nbead; i++ ) {

      // compute position increments

      incr[i].x = a5*force[i].x;
      incr[i].y = a5*force[i].y;
      incr[i].z = a5*force[i].z;

      // update bead positions

      unc_pos[i].x += incr[i].x;
      unc_pos[i].y += incr[i].y;
      unc_pos[i].z += incr[i].z;

      pos[i].x += incr[i].x;
      pos[i].y += incr[i].y;
      pos[i].z += incr[i].z;

      pos[i].x -= boxl*rnd(pos[i].x/boxl);
      pos[i].y -= boxl*rnd(pos[i].y/boxl);
      pos[i].z -= boxl*rnd(pos[i].z/boxl);

   }

   // force_update

   force_eval();

}

void overdamped_ctrl()
{

  using namespace std;

  char oline[2048];
  double istep = 1.0;
  int iup = 1;
  ofstream out(ufname,ios::out|ios::app);
  static int first_time = 1;

  coord* incr = new coord[nbead+1];

  if( (!restart)&&first_time ) { // zero out the velocities and forces
    for( int i=1; i<=nbead; i++ ) {
      vel[i].x = 0.0;
      vel[i].y = 0.0;
      vel[i].z = 0.0;
      force[i].x = 0.0;
      force[i].y = 0.0;
      force[i].z = 0.0;
    }
  }

  print_sim_params();  
  if(neighborlist == 1){
    run_neighbor_list_update();
    run_pair_list_update();
  }else if(celllist == 1){
    run_cell_list_update();
    run_pair_list_update();
  }

  set_potential();
  set_forces();

  char line[2048];

  if( restart ) {
    load_coords(cfname,unccfname);
    //    load_vels(vfname);
    istep = istep_restart + 1.0;
  }

  if( rgen_restart ) {
    generator.restart();
  }

  if( first_time ) {
    energy_eval();
    force_eval();
  }

  if( binsave ) {
    if( (first_time)&&(!rgen_restart) ) {
      record_traj(binfname,uncbinfname);
    }
    while( istep <= nstep ) {

      // compute pair separation list
      if ((inlup % nnlup) == 0) {        
        if(neighborlist == 1){
          run_neighbor_list_update();
        }else if(celllist == 1){
          run_cell_list_update();
        }
        
        inlup = 0;
      }
      inlup++;

      if(neighborlist == 1 || celllist == 1){
        run_pair_list_update();
      }

      overdamped_iteration(incr);
      if( !(iup%nup) ) { // updates
	      energy_eval();
	      calculate_observables(incr);
        sprintf(oline,"%.0lf %f %f %f %f %f %f %f %d %f",istep,T,kinT,e_bnd,e_ang_ss,e_vdw_rr,rna_etot,Q,contct_nat,rgsq);
	      out << oline << endl;
        iup = 0;
        record_traj(binfname,uncbinfname);
        save_coords(cfname,unccfname);
        save_vels(vfname);
        generator.save_state();
      }
      istep += 1.0;
      iup++;
    }
    out.close();
  }

  if(first_time){
    first_time = 0;
  }

  delete [] incr;

  return;
}

void run_ctrl()
{

  using namespace std;

  char oline[2048];
  double istep = 1.0;
  int iup = 1;
  ofstream out(ufname,ios::out|ios::app);
  static int first_time = 1;

  coord* incr = new coord[nbead+1];

  if( (!restart)&&first_time ) { // zero out the velocities and forces
    for( int i=1; i<=nbead; i++ ) {
      vel[i].x = 0.0;
      vel[i].y = 0.0;
      vel[i].z = 0.0;
      force[i].x = 0.0;
      force[i].y = 0.0;
      force[i].z = 0.0;
    }
  }

  print_sim_params();  
  if(neighborlist == 1){
    run_neighbor_list_update();
    run_pair_list_update();
  }else if(celllist == 1){
    run_cell_list_update();
    run_pair_list_update();
  }

  set_potential();
  set_forces();

  char line[2048];

  if( restart ) {
    load_coords(cfname,unccfname);
    load_vels(vfname);
    istep = istep_restart + 1.0;
  }

  if( rgen_restart ) {
    generator.restart();
  }

  if( first_time ) {
    energy_eval();
    force_eval();
  }

  if( binsave ) {
    if( (first_time)&&(!rgen_restart) ) {
      record_traj(binfname,uncbinfname);
    }
    while( istep <= nstep ) {

      // compute pair separation list
      if ((inlup % nnlup) == 0) {        
        if(neighborlist == 1){
          run_neighbor_list_update();
        }else if(celllist == 1){
          run_cell_list_update();
        }
        
        inlup = 0;
      }
      inlup++;

      if(neighborlist == 1 || celllist == 1){
        run_pair_list_update();
      }

      switch( sim_type ) {
        case 1:
          underdamped_iteration(incr);
          break;
        case 2:
          overdamped_iteration(incr);
          break;
        default:
          cerr << "UNRECOGNIZED SIM_TYPE!" << endl;
          exit(-1);
      }
      
      if( !(iup%nup) ) { // updates
	      energy_eval();
	      calculate_observables(incr);
        sprintf(oline,"%.0lf %f %f %f %f %f %f %f %d %f",istep,T,kinT,e_bnd,e_ang_ss,e_vdw_rr,rna_etot,Q,contct_nat,rgsq);
	      out << oline << endl;
        iup = 0;
        record_traj(binfname,uncbinfname);
        save_coords(cfname,unccfname);
        save_vels(vfname);
        generator.save_state();
      }
      istep += 1.0;
      iup++;
    }
    out.close();
  }

  if(first_time){
    first_time = 0;
  }

  delete [] incr;

  return;
}

void run_pair_list_update(){
  using namespace std;
  if(usegpu_pl == 0){
    update_pair_list();
  }else{
    if(!strcmp(pl_algorithm,"RL")){
      update_pair_list_RL();
    }else if(!strcmp(pl_algorithm,"thrust")){
      update_pair_list_thrust();
    }else if(!strcmp(pl_algorithm,"CL")){
      update_pair_list_CL();
    }else{
      cout << "INVALID ALGORITHM TYPE: " << pl_algorithm << endl;
      exit(-1);
    }
  }
}

void run_cell_list_update(){
  using namespace std;
  // Able to add additional algorithms
  update_cell_list();
}

void run_neighbor_list_update(){
  using namespace std;
  if(usegpu_nl == 0){
    update_neighbor_list();
  }else{
    if(!strcmp(nl_algorithm,"RL")){
      update_neighbor_list_RL();
    }else if(!strcmp(nl_algorithm,"thrust")){
      update_neighbor_list_thrust();
    }else if(!strcmp(nl_algorithm,"CL")){
      update_neighbor_list_CL();
    }else{
      cout << "INVALID ALGORITHM TYPE: " << nl_algorithm << endl;
      exit(-1);
    }
  }
}
