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

#include <math.h>
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
     else if( !strcmp(cmd[i],"set") ) { set_params(i); }
     // run simulation
     else if( !strcmp(cmd[i],"run") ) { simulation_ctrl(); }
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

  /*
  if (neighborlist == 1) {
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
  } else if (celllist == 1) {
    update_cell_list();
    update_pair_list();
  }*/

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
        /*
        if (neighborlist == 1) {
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
        } else if (celllist == 1) {
          update_cell_list();
      }*/
      
      if(neighborlist == 1){
        run_neighbor_list_update();
      }else if(celllist == 1){
        run_cell_list_update();
      }
      
	//	fprintf(stderr, "(%.0lf) neighbor list: (%d/%d)\n", istep, nnl_att, nnl_rep);
        inlup = 0;
      }
      inlup++;
      /*
      if (celllist == 1) {
        update_pair_list();
        //	fprintf(stderr, "(%.0lf) pair list: (%d/%d)\n", istep, nil_att, nil_rep);
      }else if(neighborlist == 1){
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
      }*/
      
      if(celllist == 1 || neighborlist == 1){
        run_pair_list_update();
      }

      underdamped_iteration(incr);
      if( !(iup%nup) ) { // updates
	energy_eval();
	calculate_observables(incr);
        sprintf(oline,"%.0lf %f %f %f %f %f %f %f %d %f",
                istep,T,kinT,e_bnd,e_ang_ss,e_vdw_rr,rna_etot,
                Q,contct_nat,rgsq);
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

  if( first_time ) first_time = 0;

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
      sumvsq += vel[i].x*vel[i].x
	+ vel[i].y*vel[i].y
	+ vel[i].z*vel[i].z;
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
  /*
  if (neighborlist == 1) {
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
  } else if (celllist == 1) {
    update_cell_list();
    update_pair_list();
  }*/

  /*
  if(neighborlist == 1){
    run_neighbor_list_update();
    run_pair_list_update();
  }else if(celllist == 1){
    run_cell_list_update();
    run_pair_list_update();
  }
  */


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
        /*
        if (neighborlist == 1) {
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
        } else if (celllist == 1) {
          update_cell_list();
        }*/
        
        if(neighborlist == 1){
          run_neighbor_list_update();
        }else if(celllist == 1){
          run_cell_list_update();
        }
        
	//	fprintf(stderr, "(%.0lf) neighbor list: (%d/%d)\n", istep, nnl_att, nnl_rep);
        inlup = 0;
      }
      inlup++;
      /*
      if (celllist == 1) {
        update_pair_list();
        //	fprintf(stderr, "(%.0lf) pair list: (%d/%d)\n", istep, nil_att, nil_rep);
      }else if(neighborlist == 1){
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
      }*/

      if(neighborlist == 1 || celllist == 1){
        run_pair_list_update();
      }

      overdamped_iteration(incr);
      if( !(iup%nup) ) { // updates
	energy_eval();
	calculate_observables(incr);
        sprintf(oline,"%.0lf %f %f %f %f %f %f %f %d %f",
                istep,T,kinT,e_bnd,e_ang_ss,e_vdw_rr,rna_etot,
                Q,contct_nat,rgsq);
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

  if( first_time ) first_time = 0;

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

void update_pair_list() {

  using namespace std;

  // declare host variables
  double dx, dy, dz;
  double d2;
  unsigned int ibead, jbead, itype, jtype;
  double rcut, rcut2;

  nil_att = 0;
  nil_rep = 0;

  // declare device variables

  // should be native distance
  for (int i=1; i<=nnl_att; i++) {

    ibead = ibead_neighbor_list_att[i];
    jbead = jbead_neighbor_list_att[i];
    itype = itype_neighbor_list_att[i];
    jtype = jtype_neighbor_list_att[i];

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;

    rcut = 2.5*nl_lj_nat_pdb_dist[i];
    rcut2 = rcut*rcut;

    if (d2 < rcut2) {
      // add to interaction pair list
      nil_att++;
      ibead_pair_list_att[nil_att] = ibead;
      jbead_pair_list_att[nil_att] = jbead;
      itype_pair_list_att[nil_att] = itype;
      jtype_pair_list_att[nil_att] = jtype;
      pl_lj_nat_pdb_dist[nil_att] = nl_lj_nat_pdb_dist[i];
      pl_lj_nat_pdb_dist2[nil_att] = nl_lj_nat_pdb_dist2[i];
      pl_lj_nat_pdb_dist6[nil_att] = nl_lj_nat_pdb_dist6[i];
      pl_lj_nat_pdb_dist12[nil_att] = nl_lj_nat_pdb_dist12[i];
    }

  }

  for (int i=1; i<=nnl_rep; i++) {

    ibead = ibead_neighbor_list_rep[i];
    jbead = jbead_neighbor_list_rep[i];
    itype = itype_neighbor_list_rep[i];
    jtype = jtype_neighbor_list_rep[i];

    dx = unc_pos[jbead].x - unc_pos[ibead].x;
    dy = unc_pos[jbead].y - unc_pos[ibead].y;
    dz = unc_pos[jbead].z - unc_pos[ibead].z;

    dx -= boxl*rnd(dx/boxl);
    dy -= boxl*rnd(dy/boxl);
    dz -= boxl*rnd(dz/boxl);

    d2 = dx*dx+dy*dy+dz*dz;

    rcut = 2.5*sigma_rep[itype][jtype];
    rcut2 = rcut*rcut;

    if (d2 < rcut2) {
      // add to interaction pair list
      nil_rep++;
      ibead_pair_list_rep[nil_rep] = ibead;
      jbead_pair_list_rep[nil_rep] = jbead;
      itype_pair_list_rep[nil_rep] = itype;
      jtype_pair_list_rep[nil_rep] = jtype;
      //printf("%d\n", 1);
    }else{
      //printf("%d\n", 0);
    }
  }
  //fflush(stdout);
  //exit(0);
}


void update_pair_list_CL(){
    int N;
    int *value;

    N = nnl_att;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_native_pl(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att, jtype_neighbor_list_att, unc_pos, nl_lj_nat_pdb_dist, value, boxl, N);

    compact_native_pl_CL(value);

    free(value);

    N = nnl_rep;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_non_native_pl(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep, unc_pos, value, boxl, N);

    compact_non_native_pl_CL(value);

    free(value);

}

void compact_native_pl_CL(int *value){
    int N;
    
    N = nnl_att;

    // typedef these iterators for shorthand
    typedef thrust::device_vector<int>::iterator   IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;

    // typedef a tuple of these iterators
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator, DoubleIterator, DoubleIterator, DoubleIterator, DoubleIterator> IteratorTuple;
    typedef thrust::tuple<int *, int *, int *, int *, double *, double *, double *, double *> HostIteratorTuple;

    // typedef the zip_iterator of this tuple
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::zip_iterator<HostIteratorTuple> HostZipIterator;

    // Create device initial vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_att(ibead_neighbor_list_att, ibead_neighbor_list_att+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att(jbead_neighbor_list_att, jbead_neighbor_list_att+N);
    thrust::device_vector<int> dev_itype_neighbor_list_att(itype_neighbor_list_att, itype_neighbor_list_att+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att(jtype_neighbor_list_att, jtype_neighbor_list_att+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist(nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2(nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6(nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12(nl_lj_nat_pdb_dist12, nl_lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_att(N);
    thrust::device_vector<int> dev_jbead_pair_list_att(N);
    thrust::device_vector<int> dev_itype_pair_list_att(N);
    thrust::device_vector<int> dev_jtype_pair_list_att(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist2(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist6(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist12(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_att.begin(), dev_jbead_neighbor_list_att.begin(), dev_itype_neighbor_list_att.begin(), dev_jtype_neighbor_list_att.begin(),
                                            dev_nl_lj_nat_pdb_dist.begin(), dev_nl_lj_nat_pdb_dist2.begin(), dev_nl_lj_nat_pdb_dist6.begin(), dev_nl_lj_nat_pdb_dist12.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_att.end(), dev_jbead_neighbor_list_att.end(), dev_itype_neighbor_list_att.end(), dev_jtype_neighbor_list_att.end(),
                                            dev_nl_lj_nat_pdb_dist.end(), dev_nl_lj_nat_pdb_dist2.end(), dev_nl_lj_nat_pdb_dist6.end(), dev_nl_lj_nat_pdb_dist12.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_att.begin(), dev_jbead_pair_list_att.begin(), dev_itype_pair_list_att.begin(),
                                            dev_jtype_pair_list_att.begin(), dev_pl_lj_nat_pdb_dist.begin(), dev_pl_lj_nat_pdb_dist2.begin(), 
                                            dev_pl_lj_nat_pdb_dist6.begin(), dev_pl_lj_nat_pdb_dist12.begin()));

    thrust::sort_by_key(dev_value.begin(), dev_value.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value.begin(), dev_value.end(), dev_value.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value.end()-1, dev_value.end(), &arrSize);

    nil_att = arrSize;

    free(ibead_pair_list_att);
    free(jbead_pair_list_att);
    free(itype_pair_list_att);
    free(jtype_pair_list_att);
    free(pl_lj_nat_pdb_dist);
    free(pl_lj_nat_pdb_dist2);
    free(pl_lj_nat_pdb_dist6);
    free(pl_lj_nat_pdb_dist12);

    ibead_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    jbead_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    itype_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    jtype_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    pl_lj_nat_pdb_dist = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist2 = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist6 = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist12 = (double *)malloc(nil_att*sizeof(double));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_pair_list_att, jbead_pair_list_att, itype_pair_list_att,
                                            jtype_pair_list_att, pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist2, pl_lj_nat_pdb_dist6, pl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_att.begin() + nil_att, dev_jbead_pair_list_att.begin() + nil_att,
                                                dev_itype_pair_list_att.begin() + nil_att, dev_jtype_pair_list_att.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist.begin() + nil_att, dev_pl_lj_nat_pdb_dist2.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist6.begin() + nil_att, dev_pl_lj_nat_pdb_dist12.begin() + nil_att));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nil_att--;
}

void compact_non_native_pl_CL(int *value){
    int N;

    N = nnl_rep;

    // typedef these iterators for shorthand
    typedef thrust::device_vector<int>::iterator   IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;

    // typedef a tuple of these iterators
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator> IteratorTuple;
    typedef thrust::tuple<int *, int *, int *, int *> HostIteratorTuple;

    // typedef the zip_iterator of this tuple
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::zip_iterator<HostIteratorTuple> HostZipIterator;

    // Create device initial vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_rep(ibead_neighbor_list_rep, ibead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep(jbead_neighbor_list_rep, jbead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep(itype_neighbor_list_rep, itype_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep(jtype_neighbor_list_rep, jtype_neighbor_list_rep+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_rep(N);
    thrust::device_vector<int> dev_jbead_pair_list_rep(N);
    thrust::device_vector<int> dev_itype_pair_list_rep(N);
    thrust::device_vector<int> dev_jtype_pair_list_rep(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep.begin(), dev_jbead_neighbor_list_rep.begin(), dev_itype_neighbor_list_rep.begin(), dev_jtype_neighbor_list_rep.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_rep.end(), dev_jbead_neighbor_list_rep.end(), dev_itype_neighbor_list_rep.end(), dev_jtype_neighbor_list_rep.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_rep.begin(), dev_jbead_pair_list_rep.begin(), dev_itype_pair_list_rep.begin(),
                                            dev_jtype_pair_list_rep.begin()));

    thrust::sort_by_key(dev_value.begin(), dev_value.end(), dev_initial_begin, thrust::greater<int>());

    thrust::inclusive_scan(dev_value.begin(), dev_value.end(), dev_value.begin(), thrust::plus<int>());

    int arrSize;

    thrust::copy(dev_value.end()-1, dev_value.end(), &arrSize);

    nil_rep = arrSize;

    free(ibead_pair_list_rep);
    free(jbead_pair_list_rep);
    free(itype_pair_list_rep);
    free(jtype_pair_list_rep);

    ibead_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    jbead_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    itype_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    jtype_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_pair_list_rep, jbead_pair_list_rep, itype_pair_list_rep, jtype_pair_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_rep.begin() + nil_rep, dev_jbead_pair_list_rep.begin() + nil_rep,
                                                dev_itype_pair_list_rep.begin() + nil_rep, dev_jtype_pair_list_rep.begin() + nil_rep));

    thrust::copy(dev_initial_begin, dev_initial_begin + arrSize, host_result_begin);

    nil_rep--;
}

void update_pair_list_thrust(){
    int N;
    int *value;

    N = nnl_att;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_native_pl(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att, jtype_neighbor_list_att, unc_pos, nl_lj_nat_pdb_dist, value, boxl, N);

    compact_native_pl_thrust(value);

    free(value);

    N = nnl_rep;
    
    value = (int *)malloc(N*sizeof(int));

    calculate_array_non_native_pl(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep, unc_pos, value, boxl, N);

    compact_non_native_pl_thrust(value);

    free(value);

}

void compact_native_pl_thrust(int *value){
    int N;

    N = nnl_att;

    // typedef these iterators for shorthand
    typedef thrust::device_vector<int>::iterator   IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;

    // typedef a tuple of these iterators
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator, DoubleIterator, DoubleIterator, DoubleIterator, DoubleIterator> IteratorTuple;
    typedef thrust::tuple<int *, int *, int *, int *, double *, double *, double *, double *> HostIteratorTuple;

    // typedef the zip_iterator of this tuple
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::zip_iterator<HostIteratorTuple> HostZipIterator;

    // Create device initial vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_att(ibead_neighbor_list_att, ibead_neighbor_list_att+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_att(jbead_neighbor_list_att, jbead_neighbor_list_att+N);
    thrust::device_vector<int> dev_itype_neighbor_list_att(itype_neighbor_list_att, itype_neighbor_list_att+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_att(jtype_neighbor_list_att, jtype_neighbor_list_att+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist(nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist2(nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist2+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist6(nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist6+N);
    thrust::device_vector<double> dev_nl_lj_nat_pdb_dist12(nl_lj_nat_pdb_dist12, nl_lj_nat_pdb_dist12+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_att(N);
    thrust::device_vector<int> dev_jbead_pair_list_att(N);
    thrust::device_vector<int> dev_itype_pair_list_att(N);
    thrust::device_vector<int> dev_jtype_pair_list_att(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist2(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist6(N);
    thrust::device_vector<double> dev_pl_lj_nat_pdb_dist12(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_att.begin(), dev_jbead_neighbor_list_att.begin(), dev_itype_neighbor_list_att.begin(), dev_jtype_neighbor_list_att.begin(),
                                            dev_nl_lj_nat_pdb_dist.begin(), dev_nl_lj_nat_pdb_dist2.begin(), dev_nl_lj_nat_pdb_dist6.begin(), dev_nl_lj_nat_pdb_dist12.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_att.end(), dev_jbead_neighbor_list_att.end(), dev_itype_neighbor_list_att.end(), dev_jtype_neighbor_list_att.end(),
                                            dev_nl_lj_nat_pdb_dist.end(), dev_nl_lj_nat_pdb_dist2.end(), dev_nl_lj_nat_pdb_dist6.end(), dev_nl_lj_nat_pdb_dist12.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_att.begin(), dev_jbead_pair_list_att.begin(), dev_itype_pair_list_att.begin(),
                                            dev_jtype_pair_list_att.begin(), dev_pl_lj_nat_pdb_dist.begin(), dev_pl_lj_nat_pdb_dist2.begin(), 
                                            dev_pl_lj_nat_pdb_dist6.begin(), dev_pl_lj_nat_pdb_dist12.begin()));

    nil_att = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    free(ibead_pair_list_att);
    free(jbead_pair_list_att);
    free(itype_pair_list_att);
    free(jtype_pair_list_att);
    free(pl_lj_nat_pdb_dist);
    free(pl_lj_nat_pdb_dist2);
    free(pl_lj_nat_pdb_dist6);
    free(pl_lj_nat_pdb_dist12);

    ibead_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    jbead_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    itype_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    jtype_pair_list_att = (int *)malloc(nil_att*sizeof(int));
    pl_lj_nat_pdb_dist = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist2 = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist6 = (double *)malloc(nil_att*sizeof(double));
    pl_lj_nat_pdb_dist12 = (double *)malloc(nil_att*sizeof(double));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_pair_list_att, jbead_pair_list_att, itype_pair_list_att,
                                            jtype_pair_list_att, pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist2, pl_lj_nat_pdb_dist6, pl_lj_nat_pdb_dist12));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_att.begin() + nil_att, dev_jbead_pair_list_att.begin() + nil_att,
                                                dev_itype_pair_list_att.begin() + nil_att, dev_jtype_pair_list_att.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist.begin() + nil_att, dev_pl_lj_nat_pdb_dist2.begin() + nil_att, 
                                                dev_pl_lj_nat_pdb_dist6.begin() + nil_att, dev_pl_lj_nat_pdb_dist12.begin() + nil_att));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nil_att--;
}

void compact_non_native_pl_thrust(int *value){
    int N;

    N = nnl_rep;

    // typedef these iterators for shorthand
    typedef thrust::device_vector<int>::iterator   IntIterator;
    typedef thrust::device_vector<double>::iterator DoubleIterator;

    // typedef a tuple of these iterators
    typedef thrust::tuple<IntIterator, IntIterator, IntIterator, IntIterator> IteratorTuple;
    typedef thrust::tuple<int *, int *, int *, int *> HostIteratorTuple;

    // typedef the zip_iterator of this tuple
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef thrust::zip_iterator<HostIteratorTuple> HostZipIterator;

    // Create device initial vectors
    thrust::device_vector<int> dev_ibead_neighbor_list_rep(ibead_neighbor_list_rep, ibead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jbead_neighbor_list_rep(jbead_neighbor_list_rep, jbead_neighbor_list_rep+N);
    thrust::device_vector<int> dev_itype_neighbor_list_rep(itype_neighbor_list_rep, itype_neighbor_list_rep+N);
    thrust::device_vector<int> dev_jtype_neighbor_list_rep(jtype_neighbor_list_rep, jtype_neighbor_list_rep+N);

    // Create device value vector
    thrust::device_vector<int> dev_value(value, value+N);

    // Create result vectors
    thrust::device_vector<int> dev_ibead_pair_list_rep(N);
    thrust::device_vector<int> dev_jbead_pair_list_rep(N);
    thrust::device_vector<int> dev_itype_pair_list_rep(N);
    thrust::device_vector<int> dev_jtype_pair_list_rep(N);

    ZipIterator dev_initial_begin(thrust::make_tuple(dev_ibead_neighbor_list_rep.begin(), dev_jbead_neighbor_list_rep.begin(), dev_itype_neighbor_list_rep.begin(), dev_jtype_neighbor_list_rep.begin()));
                                            
    ZipIterator dev_initial_end(thrust::make_tuple(dev_ibead_neighbor_list_rep.end(), dev_jbead_neighbor_list_rep.end(), dev_itype_neighbor_list_rep.end(), dev_jtype_neighbor_list_rep.end()));

    ZipIterator dev_result_begin(thrust::make_tuple(dev_ibead_pair_list_rep.begin(), dev_jbead_pair_list_rep.begin(), dev_itype_pair_list_rep.begin(),
                                            dev_jtype_pair_list_rep.begin()));

    nil_rep = thrust::copy_if(dev_initial_begin,  dev_initial_end, dev_value.begin(),  dev_result_begin, (thrust::placeholders::_1 == 1)) - dev_result_begin;

    free(ibead_pair_list_rep);
    free(jbead_pair_list_rep);
    free(itype_pair_list_rep);
    free(jtype_pair_list_rep);

    ibead_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    jbead_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    itype_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));
    jtype_pair_list_rep = (int *)malloc(nil_rep*sizeof(int));

    HostZipIterator host_result_begin(thrust::make_tuple(ibead_pair_list_rep, jbead_pair_list_rep, itype_pair_list_rep, jtype_pair_list_rep));

    ZipIterator dev_result_end(thrust::make_tuple(dev_ibead_pair_list_rep.begin() + nil_rep, dev_jbead_pair_list_rep.begin() + nil_rep,
                                                dev_itype_pair_list_rep.begin() + nil_rep, dev_jtype_pair_list_rep.begin() + nil_rep));

    thrust::copy(dev_result_begin, dev_result_end, host_result_begin);

    nil_rep--;
}

void update_pair_list_RL(){
    // Declare N
	int N;
	
	// Set N
	N = nnl_att+1;
	
	// Declare value array
	int *value;
	value = (int *)malloc(N*sizeof(int));
	
	// Calculate binary list for att
	calculate_array_native_pl(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att, jtype_neighbor_list_att, unc_pos, nl_lj_nat_pdb_dist, value, boxl, N);
  
    /*
	// Compact ibead_pair_list_att
	nil_att = compact(ibead_neighbor_list_att, value, N, ibead_pair_list_att)-1;
	
	// Compact jbead_pair_list_att
	compact(jbead_neighbor_list_att, value, N, jbead_pair_list_att);
	
	// Compact itype_pair_list_att
	compact(itype_neighbor_list_att, value, N, itype_pair_list_att);
	
	// Compact jtype_pair_list_att
	compact(jtype_neighbor_list_att, value, N, jtype_pair_list_att);
	
	// Compact pl_lj_nat_pdb_dist
	compact(nl_lj_nat_pdb_dist, value, N, pl_lj_nat_pdb_dist);
	
	// Compact pl_lj_nat_pdb_dist2
	compact(nl_lj_nat_pdb_dist2, value, N, pl_lj_nat_pdb_dist2);
	
	// Compact pl_lj_nat_pdb_dist6
	compact(nl_lj_nat_pdb_dist6, value, N, pl_lj_nat_pdb_dist6);
	
	// Compact pl_lj_nat_pdb_dist12
	compact(nl_lj_nat_pdb_dist12, value, N, pl_lj_nat_pdb_dist12);*/
    
  nil_att = compact_native_pl(ibead_neighbor_list_att, jbead_neighbor_list_att, itype_neighbor_list_att, jtype_neighbor_list_att, nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist12, value, N, 
                  ibead_pair_list_att, jbead_pair_list_att, itype_pair_list_att, jtype_pair_list_att, pl_lj_nat_pdb_dist, pl_lj_nat_pdb_dist2,
                  pl_lj_nat_pdb_dist6, pl_lj_nat_pdb_dist12) - 1;

  /*
  printf("Native: %d\n", nil_att);
  fflush(stdout);*/
	
	// Free value memory
	free(value);
	
	
	/**********************************
	 *								  *
	 * End of Attractive Calculations *
	 *								  *
	 **********************************/
	
	
	// Set N
	N = nnl_rep+1;
	
	// Declare value array
	value = (int *)malloc(N*sizeof(int));
	
	// Calculate binary list for rep
	calculate_array_non_native_pl(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep, unc_pos, value, boxl, N);
  /*
  for(int i = 0; i <= N; i++){
    printf("%d\n", value[i]);
  }
  fflush(stdout);
  exit(0);*/

    /*
	// Compact ibead_pair_list_rep
	nil_rep = compact(ibead_neighbor_list_rep, value, N, ibead_pair_list_rep)-1;
	
	// Compact jbead_pair_list_rep
	compact(jbead_neighbor_list_rep, value, N, jbead_pair_list_rep);
	
	// Compact itype_pair_list_rep
	compact(itype_neighbor_list_rep, value, N, itype_pair_list_rep);
	
	// Compact jtype_pair_list_rep
	compact(jtype_neighbor_list_rep, value, N, jtype_pair_list_rep);*/
    
  nil_rep = compact_non_native_pl(ibead_neighbor_list_rep, jbead_neighbor_list_rep, itype_neighbor_list_rep, jtype_neighbor_list_rep, value, N, 
                  ibead_pair_list_rep, jbead_pair_list_rep, itype_pair_list_rep, jtype_pair_list_rep) - 1;
  /*
  printf("Non-Native: %d\n", nil_rep);
  fflush(stdout);*/

  free(value);
}

void calculate_array_native_pl(int *ibead_neighbor_list_att, int *jbead_neighbor_list_att, int *itype_neighbor_list_att, int *jtype_neighbor_list_att, double3 *unc_pos, double *nl_lj_nat_pdb_dist, 
                            int *value, int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Declare device pointers
	int *dev_ibead_neighbor_list_att;
	int *dev_jbead_neighbor_list_att;
	int *dev_itype_neighbor_list_att;
	int *dev_jtype_neighbor_list_att;
	double3 *dev_unc_pos;
	double *dev_nl_lj_nat_pdb_dist; 
	int *dev_value;
	
	// Allocate device arrays
	cudaMalloc((void **)&dev_ibead_neighbor_list_att, size_int);	
	cudaMalloc((void **)&dev_jbead_neighbor_list_att, size_int);
	cudaMalloc((void **)&dev_itype_neighbor_list_att, size_int);
	cudaMalloc((void **)&dev_jtype_neighbor_list_att, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	cudaMalloc((void **)&dev_nl_lj_nat_pdb_dist, size_double);
	cudaMalloc((void **)&dev_value, size_int);
	
	// Copy host arrays to device arrays
	cudaMemcpy(dev_ibead_neighbor_list_att, ibead_neighbor_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_neighbor_list_att, jbead_neighbor_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_neighbor_list_att, itype_neighbor_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_neighbor_list_att, jtype_neighbor_list_att, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist, size_double, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
  int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	// Compute binary array
	array_native_pl_kernel<<<blocks, threads>>>(dev_ibead_neighbor_list_att, dev_jbead_neighbor_list_att, dev_itype_neighbor_list_att, dev_jtype_neighbor_list_att, dev_unc_pos, dev_nl_lj_nat_pdb_dist, dev_value, boxl, N);

  // Sync device
  cudaDeviceSynchronize();

	// Copy device array to host array
	cudaMemcpy(value, dev_value, size_int, cudaMemcpyDeviceToHost);
	
  cudaDeviceSynchronize();

	// Free GPU memory
	cudaFree(dev_ibead_neighbor_list_att);
	cudaFree(dev_jbead_neighbor_list_att);
	cudaFree(dev_itype_neighbor_list_att);
	cudaFree(dev_jtype_neighbor_list_att);
	cudaFree(dev_unc_pos);
	cudaFree(dev_nl_lj_nat_pdb_dist);
	cudaFree(dev_value);
}

__global__ void array_native_pl_kernel(int *dev_ibead_neighbor_list_att, int *dev_jbead_neighbor_list_att, int *dev_itype_neighbor_list_att, int *dev_jtype_neighbor_list_att, double3 *dev_unc_pos, double *dev_nl_lj_nat_pdb_dist, 
                            int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i <= N){
    double dx, dy, dz;
    double d2;
    int ibead, jbead, itype, jtype;
    double rcut, rcut2;

    // record sigma for ibead and jbead
    ibead = dev_ibead_neighbor_list_att[i];

    jbead = dev_jbead_neighbor_list_att[i];

    // record type of bead for ibead and jbead
    itype = dev_itype_neighbor_list_att[i];

    jtype = dev_jtype_neighbor_list_att[i];
    
    // calculate distance in x, y, and z for ibead and jbead
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

    /* 
    Compute the cutoff distance for the given bead
    This is based off of nl_lj_nat_pdb_dist[i], which is the distance 
    from ibead to jbead in the resulting folded structure
    */
    rcut = 2.5*dev_nl_lj_nat_pdb_dist[i];

    // square cutoff distance, since sqrt(d2) is computationally expensive
    rcut2 = rcut*rcut;

    if(d2 < rcut2){
      dev_value[i] = 1;
    }else{
      dev_value[i] = 0;
    }
  }else if(i == 0){
      dev_value[i] = 1;
  }
}

void calculate_array_non_native_pl(int *ibead_neighbor_list_rep, int *jbead_neighbor_list_rep, int *itype_neighbor_list_rep, int *jtype_neighbor_list_rep, double3 *unc_pos,
                            int *value, int boxl, int N){
							
	// Calculate array sizes
	int size_int = N*sizeof(int);
	int size_double = N*sizeof(double);
	int size_double3 = (nbead+1)*sizeof(double3);
	
	// Declare device pointers
	int *dev_ibead_neighbor_list_rep;
	int *dev_jbead_neighbor_list_rep;
	int *dev_itype_neighbor_list_rep;
	int *dev_jtype_neighbor_list_rep;
	double3 *dev_unc_pos; 
	int *dev_value;
	
	// Allocate device arrays
	cudaMalloc((void **)&dev_ibead_neighbor_list_rep, size_int);	
	cudaMalloc((void **)&dev_jbead_neighbor_list_rep, size_int);
	cudaMalloc((void **)&dev_itype_neighbor_list_rep, size_int);
	cudaMalloc((void **)&dev_jtype_neighbor_list_rep, size_int);
	cudaMalloc((void **)&dev_unc_pos, size_double3);
	cudaMalloc((void **)&dev_value, size_int);
	
	// Copy host arrays to device arrays
	cudaMemcpy(dev_ibead_neighbor_list_rep, ibead_neighbor_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jbead_neighbor_list_rep, jbead_neighbor_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itype_neighbor_list_rep, itype_neighbor_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_jtype_neighbor_list_rep, jtype_neighbor_list_rep, size_int, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_unc_pos, unc_pos, size_double3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_value, value, size_int, cudaMemcpyHostToDevice);
	
	// Calculate block/thread count
	int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);
	
	// Compute binary array
	array_non_native_pl_kernel<<<blocks, threads>>>(dev_ibead_neighbor_list_rep, dev_jbead_neighbor_list_rep, dev_itype_neighbor_list_rep, dev_jtype_neighbor_list_rep, 
                                                dev_unc_pos, dev_value, boxl, N);
    /*
    cudaDeviceSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }else{
        printf("Success\n");
        exit(0);
    }*/
	
    // Sync device
    cudaDeviceSynchronize();

	// Copy device array to host array
	cudaMemcpy(value, dev_value, size_int, cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(dev_ibead_neighbor_list_rep);
	cudaFree(dev_jbead_neighbor_list_rep);
	cudaFree(dev_itype_neighbor_list_rep);
	cudaFree(dev_jtype_neighbor_list_rep);
	cudaFree(dev_unc_pos);
	cudaFree(dev_value);
}

__global__ void array_non_native_pl_kernel(int *dev_ibead_neighbor_list_rep, int *dev_jbead_neighbor_list_rep, int *dev_itype_neighbor_list_rep, int *dev_jtype_neighbor_list_rep, 
                                        double3 *dev_unc_pos, int *dev_value, int boxl, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > 0 && i <= N){
    double dx, dy, dz;
    double d2;
    int ibead, jbead, itype, jtype;
    double rcut, rcut2;

    // record sigma for ibead and jbead
    ibead = dev_ibead_neighbor_list_rep[i];
    jbead = dev_jbead_neighbor_list_rep[i];

    // record type of bead for ibead and jbead
    itype = dev_itype_neighbor_list_rep[i];
    jtype = dev_jtype_neighbor_list_rep[i];
    
    // calculate distance in x, y, and z for ibead and jbead
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

    /* 
    Compute the cutoff distance for the given bead
    This is based off of nl_lj_nat_pdb_dist[i], which is the distance 
    from ibead to jbead in the resulting folded structure
    */
	// May need to change to dev_sigma_rep[N*itype + jtype]
    rcut = 2.5*dev_sigma_rep[itype][jtype];

    // square cutoff distance, since sqrt(d2) is computationally expensive
    rcut2 = rcut*rcut;

    if(d2 < rcut2){
      dev_value[i] = 1;
    }else{
      dev_value[i] = 0;
    }
  }else if(i == 0){
      dev_value[i] = 1;
  }
}


int compact_native_pl(int *ibead_neighbor_list_att, int *jbead_neighbor_list_att, int *itype_neighbor_list_att, int *jtype_neighbor_list_att, double *nl_lj_nat_pdb_dist,
                    double *nl_lj_nat_pdb_dist2, double *nl_lj_nat_pdb_dist6, double *nl_lj_nat_pdb_dist12, int *value, int N, 
                    int *&ibead_pair_list_att, int *&jbead_pair_list_att, int *&itype_pair_list_att,
                    int *&jtype_pair_list_att, double *&pl_lj_nat_pdb_dist, double *&pl_lj_nat_pdb_dist2,
                    double *&pl_lj_nat_pdb_dist6, double *&pl_lj_nat_pdb_dist12){
    // Declare pointers for dev_output and dev_value arrays
    int *dev_output;
    int *dev_value;

    // Calculate array size
    int size = N * sizeof(int);

    // Allocate dev_value and dev_output arrays
    cudaMalloc((void**)&dev_value, size);
    cudaMalloc((void**)&dev_output, size);
 
    // Copy data from value array to device (dev_value)
    cudaMemcpy(dev_value, value, size, cudaMemcpyHostToDevice);

    // Perform hierarchical Kogge-Stone scan on dev_value array and store result in dev_output
    hier_ks_scan(dev_value, dev_output, N, 0);

    // Copy size of compacted array from device to host and store in arrSize
    int arrSize;
    cudaMemcpy(&arrSize, &dev_output[N-1], sizeof(int), cudaMemcpyDeviceToHost); 

    // Increment arrSize by 1 if needed
    if(value[N-1]){
        arrSize++;
    }

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    // Declare and allocate dev_result array to store compacted indices on device (on GPU)
    int *dev_ibead_neighbor_list_att;
    cudaMalloc((void**)&dev_ibead_neighbor_list_att, N*sizeof(int));
    cudaMemcpy(dev_ibead_neighbor_list_att, ibead_neighbor_list_att, N*sizeof(int), cudaMemcpyHostToDevice);
    int *dev_ibead_pair_list_att;
    cudaMalloc((void**)&dev_ibead_pair_list_att, arrSize*sizeof(int));

    copyElements<<<blocks, threads>>>(dev_ibead_neighbor_list_att, dev_value, dev_output, dev_ibead_pair_list_att, N);
    cudaDeviceSynchronize();
    free(ibead_pair_list_att);
    ibead_pair_list_att = (int *)malloc(arrSize*sizeof(int));
    cudaMemcpy(ibead_pair_list_att, dev_ibead_pair_list_att, arrSize*sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(dev_ibead_neighbor_list_att);
    cudaFree(dev_ibead_pair_list_att);

    cudaDeviceSynchronize();
    
    int *dev_jbead_neighbor_list_att;
    cudaMalloc((void**)&dev_jbead_neighbor_list_att, N*sizeof(int));
    cudaMemcpy(dev_jbead_neighbor_list_att, jbead_neighbor_list_att, N*sizeof(int), cudaMemcpyHostToDevice);
    int *dev_jbead_pair_list_att;
    cudaMalloc((void**)&dev_jbead_pair_list_att, arrSize*sizeof(int));

    copyElements<<<blocks, threads>>>(dev_jbead_neighbor_list_att, dev_value, dev_output, dev_jbead_pair_list_att, N);
    cudaDeviceSynchronize();
    free(jbead_pair_list_att);
    jbead_pair_list_att = (int *)malloc(arrSize*sizeof(int));
    cudaMemcpy(jbead_pair_list_att, dev_jbead_pair_list_att, arrSize*sizeof(int), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_jbead_neighbor_list_att);
    cudaFree(dev_jbead_pair_list_att);

    cudaDeviceSynchronize();

    int *dev_itype_neighbor_list_att;
    cudaMalloc((void**)&dev_itype_neighbor_list_att, N*sizeof(int));
    cudaMemcpy(dev_itype_neighbor_list_att, itype_neighbor_list_att, N*sizeof(int), cudaMemcpyHostToDevice);
    int *dev_itype_pair_list_att;
    cudaMalloc((void**)&dev_itype_pair_list_att, arrSize*sizeof(int));

    copyElements<<<blocks, threads>>>(dev_itype_neighbor_list_att, dev_value, dev_output, dev_itype_pair_list_att, N);
    cudaDeviceSynchronize();
    free(itype_pair_list_att);
    itype_pair_list_att = (int *)malloc(arrSize*sizeof(int));
    cudaMemcpy(itype_pair_list_att, dev_itype_pair_list_att, arrSize*sizeof(int), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_itype_neighbor_list_att);
    cudaFree(dev_itype_pair_list_att);

    cudaDeviceSynchronize();


    int *dev_jtype_neighbor_list_att;
    cudaMalloc((void**)&dev_jtype_neighbor_list_att, N*sizeof(int));
    cudaMemcpy(dev_jtype_neighbor_list_att, jtype_neighbor_list_att, N*sizeof(int), cudaMemcpyHostToDevice);
    int *dev_jtype_pair_list_att;
    cudaMalloc((void**)&dev_jtype_pair_list_att, arrSize*sizeof(int));

    copyElements<<<blocks, threads>>>(dev_jtype_neighbor_list_att, dev_value, dev_output, dev_jtype_pair_list_att, N);
    cudaDeviceSynchronize();
    free(jtype_pair_list_att);
    jtype_pair_list_att = (int *)malloc(arrSize*sizeof(int));
    cudaMemcpy(jtype_pair_list_att, dev_jtype_pair_list_att, arrSize*sizeof(int), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_jtype_neighbor_list_att);
    cudaFree(dev_jtype_pair_list_att);

    cudaDeviceSynchronize();

    double *dev_nl_lj_nat_pdb_dist;
    cudaMalloc((void**)&dev_nl_lj_nat_pdb_dist, N*sizeof(double));
    cudaMemcpy(dev_nl_lj_nat_pdb_dist, nl_lj_nat_pdb_dist, N*sizeof(double), cudaMemcpyHostToDevice);
    double *dev_pl_lj_nat_pdb_dist;
    cudaMalloc((void**)&dev_pl_lj_nat_pdb_dist, arrSize*sizeof(double));

    copyElements<<<blocks, threads>>>(dev_nl_lj_nat_pdb_dist, dev_value, dev_output, dev_pl_lj_nat_pdb_dist, N);
    cudaDeviceSynchronize();
    free(pl_lj_nat_pdb_dist);
    pl_lj_nat_pdb_dist = (double *)malloc(arrSize*sizeof(double));
    cudaMemcpy(pl_lj_nat_pdb_dist, dev_pl_lj_nat_pdb_dist, arrSize*sizeof(double), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_nl_lj_nat_pdb_dist);
    cudaFree(dev_pl_lj_nat_pdb_dist);

    cudaDeviceSynchronize();

    double *dev_nl_lj_nat_pdb_dist2;
    cudaMalloc((void**)&dev_nl_lj_nat_pdb_dist2, N*sizeof(double));
    cudaMemcpy(dev_nl_lj_nat_pdb_dist2, nl_lj_nat_pdb_dist2, N*sizeof(double), cudaMemcpyHostToDevice);
    double *dev_pl_lj_nat_pdb_dist2;
    cudaMalloc((void**)&dev_pl_lj_nat_pdb_dist2, arrSize*sizeof(double));

    copyElements<<<blocks, threads>>>(dev_nl_lj_nat_pdb_dist2, dev_value, dev_output, dev_pl_lj_nat_pdb_dist2, N);
    cudaDeviceSynchronize();
    free(pl_lj_nat_pdb_dist2);
    pl_lj_nat_pdb_dist2 = (double *)malloc(arrSize*sizeof(double));
    cudaMemcpy(pl_lj_nat_pdb_dist2, dev_pl_lj_nat_pdb_dist2, arrSize*sizeof(double), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_nl_lj_nat_pdb_dist2);
    cudaFree(dev_pl_lj_nat_pdb_dist2);

    cudaDeviceSynchronize();

    double *dev_nl_lj_nat_pdb_dist6;
    cudaMalloc((void**)&dev_nl_lj_nat_pdb_dist6, N*sizeof(double));
    cudaMemcpy(dev_nl_lj_nat_pdb_dist6, nl_lj_nat_pdb_dist6, N*sizeof(double), cudaMemcpyHostToDevice);
    double *dev_pl_lj_nat_pdb_dist6;
    cudaMalloc((void**)&dev_pl_lj_nat_pdb_dist6, arrSize*sizeof(double));

    copyElements<<<blocks, threads>>>(dev_nl_lj_nat_pdb_dist6, dev_value, dev_output, dev_pl_lj_nat_pdb_dist6, N);
    cudaDeviceSynchronize();
    free(pl_lj_nat_pdb_dist6);
    pl_lj_nat_pdb_dist6 = (double *)malloc(arrSize*sizeof(double));
    cudaMemcpy(pl_lj_nat_pdb_dist6, dev_pl_lj_nat_pdb_dist6, arrSize*sizeof(double), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_nl_lj_nat_pdb_dist6);
    cudaFree(dev_pl_lj_nat_pdb_dist6);

    cudaDeviceSynchronize();

    double *dev_nl_lj_nat_pdb_dist12;
    cudaMalloc((void**)&dev_nl_lj_nat_pdb_dist12, N*sizeof(double));
    cudaMemcpy(dev_nl_lj_nat_pdb_dist12, nl_lj_nat_pdb_dist12, N*sizeof(double), cudaMemcpyHostToDevice);
    double *dev_pl_lj_nat_pdb_dist12;
    cudaMalloc((void**)&dev_pl_lj_nat_pdb_dist12, arrSize*sizeof(double));

    copyElements<<<blocks, threads>>>(dev_nl_lj_nat_pdb_dist12, dev_value, dev_output, dev_pl_lj_nat_pdb_dist12, N);
    cudaDeviceSynchronize();
    free(pl_lj_nat_pdb_dist12);
    pl_lj_nat_pdb_dist12 = (double *)malloc(arrSize*sizeof(double));
    cudaMemcpy(pl_lj_nat_pdb_dist12, dev_pl_lj_nat_pdb_dist12, arrSize*sizeof(double), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_nl_lj_nat_pdb_dist12);
    cudaFree(dev_pl_lj_nat_pdb_dist12);

    cudaDeviceSynchronize();

    cudaFree(dev_value);
    cudaFree(dev_output);

    return arrSize;
}


int compact_non_native_pl(int *ibead_neighbor_list_rep, int *jbead_neighbor_list_rep, int *itype_neighbor_list_rep, int *jtype_neighbor_list_rep, int *value, int N, 
                    int *&ibead_pair_list_rep, int *&jbead_pair_list_rep, int *&itype_pair_list_rep, int *&jtype_pair_list_rep){
    // Declare pointers for dev_output and dev_value arrays
    int *dev_output;
    int *dev_value;

    // Calculate array size
    int size = N * sizeof(int);

    // Allocate dev_value and dev_output arrays
    cudaMalloc((void**)&dev_value, size);
    cudaMalloc((void**)&dev_output, size);
 
    // Copy data from value array to device (dev_value)
    cudaMemcpy(dev_value, value, size, cudaMemcpyHostToDevice);

    // Perform hierarchical Kogge-Stone scan on dev_value array and store result in dev_output
    hier_ks_scan(dev_value, dev_output, N, 0);

    // Copy size of compacted array from device to host and store in arrSize
    /* 
     * TODO: If the entire array has 1 as the value, an exclusive scan will have N-1 as the last value in the array.
     * However, allocating an array with N-1 entries will not store all N values from the index array.
     * Change code to determine when we need to increment arrSize and when we don't.
     * Options include:
     *  1) Changing the hierarchical scan kernel to determine if the final value in the value array is 1
     *  2) Checking to see if the final value is 1 in the value array
     * Option 2 was selected, but please double-check this approach
     */ 
    int arrSize;
    cudaMemcpy(&arrSize, &dev_output[N-1], sizeof(int), cudaMemcpyDeviceToHost); 

    // Increment arrSize by 1 if needed
    if(value[N-1]){
        arrSize++;
    }

    int threads = (int)min(N, SECTION_SIZE);
    int blocks = (int)ceil(1.0*N/SECTION_SIZE);

    // Declare and allocate dev_result array to store compacted indices on device (on GPU)
    int *dev_ibead_neighbor_list_rep;
    cudaMalloc((void**)&dev_ibead_neighbor_list_rep, N*sizeof(int));
    cudaMemcpy(dev_ibead_neighbor_list_rep, ibead_neighbor_list_rep, N*sizeof(int), cudaMemcpyHostToDevice);
    int *dev_ibead_pair_list_rep;
    cudaMalloc((void**)&dev_ibead_pair_list_rep, arrSize*sizeof(int));

    copyElements<<<blocks, threads>>>(dev_ibead_neighbor_list_rep, dev_value, dev_output, dev_ibead_pair_list_rep, N);
    cudaDeviceSynchronize();
    free(ibead_pair_list_rep);
    ibead_pair_list_rep = (int *)malloc(arrSize*sizeof(int));
    cudaMemcpy(ibead_pair_list_rep, dev_ibead_pair_list_rep, arrSize*sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(dev_ibead_neighbor_list_rep);
    cudaFree(dev_ibead_pair_list_rep);

    cudaDeviceSynchronize();

    
    int *dev_jbead_neighbor_list_rep;
    cudaMalloc((void**)&dev_jbead_neighbor_list_rep, N*sizeof(int));
    cudaMemcpy(dev_jbead_neighbor_list_rep, jbead_neighbor_list_rep, N*sizeof(int), cudaMemcpyHostToDevice);
    int *dev_jbead_pair_list_rep;
    cudaMalloc((void**)&dev_jbead_pair_list_rep, arrSize*sizeof(int));

    copyElements<<<blocks, threads>>>(dev_jbead_neighbor_list_rep, dev_value, dev_output, dev_jbead_pair_list_rep, N);
    cudaDeviceSynchronize();
    free(jbead_pair_list_rep);
    jbead_pair_list_rep = (int *)malloc(arrSize*sizeof(int));
    cudaMemcpy(jbead_pair_list_rep, dev_jbead_pair_list_rep, arrSize*sizeof(int), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_jbead_neighbor_list_rep);
    cudaFree(dev_jbead_pair_list_rep);

    cudaDeviceSynchronize();


    int *dev_itype_neighbor_list_rep;
    cudaMalloc((void**)&dev_itype_neighbor_list_rep, N*sizeof(int));
    cudaMemcpy(dev_itype_neighbor_list_rep, itype_neighbor_list_rep, N*sizeof(int), cudaMemcpyHostToDevice);
    int *dev_itype_pair_list_rep;
    cudaMalloc((void**)&dev_itype_pair_list_rep, arrSize*sizeof(int));

    copyElements<<<blocks, threads>>>(dev_itype_neighbor_list_rep, dev_value, dev_output, dev_itype_pair_list_rep, N);
    cudaDeviceSynchronize();
    free(itype_pair_list_rep);
    itype_pair_list_rep = (int *)malloc(arrSize*sizeof(int));
    cudaMemcpy(itype_pair_list_rep, dev_itype_pair_list_rep, arrSize*sizeof(int), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_itype_neighbor_list_rep);
    cudaFree(dev_itype_pair_list_rep);

    cudaDeviceSynchronize();


    int *dev_jtype_neighbor_list_rep;
    cudaMalloc((void**)&dev_jtype_neighbor_list_rep, N*sizeof(int));
    cudaMemcpy(dev_jtype_neighbor_list_rep, jtype_neighbor_list_rep, N*sizeof(int), cudaMemcpyHostToDevice);
    int *dev_jtype_pair_list_rep;
    cudaMalloc((void**)&dev_jtype_pair_list_rep, arrSize*sizeof(int));

    copyElements<<<blocks, threads>>>(dev_jtype_neighbor_list_rep, dev_value, dev_output, dev_jtype_pair_list_rep, N);
    cudaDeviceSynchronize();
    free(jtype_pair_list_rep);
    jtype_pair_list_rep = (int *)malloc(arrSize*sizeof(int));
    cudaMemcpy(jtype_pair_list_rep, dev_jtype_pair_list_rep, arrSize*sizeof(int), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    cudaFree(dev_jtype_neighbor_list_rep);
    cudaFree(dev_jtype_pair_list_rep);

    cudaDeviceSynchronize();

    cudaFree(dev_value);
    cudaFree(dev_output);

    return arrSize;
}