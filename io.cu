#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "io.h"
#include "global.h"

void read_input(const char* const ifile)
{

  using namespace std;

  ifstream in;
  char line[1024];
  char* tokPtr;
  char term = ';'; // terminates a command
  int newcmd = 1;
  int icmd;
  int nopt_tot = 0;
  int iopt;

  ncmd = 0;
  in.clear();
  in.open(ifile,ios::in);
  while(1) {
    in.getline(line,1024);
    if( in.eof() ) break;
    tokPtr = strtok(line," ");
    if( strchr(tokPtr,term)!=NULL ) { ncmd++; }
    while( tokPtr = strtok(NULL," ") ) {
      if( strchr(tokPtr,term)!=NULL ) { ncmd++; }
    }
  }
  in.close();

  //  cout << "NUMBER OF COMMANDS: " << ncmd << endl;

  in.clear();
  in.open(ifile,ios::in);
  icmd = 0;
  while(1) {
    in.getline(line,1024);
    if( in.eof() ) break;
    tokPtr = strtok(line," ");;
    if( newcmd ) {
      icmd++;
      strcpy( cmd[icmd],tokPtr );
      opt_ptr[icmd] = nopt_tot+1;
      newcmd = 0;
    } else {
      nopt_tot++;
      strcpy( opt[nopt_tot],tokPtr );
    }
    if( strchr(tokPtr,term)!=NULL ) {
      newcmd = 1;
    }
    while( tokPtr = strtok(NULL," ") ) {
      if( newcmd ) {
        icmd++;
        strcpy( cmd[icmd],tokPtr );
        opt_ptr[icmd] = nopt_tot+1;
        newcmd = 0;
      } else {
        nopt_tot++;
        strcpy( opt[nopt_tot],tokPtr );
      }
      if( strchr(tokPtr,term)!=NULL ) {
        newcmd = 1;
      }
    }
  }
  opt_ptr[ncmd+1] = nopt_tot + 1;
  in.close();

  for( int icmd=1; icmd<=ncmd; icmd++ ) {
    for( int i=0; i<strlen(cmd[icmd]); i++ ) {
      if( cmd[icmd][i] == term ) cmd[icmd][i] = '\0';
    }
    //    cout << "COMMAND[" << icmd << "]: " << cmd[icmd] << endl;
    for( int iopt = opt_ptr[icmd]; iopt < opt_ptr[icmd+1]; iopt++ ) {
      for( int i=0; i<strlen(opt[iopt]); i++ ) {
        if( opt[iopt][i] == ';' ) opt[iopt][i] = '\0';
      }
      //      cout << opt[iopt] << endl;
    }
  }

}

void print_sim_params() {

  using namespace std;

  char oline[2048];

  cout << endl;
  sprintf(oline,"+------------------------+");
  cout << oline << endl;
  sprintf(oline,"| Simulation Parameters: |");
  cout << oline << endl;
  sprintf(oline,"+------------------------+");
  cout << oline << endl;

  if (sim_type == 1) {
    sprintf(oline,"Simulation Type                   : %s", "Underdamped");
    cout << oline << endl;
  } else if (sim_type == 2) {
    sprintf(oline,"Simulation Type                   : %s", "Overdamped");
    cout << oline << endl;
  } else {
    cerr << "UNRECOGNIZED SIMULATION TYPE!" << endl;
    exit(-1);
  }

  sprintf(oline,"Simulation Temperature            : %.3f",T);
  cout << oline << endl;

  sprintf(oline,"Start Time Step                   : %.0lf", istep_restart);
  cout << oline << endl;

  sprintf(oline,"Final Time Step                   : %.0lf", nstep);
  cout << oline << endl;

  sprintf(oline,"Output Frequency                  : %d", nup);
  cout << oline << endl;

  sprintf(oline,"Friction Coefficient              : %.0e", zeta);
  cout << oline << endl;

  sprintf(oline,"PBC Box Length                    : %.1f", boxl);
  cout << oline << endl;

  if (neighborlist == 1) {
    sprintf(oline,"Long-range Cutoff Type            : %s", "Neighbor List");
    cout << oline << endl;
    sprintf(oline,"Neighbor List Update Frequency    : %d", nnlup);
    cout << oline << endl;
    sprintf(oline,"Use GPU NL                        : %d", usegpu_nl);
    cout << oline << endl;

    if(usegpu_nl == 1){
      if(!strcmp(nl_algorithm,"RL")){
        sprintf(oline,"Neighbor List Algorithm           : %s", "RL");
        cout << oline << endl;
      }else if(!strcmp(nl_algorithm,"thrust")){
        sprintf(oline,"Neighbor List Algorithm           : %s", "thrust");
        cout << oline << endl;
      }else if(!strcmp(nl_algorithm,"CL")){
        sprintf(oline,"Neighbor List Algorithm           : %s", "CL");
        cout << oline << endl;
      }else{
        sprintf(oline,"INVALID ALGORITHM SELECTION");
        cout << oline << endl;
        exit(-1);
      }
    }
    
    sprintf(oline,"Use GPU PL                        : %d", usegpu_pl);
    cout << oline << endl;

    if(usegpu_pl == 1){
      if(!strcmp(pl_algorithm,"RL")){
        sprintf(oline,"Pair List Algorithm               : %s", "RL");
        cout << oline << endl;
      }else if(!strcmp(pl_algorithm,"thrust")){
        sprintf(oline,"Pair List Algorithm               : %s", "thrust");
        cout << oline << endl;
      }else if(!strcmp(pl_algorithm,"CL")){
        sprintf(oline,"Pair List Algorithm               : %s", "CL");
        cout << oline << endl;
      }else{
        sprintf(oline,"INVALID ALGORITHM SELECTION");
        cout << oline << endl;
        exit(-1);
      }
    }

  } else if (celllist == 1) {
    sprintf(oline,"Long-range Cutoff Type            : %s", "Cell List");
    cout << oline << endl;
    sprintf(oline,"Cell List Update Frequency        : %d", nnlup);
    cout << oline << endl;

    sprintf(oline,"Number of Cells Each Dimension    : %.0lf", ncell);
    cout << oline << endl;
  } else {
    sprintf(oline,"Long-range Cutoff Type            : %s", "None");
    cout << oline << endl;
  }

  sprintf(oline,"Use GPU VDW Energy                : %d", usegpu_vdw_energy);
  cout << oline << endl;
  
  sprintf(oline,"Use GPU VDW Force                 : %d", usegpu_vdw_force);
  cout << oline << endl;

  cout << endl;

}

void save_coords(char* fname,char* fname2)
{

  using namespace std;

  char oline[1024];
  ofstream ofile;
  ofstream ofile2;

  ofile.open(fname,ios::out);
  ofile2.open(fname2,ios::out);
  for( int i=1; i<=nbead; i++ ) {
    sprintf(oline,"%d %f %f %f",i,pos[i].x,
	    pos[i].y,pos[i].z);
    ofile << oline << endl;
    sprintf(oline,"%d %f %f %f",i,unc_pos[i].x,
	    unc_pos[i].y,unc_pos[i].z);
    ofile2 << oline << endl;
  }
  ofile.close();
  ofile2.close();

}

void load_coords(char* fname,char* fname2)
{

  using namespace std;

  char iline[1024];
  ifstream ifile;
  ifstream ifile2;
  char* tokPtr;

  ifile.clear();
  ifile2.clear();
  ifile.open(fname,ios::in);
  ifile2.open(fname2,ios::in);
  for( int i=1; i<=nbead; i++ ) {
    ifile.getline(iline,1024);
    tokPtr = strtok(iline," ");
    tokPtr = strtok(NULL," ");
    pos[i].x = atof(tokPtr);
    tokPtr = strtok(NULL," ");
    pos[i].y = atof(tokPtr);
    tokPtr = strtok(NULL," ");
    pos[i].z = atof(tokPtr);

    ifile2.getline(iline,1024);
    tokPtr = strtok(iline," ");
    tokPtr = strtok(NULL," ");
    unc_pos[i].x = atof(tokPtr);
    tokPtr = strtok(NULL," ");
    unc_pos[i].y = atof(tokPtr);
    tokPtr = strtok(NULL," ");
    unc_pos[i].z = atof(tokPtr);
  }
  ifile.close();
  ifile2.close();

}

void load_vels(char* fname)
{

  using namespace std;

  char iline[1024];
  ifstream ifile;
  char* tokPtr;

  ifile.clear();
  ifile.open(fname,ios::in);
  for( int i=1; i<=nbead; i++ ) {
    ifile.getline(iline,1024);
    tokPtr = strtok(iline," ");
    tokPtr = strtok(NULL," ");
    vel[i].x = atof(tokPtr);
    tokPtr = strtok(NULL," ");
    vel[i].y = atof(tokPtr);
    tokPtr = strtok(NULL," ");
    vel[i].z = atof(tokPtr);

  }
  ifile.close();

}

void save_unccoords(char* fname)
{

  using namespace std;

  char oline[1024];
  ofstream ofile;

  ofile.open(fname,ios::out);
  for( int i=1; i<=nbead; i++ ) {
    sprintf(oline,"%d %f %f %f",i,unc_pos[i].x,
	    unc_pos[i].y,unc_pos[i].z);
    ofile << oline << endl;
  }
  ofile.close();

}

void save_vels(char* fname)
{

  using namespace std;

  char oline[1024];
  ofstream ofile;

  ofile.open(fname,ios::out);
  for( int i=1; i<=nbead; i++ ) {
    sprintf(oline,"%d %f %f %f",i,vel[i].x,
	    vel[i].y,vel[i].z);
    ofile << oline << endl;
  }
  ofile.close();

}

void record_traj(char* fname,char* fname2)
{

  using namespace std;

  char oline[1024];
  char oline2[1024];
  ofstream trajfile;
  ofstream trajfile2;

  trajfile.open(fname,ios::out | ios::app);
  trajfile2.open(fname2,ios::out | ios::app);

  for( int i=1; i<=nbead; i++ ) {
    sprintf(oline,"%f %f %f",pos[i].x,pos[i].y,pos[i].z);
    sprintf(oline2,"%f %f %f",unc_pos[i].x,unc_pos[i].y,unc_pos[i].z);
    trajfile << oline << endl;
    trajfile2 << oline2 << endl;

  }

  trajfile.close();
  trajfile2.close();

}
