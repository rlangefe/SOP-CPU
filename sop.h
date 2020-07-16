#ifndef SOP_H
#define SOP_H

#include "global.h"

const int nstck_ang = 4;
const int nstck_dist = 2;
const int nstck_tor = 2;

extern class Ran_Gen generator;

void ex_cmds(); // sequentially execute cmds found in input_file
// void release_torsions();
// void init_torsions(int);
// void release_stacks();
void simulation_ctrl();
void underdamped_ctrl();
void overdamped_ctrl();
void run_ctrl();
// void init_native();
void underdamped_iteration(coord*);
void overdamped_iteration(coord*);
void calculate_observables(coord*);
// void init_crowder_config();
// void save_init_crowder_config();
// void generator_warmup(double);
void check_distances();

void run_pair_list_update();
void run_cell_list_update();
void run_neighbor_list_update();

#endif /* SOP_H */
