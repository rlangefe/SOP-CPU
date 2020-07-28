#ifndef SOP_H
#define SOP_H

#include "global.h"
#include "GPUvars.h"

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
// void init_native();
void underdamped_iteration(double3*);
void overdamped_iteration(double3*);
void calculate_observables(double3*);
// void init_crowder_config();
// void save_init_crowder_config();
// void generator_warmup(double);
void check_distances();

void run_pair_list_update();
void run_cell_list_update();
void run_neighbor_list_update();

void underdamped_update_pos();
void underdamped_update_pos_gpu();
__global__ void underdamped_update_pos_kernel(double3 *dev_vel, double3 *dev_force, double3 *dev_pos, double3 *dev_unc_pos, double3 *dev_incr, double a1, double a2, double boxl, int N);

void underdamped_update_vel();
void underdamped_update_vel_gpu();
__global__ void underdamped_update_vel_kernel(double3 *dev_vel, double3 *dev_force, double3 *dev_incr, double a3, double a4, int N);

#endif /* SOP_H */
