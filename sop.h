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
void underdamped_iteration(float3*);
void overdamped_iteration(float3*);
void calculate_observables(float3*);
// void init_crowder_config();
// void save_init_crowder_config();
// void generator_warmup(float);
void check_distances();

void run_pair_list_update();
void run_cell_list_update();
void run_neighbor_list_update();

void underdamped_update_pos();
void underdamped_update_pos_gpu();
__global__ void underdamped_update_pos_kernel(float3 *dev_vel, float3 *dev_force, float3 *dev_pos, float3 *dev_unc_pos, float3 *dev_incr, float a1, float a2, float boxl, int N);

void underdamped_update_vel();
void underdamped_update_vel_gpu();
__global__ void underdamped_update_vel_kernel(float3 *dev_vel, float3 *dev_force, float3 *dev_incr, float a3, float a4, int N);

#endif /* SOP_H */
