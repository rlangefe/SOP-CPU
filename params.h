#ifndef PARAMS_H
#define PARAMS_H

void load(int);
void set_temp(float);
void alloc_arrays();
void release_bonds();
void init_bonds(int);
void release_angles();
void init_angles(int);
void init_pos(int);
void release_lj();
void init_lj(int,int);
void set_params(int);

#endif
