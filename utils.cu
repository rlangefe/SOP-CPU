#include <stdio.h>
#include <stdlib.h>

void cudaCheck(){					
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }else{
      printf("Success!\n");
      exit(0);
  }
}