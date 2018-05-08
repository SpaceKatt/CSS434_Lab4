/**
 * I wrote, compiled, and ran this code on the cssgpu01 machine.
 * Which I believe runs Ubuntu 16.04
 *
 * There appeared to be other intensive computations happening, which may
 * have slowed my execution output.
 *
 * To compile:
 *   nvcc vector.cu -o vector.out
 *
 * To run:
 *   ./vector.out <vector_size>
 */

#include <time.h>
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_NUM 5000

/** -------------------------------------------------------------------------
 * add 
 * Adds two vectors in parallel on a GPU
 *
 * @param a The first vector to add
 * @param b The second vector to add
 * @param c The vector which will hold the result of the addition
 */
__global__ void add(int* a, int* b, int* c) {
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints(int* a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = rand() % MAX_NUM;
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: ./vector <vector_size>" << std::endl;
    return 1;
  }

  int* a;
  int* b;
  int* c;
  
  int* d_a;
  int* d_b;
  int* d_c;

  int vector_size = atoi(argv[1]);

  int size = vector_size * sizeof(int);

  // Allocate memory on device
  cudaMalloc((void**) &d_a, size);
  cudaMalloc((void**) &d_b, size);
  cudaMalloc((void**) &d_c, size);

  // Initialize vars
  a = new int[size]; random_ints(a, vector_size);
  b = new int[size]; random_ints(b, vector_size);
  c = new int[size];

  // Copy parameters to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Start clock
  clock_t start = clock();

  // Perform vector addition
  add<<<vector_size,1>>>(d_a, d_b, d_c);

  // Copy results back
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  // Stop clock
  clock_t timeElapsed = (clock() - start) / (CLOCKS_PER_SEC / 1000000);

  std::cout << timeElapsed << std::endl;

  // Clean up resources
  delete[] a;
  delete[] b;
  delete[] c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
