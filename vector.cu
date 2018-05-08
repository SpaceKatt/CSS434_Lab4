#include <time.h>
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_NUM 5000

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

  cudaMalloc((void**) &d_a, size);
  cudaMalloc((void**) &d_b, size);
  cudaMalloc((void**) &d_c, size);

  a = new int[size]; random_ints(a, vector_size);
  b = new int[size]; random_ints(b, vector_size);
  c = new int[size];

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  clock_t start = clock();

  add<<<vector_size,1>>>(d_a, d_b, d_c);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  clock_t timeElapsed = (clock() - start) / (CLOCKS_PER_SEC / 1000000);

  std::cout << timeElapsed << std::endl;

  delete[] a;
  delete[] b;
  delete[] c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
