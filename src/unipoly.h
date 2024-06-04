#ifndef UNIPOLY_H
#define UNIPOLY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void unipoly_populate(int *const a, const int length);
void unipoly_print(const int *const a, const int length);
int unipoly_equal(const int *const a, const int *const b, const int length);
void unipoly_multiply_cpu(const int *const a, const int *const b, int *const c, const int length);
void unipoly_multiply_cpu_parametric_1(const int *const a, const int *const b, int *const c, const int length);
void unipoly_multiply_cpu_parametric_2(const int *const a, const int *const b, int *const c, const int length, const int B);
__global__ void unipoly_multiply_gpu_parametric_1(int *a, int *b, int *c, int n);
__global__ void unipoly_multiply_gpu_parametric_2(int *a, int *b, int *c, int n, int B); // This is the one that matters

#endif
