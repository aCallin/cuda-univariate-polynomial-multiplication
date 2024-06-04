#ifndef TEST_H
#define TEST_H

#include "unipoly.h"
#include "ctimer.h"

void test_equality();
void test_multiply_cpu(const int length_in);
void test_multiply_cpu_parametric_1(const int length_in);
void test_multiply_cpu_parametric_2(const int length_in);
void test_multiply_gpu_parametric_1(const int length_in);
void test_multiply_gpu_parametric_2(const int length_in, const int B);
void test_1();
void test_2();

#endif
