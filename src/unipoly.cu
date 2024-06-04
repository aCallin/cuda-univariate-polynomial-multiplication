#include "unipoly.h"

// Dense representation: {1, 0, -1, 1} corresponds to x^3 - x + 1
void unipoly_populate(int *const a, const int length) {
    for (int i = 0; i < length; i++) {
        int value = (rand() % 3) - 1; // Values: {-1, 0, 1}
        a[i] = value;
    }
}

void unipoly_print(const int *const a, const int length) {
    for (int i = 0; i < length; i++)
        printf("%3d", a[i]);
    printf("\n");
}

int unipoly_equal(const int *const a, const int *const b, const int length) {
    for (int i = 0; i < length; i++) {
        if (a[i] != b[i])
            return 0;
    }
    return 1;
}

void unipoly_multiply_cpu(const int *const a, const int *const b, int *const c, const int length) {
    memset(c, 0, (2 * length - 1) * sizeof(int));
    const int n = length;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i + j] += a[i] * b[j];
        }
    }
}

void unipoly_multiply_cpu_parametric_1(const int *const a, const int *const b, int *const c, const int length) {
    memset(c, 0, (2 * length - 1) * sizeof(int));
    const int n = length - 1;
    for (int p = 0; p <= 2 * n; p++) {
        for (int t = max(0, n - p); t <= min(n, 2 * n - p); t++) {
            c[p] += a[t + p - n] * b[n - t];
        }
    }
}

void unipoly_multiply_cpu_parametric_2(const int *const a, const int *const b, int *const c, const int length, const int B) {
    memset(c, 0, (2 * length - 1) * sizeof(int));
    const int n = length - 1;
    for (int _b = 0; _b <= 2 * n / B; _b++) {
        for (int u = 0; u <= min(B - 1, 2 * n - B * _b); u++) {
            int p = _b * B + u;
            for (int t = max(0, n - p); t <= min(n, 2 * n - p); t++) {
                c[p] += a[t + p - n] * b[n - t];
            }
        }
    }
}

__global__ void unipoly_multiply_gpu_parametric_1(int *a, int *b, int *c, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    for (int t = max(0, n - p); t <= min(n, 2 * n - p); t++) {
        c[p] += a[t + p - n] * b[n - t];
    }
}

__device__ void unipoly_multiply_gpu_parametric_2_inner_loop(int *a, int *b, int *c, int n, int B, int _b) {
    for (int u = 0; u <= min(B - 1, 2 * n - B * _b); u++) {
        int p = _b * B + u;
        for (int t = max(0, n - p); t <= min(n, 2 * n - p); t++) {
            c[p] += a[t + p - n] * b[n - t];
        }
    }
}

__global__ void unipoly_multiply_gpu_parametric_2(int *a, int *b, int *c, int n, int B) {
    int _b = blockIdx.x * blockDim.x + threadIdx.x;
    unipoly_multiply_gpu_parametric_2_inner_loop(a, b, c, n, B, _b);
}
