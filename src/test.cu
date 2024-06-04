#include "test.h"

void test_equality() {
    // Test 1: equal
    const int a[] = {0, 1, 0, 1};
    const int b[] = {0, 1, 0, 1};
    const int length_1 = sizeof(a) / sizeof(a[0]);
    printf("a:");
    unipoly_print(a, length_1);
    printf("b:");
    unipoly_print(b, length_1);
    printf("a and b are equal: %d\n", unipoly_equal(a, b, length_1));
    
    // Test 2: not equal
    const int c[] = {1, 1, 0, 1};
    const int d[] = {0, 1, 0, 1};
    const int length_2 = sizeof(c) / sizeof(c[0]);
    printf("c:");
    unipoly_print(c, length_2);
    printf("d:");
    unipoly_print(d, length_2);
    printf("c and d are equal: %d\n", unipoly_equal(c, d, length_2));
}

void test_multiply_cpu(const int length_in) {
    const int length_out = (2 * length_in) - 1;
    int *const a = (int *)malloc(length_in * sizeof(int));
    int *const b = (int *)malloc(length_in * sizeof(int));
    int *const c = (int *)malloc(length_out * sizeof(int));
    unipoly_populate(a, length_in);
    unipoly_populate(b, length_in);
    unipoly_multiply_cpu(a, b, c, length_in);
    printf("a: ");
    unipoly_print(a, length_in);
    printf("b: ");
    unipoly_print(b, length_in);
    printf("c: ");
    unipoly_print(c, length_out);
    free(a);
    free(b);
    free(c);
}

void test_multiply_cpu_parametric_1(const int length_in) {
    const int length_out = (2 * length_in) - 1;
    int *const a = (int *)malloc(length_in * sizeof(int));
    int *const b = (int *)malloc(length_in * sizeof(int));
    int *const c = (int *)malloc(length_out * sizeof(int));
    unipoly_populate(a, length_in);
    unipoly_populate(b, length_in);
    unipoly_multiply_cpu_parametric_1(a, b, c, length_in);
    printf("a: ");
    unipoly_print(a, length_in);
    printf("b: ");
    unipoly_print(b, length_in);
    printf("c: ");
    unipoly_print(c, length_out);
    free(a);
    free(b);
    free(c);
}

void test_multiply_cpu_parametric_2(const int length_in) {
    const int length_out = (2 * length_in) - 1;
    const int B = 32;
    int *const a = (int *)malloc(length_in * sizeof(int));
    int *const b = (int *)malloc(length_in * sizeof(int));
    int *const c = (int *)malloc(length_out * sizeof(int));
    unipoly_populate(a, length_in);
    unipoly_populate(b, length_in);
    unipoly_multiply_cpu_parametric_2(a, b, c, length_in, B);
    printf("a: ");
    unipoly_print(a, length_in);
    printf("b: ");
    unipoly_print(b, length_in);
    printf("c: ");
    unipoly_print(c, length_out);
    free(a);
    free(b);
    free(c);
}

void test_multiply_gpu_parametric_1(const int length_in) {
    // Host
    const int length_out = (2 * length_in) - 1;
    int *const a = (int *)malloc(length_in * sizeof(int));
    int *const b = (int *)malloc(length_in * sizeof(int));
    int *const c = (int *)malloc(length_out * sizeof(int));
    unipoly_populate(a, length_in);
    unipoly_populate(b, length_in);
    memset(c, 0, length_out * sizeof(int));

    // CUDA
    int n = length_in - 1; // Degree
    dim3 dim_grid(2 * n + 1); // Thread block dimensions (p <= 2 * n)
    dim3 dim_block(1); // Thread dimensions
    int *cuda_a;
    int *cuda_b;
    int *cuda_c;
    cudaMalloc((void **)&cuda_a, length_in * sizeof(int));
    cudaMalloc((void **)&cuda_b, length_in * sizeof(int));
    cudaMalloc((void **)&cuda_c, length_out * sizeof(int));
    cudaMemcpy(cuda_a, a, length_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, length_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_c, c, length_out * sizeof(int), cudaMemcpyHostToDevice);
    unipoly_multiply_gpu_parametric_1<<<dim_grid, dim_block>>>(cuda_a, cuda_b, cuda_c, n);
    cudaMemcpy(c, cuda_c, length_out * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    // Back to host
    printf("a: ");
    unipoly_print(a, length_in);
    printf("b: ");
    unipoly_print(b, length_in);
    printf("c: ");
    unipoly_print(c, length_out);
    free(a);
    free(b);
    free(c);
}

void test_multiply_gpu_parametric_2(const int length_in, const int B) {
    // Host
    const int length_out = (2 * length_in) - 1;
    int *const a = (int *)malloc(length_in * sizeof(int));
    int *const b = (int *)malloc(length_in * sizeof(int));
    int *const c = (int *)malloc(length_out * sizeof(int));
    unipoly_populate(a, length_in);
    unipoly_populate(b, length_in);
    memset(c, 0, length_out * sizeof(int));

    // CUDA
    int n = length_in - 1; // Degree
    dim3 dim_grid(2 * n / B + 1); // Thread block dimensions (b <= 2 * n / B)
    dim3 dim_block(1); // Thread dimensions
    int *cuda_a;
    int *cuda_b;
    int *cuda_c;
    cudaMalloc((void **)&cuda_a, length_in * sizeof(int));
    cudaMalloc((void **)&cuda_b, length_in * sizeof(int));
    cudaMalloc((void **)&cuda_c, length_out * sizeof(int));
    cudaMemcpy(cuda_a, a, length_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, length_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_c, c, length_out * sizeof(int), cudaMemcpyHostToDevice);
    unipoly_multiply_gpu_parametric_2<<<dim_grid, dim_block>>>(cuda_a, cuda_b, cuda_c, n, B);
    cudaMemcpy(c, cuda_c, length_out * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    // Back to host
    printf("a: ");
    unipoly_print(a, length_in);
    printf("b: ");
    unipoly_print(b, length_in);
    printf("c: ");
    unipoly_print(c, length_out);
    free(a);
    free(b);
    free(c);
}

void test_1() {
    int n = 64;
    int B = 32; // (2 * n / B + 1) thread blocks

    printf("[ TEST 1 ]\n");
    printf("n: %d\n", n);
    printf("b: %d\n", B);
    printf("Thread blocks: (2n + 1) / B\n\n");

    // CPU setup
    int length_in = n + 1;
    int length_out = (2 * length_in) - 1;
    int *a = (int *)malloc(length_in * sizeof(int));
    int *b = (int *)malloc(length_in * sizeof(int));
    int *c_zero = (int *)malloc(length_out * sizeof(int));
    unipoly_populate(a, length_in);
    unipoly_populate(b, length_in);
    memset(c_zero, 0, length_out * sizeof(int));

    printf("a: ");
    unipoly_print(a, length_in);
    printf("b: ");
    unipoly_print(b, length_in);

    // CPU multiply
    int *c_cpu = (int *)malloc(length_out * sizeof(int));
    unipoly_multiply_cpu(a, b, c_cpu, length_in);
    printf("c_cpu: ");
    unipoly_print(c_cpu, length_out);

    // GPU setup
    int *c_gpu = (int *)malloc(length_out * sizeof(int));
    dim3 dim_grid(2 * n / B + 1); // Thread block dimensions (b <= 2 * n / B)
    dim3 dim_block(1); // Thread dimensions
    int *cuda_a;
    int *cuda_b;
    int *cuda_c;
    cudaMalloc((void **)&cuda_a, length_in * sizeof(int));
    cudaMalloc((void **)&cuda_b, length_in * sizeof(int));
    cudaMalloc((void **)&cuda_c, length_out * sizeof(int));
    cudaMemcpy(cuda_a, a, length_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, length_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_c, c_zero, length_out * sizeof(int), cudaMemcpyHostToDevice);

    // GPU multiply
    unipoly_multiply_gpu_parametric_2<<<dim_grid, dim_block>>>(cuda_a, cuda_b, cuda_c, n, B);
    cudaMemcpy(c_gpu, cuda_c, length_out * sizeof(int), cudaMemcpyDeviceToHost);
    printf("c_gpu: ");
    unipoly_print(c_gpu, length_out);

    // Make sure both are equal
    int equal = unipoly_equal(c_cpu, c_gpu, length_in);
    printf("c_cpu and c_gpu are equal?: %s\n", (equal) ? "true" : "false");

    // Clean up
    cudaFree(cuda_c);
    cudaFree(cuda_b);
    cudaFree(cuda_a);
    free(c_cpu);
    free(c_zero);
    free(b);
    free(a);
}

void test_2() {
    printf("[ TEST 2 ]\n");
    printf("Note: time measurements reflect only the CPU and GPU multipy functions -- it does not account for setup or tear-down time (malloc, free, etc.).\n\n");

    int n_values[] = {16384, 65536}; // 2^14, 2^16
    int B_values[] = {32, 64, 128, 246, 512};

    for (int i = 0; i < sizeof(n_values) / sizeof(n_values[0]); i++) {
        int n = n_values[i];
        int length_in = n + 1;
        int length_out = (2 * length_in) - 1;
        
        printf("n: %d\n", n);

        // Populate polynomials a and b with random values in {-1, 0, 1}
        int *a = (int *)malloc(length_in * sizeof(int));
        int *b = (int *)malloc(length_in * sizeof(int));
        unipoly_populate(a, length_in);
        unipoly_populate(b, length_in);
        
        // Multiply with the CPU to compare GPU results
        int *c_cpu = (int *)malloc(length_out * sizeof(int));
        printf("Multiplying with cpu... ");
        ctimer_t cpu_timer;
        ctimer_start(&cpu_timer);
        unipoly_multiply_cpu(a, b, c_cpu, length_in);
        ctimer_stop(&cpu_timer);
        ctimer_measure(&cpu_timer);
        printf("done (%ld.%09ld sec)\n", (long)cpu_timer.elapsed.tv_sec, cpu_timer.elapsed.tv_nsec);
        free(c_cpu);

        // Have an array of zeros length of polynomial c for memcpy in GPU calculations
        int *c_zero = (int *)malloc(length_out * sizeof(int));
        memset(c_zero, 0, length_out * sizeof(int));

        for (int j = 0; j < sizeof(B_values) / sizeof(B_values[0]); j++) {
            int B = B_values[j];

            // Setup GPU
            dim3 dim_grid(2 * n / B + 1); // Thread block dimensions (b <= 2 * n / B)
            dim3 dim_block(1); // Thread dimensions
            int *cuda_a;
            int *cuda_b;
            int *cuda_c;
            cudaMalloc((void **)&cuda_a, length_in * sizeof(int));
            cudaMalloc((void **)&cuda_b, length_in * sizeof(int));
            cudaMalloc((void **)&cuda_c, length_out * sizeof(int));
            cudaMemcpy(cuda_a, a, length_in * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_b, b, length_in * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_c, c_zero, length_out * sizeof(int), cudaMemcpyHostToDevice);
            
            // Multiply with GPU
            printf("Multiplying with gpu (B: %d)... ", B);
            ctimer_t gpu_timer;
            ctimer_start(&gpu_timer);
            unipoly_multiply_gpu_parametric_2<<<dim_grid, dim_block>>>(cuda_a, cuda_b, cuda_c, n, B);
            ctimer_stop(&gpu_timer);
            ctimer_measure(&gpu_timer);
            printf("done (%ld.%09ld sec)\n", (long)gpu_timer.elapsed.tv_sec, gpu_timer.elapsed.tv_nsec);

            // GPU Clean up
            cudaFree(cuda_a);
            cudaFree(cuda_b);
            cudaFree(cuda_c);
        }

        // CPU clean up
        free(a);
        free(b);
        free(c_zero);

        printf("\n");
    }
}
