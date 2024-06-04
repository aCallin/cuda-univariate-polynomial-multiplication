# cuda-univariate-polynomial-multiplication
Univariate polynomial multiplication using CUDA.

The program runs a suite of correctness and performance tests for computing univariate polynomial multiplication on the CPU and the GPU. For the GPU, various thread block sizes are experimented with and a different algorithm is used than the CPU to maximize parallelization and performance.

## Building
`make` to build the program  
`make run` to build and execute collectively  
`make clean` to remove build files  
  
Prerequisites: a CUDA-capable GPU and the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

## Results
Sample results from the program can be found in *sample-output.txt*.
