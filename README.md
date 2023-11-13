# CUDA_PROGRAM & CACHE_CONFIGURATION
# Assignment1:CUDA_Programing
## NxN Matrix multiplication
### CODE
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 2

__global__ void matrixMult(int* A, int* B, int* C, int n)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    int Pvalue = 0;
    for (int k = 0; k < n; k++)
    {
        Pvalue += A[i * n + k] * B[k * n + j];
    }
    C[i * n + j] = Pvalue;
}

int main()
{
    int A[N * N] = { 6, 7,8, 2 };
    int B[N * N] = { 5, 6, 7, 8 };
    int C[N * N] = { 0, 0, 0, 0 };
    int* dev_A = NULL, * dev_B = NULL, * dev_C = NULL;

    cudaError_t err = cudaMalloc((void**)&dev_A, N * N * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating memory for dev_A: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&dev_B, N * N * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating memory for dev_B: %s\n", cudaGetErrorString(err));
        cudaFree(dev_A);
        return -1;
    }

    err = cudaMalloc((void**)&dev_C, N * N * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("Error allocating memory for dev_C: %s\n", cudaGetErrorString(err));
        cudaFree(dev_A);
        cudaFree(dev_B);
        return -1;
    }

    err = cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Error copying A to device: %s\n", cudaGetErrorString(err));
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);
        return -1;
    }

    err = cudaMemcpy(dev_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Error copying B to device: %s\n", cudaGetErrorString(err));
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);
        return -1;
    }

    dim3 threadsPerBlock(N, N);
    matrixMult << <1, threadsPerBlock >> > (dev_A, dev_B, dev_C, N);

    err = cudaMemcpy(C, dev_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Error copying C from device: %s\n", cudaGetErrorString(err));
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);
        return -1;
    }

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    printf("Matrix A:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", A[i * N + j]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", B[i * N + j]);
        }
        printf("\n");
    }
    printf("Resultant Matrix C:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}

```
### NOTE:

![1](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/ec604ec9-94d0-4a8f-b113-b87145455382)

ALLOCATE MEMORY ON GPU FOR dev_A,dev_B,dev_C using cudaMalloc

![2](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/b272423c-e171-4a9a-b372-dcec065548ed)

COPIES THE MATRICES FROM THE HOST TO GPU USING cudaMemcpy( ).

![3](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/c374b82f-f885-41cb-a7cf-662e6a59659f)

DEFINES THE NUMBER OF  THREADS PER BLOCK AS 2D BLOCK WITH DIM NxN

![4](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/903aa202-682e-4943-bacc-a9d0157f8743)

 COPIES MATRIX C FROM GPU TO HOST CPU USING cudaMemcpy( ).

### OUTPUT:

 1.
 
 ![5](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/633c7234-c73c-462f-b3da-2b7c937fdcef)

2.

 ![6](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/0f0ffbb6-6a03-4ecd-af04-cedc3bd4d275)


# Assignment2:CACHE CONFIGURATION

## CODE:

```
#include <stdio.h>
#include <stdlib.h>
int main()
{
    int a[10][10], b[10][10], mul[10][10], r, c, i, j, k;
    system("cls");
    printf("enter the number of row=");
    scanf("%d", &r);
    printf("enter the number of column=");
    scanf("%d", &c);
    printf("enter the first matrix element=\n");
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            scanf("%d", &a[i][j]);
        }
    }
    printf("enter the second matrix element=\n");
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            scanf("%d", &b[i][j]);
        }
    }

    printf("multiply of the matrix=\n");
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            mul[i][j] = 0;
            for (k = 0; k < c; k++)
            {
                mul[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    // for printing result
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            printf("%d\t", mul[i][j]);
        }
        printf("\n");
    }
    return 0;
}

```

# OUTPUT 

![7](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/f46ec006-c03b-4741-b9dd-c7c264c671c2)

## 1.Configure a 64-entry 8-word direct mapped cache.

![8](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/fb1b99a1-bbd1-46bb-9a8b-23a56f652472)

## 2.Configure a 16-entry 4-word 4-way set-associative cache.

![9](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/bbb97f45-e82d-4fd6-b6ce-1009df4cfae0)

## 3.	Configure a 16-entry 2-word 4-way set-associative cache with write-through.

![10](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/0d561e90-ea01-4fde-ba73-2e9e5d387dfa)

## 4.Configure a 64-entry 8-word fully associative cache with Least Recently Used replacement policy and report the numbers. Change the replacement policy to Random and report the numbers for the same cache.

### 1.	LRU

![11](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/2ec96cf9-8fa5-4367-93be-22fa006a94e4)

### 2.RANDOM

![12](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/c9e53daa-3a0d-4c0c-ae1a-debac1ed6987)

## 5.Configure a 32-entry 2-word direct-mapped cache and plot a graph using the plot configuration with numerator as Hits and denominator as Access count. Explain why the number of hits increases and decreases back down before increasing again.

![13](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/1bfa9679-b589-456c-a2ed-111dfed0bfcb)

In a 32-entry 2-word direct-mapped cache, the number of hits increases and decreases back down before increasing again because of cache conflicts. When a program accesses memory blocks that map to the same cache line, the cache can only hold one of the blocks at a time, so it will evict one and fetch the other, resulting in a cache miss. As the program continues to access these memory blocks, the cache will alternate between holding one block and the other, resulting in alternating hits and misses on the cache line. Therefore, the number of hits may increase up to a maximum of 16 per cache line (since there are only 16 cache lines available for each set of two memory blocks that map to the same cache line), but will eventually decrease due to evictions and cache misses, before increasing again when the program accesses the memory blocks that were evicted earlier.

## 6. Configure a 16-entry 4-word 2-way set-associative cache with write-back and write allocate and report the numbers. For the same configuration of cache, use write-through and no write allocate and report the numbers. Explain the differences between the two cache configurations.

### 1. WITH WRITE BACK AND WRITE ALLOCATE

![14](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/d138bd59-d98f-49a8-91a5-a269ee1958df)

### 2.WITH WRITE THROUGH AND NO WRITE ALLOCATE 

![15](https://github.com/madhumadhu1318/CUDA-PROGRAM-AND-CACHE-CONFIGURATION/assets/90201844/5ca57866-2d76-4443-940a-58d8c03e2732)

Cache with write-back and write-allocate policies, write operations are first written to the cache and marked as "dirty", but not immediately written to main memory. When a cache line containing dirty data is evicted from the cache, the dirty data is then written back to main memory. Write-allocate means that when a write operation misses in the cache, the entire cache block is brought into the cache before the write is performed.

 In a Cache with write-through and no-write-allocate policies, write operations are immediately written both to the cache and main memory. No-write-allocate means that when a write operation misses in the cache, the write is directly performed in main memory without bringing the block into the cache.















