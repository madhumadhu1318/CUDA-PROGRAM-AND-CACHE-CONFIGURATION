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
