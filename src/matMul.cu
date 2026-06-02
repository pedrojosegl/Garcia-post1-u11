#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matMulNaive(const float *A, const float *B,
                             float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}

__global__ void matMulTiled(const float *A, const float *B,
                             float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        sA[threadIdx.y][threadIdx.x] =
            (row < N && t*TILE + threadIdx.x < N)
            ? A[row*N + t*TILE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (col < N && t*TILE + threadIdx.y < N)
            ? B[(t*TILE + threadIdx.y)*N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row*N + col] = sum;
}

void matMulCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++)
                s += A[i*N+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

void benchmark(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A       = (float*)malloc(bytes);
    float *h_B       = (float*)malloc(bytes);
    float *h_C_cpu   = (float*)malloc(bytes);
    float *h_C_tiled = (float*)malloc(bytes);

    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)(i % 10) * 0.1f;
        h_B[i] = (float)(i % 7)  * 0.2f;
    }

    matMulCPU(h_A, h_B, h_C_cpu, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);
    cudaEvent_t s, e; float ms;
    cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s);
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    float naive_ms = ms;

    cudaEventRecord(s);
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    float tiled_ms = ms;
    cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost);

    int err_tiled = 0;
    for (int i = 0; i < N*N; i++)
        if (fabs(h_C_tiled[i] - h_C_cpu[i]) > 1e-3f) err_tiled++;

    printf("N=%d | Naive: %.2f ms | Tiled: %.2f ms | Speedup: %.2fx | Errores: %d\n",
           N, naive_ms, tiled_ms, naive_ms/tiled_ms, err_tiled);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_tiled);
}

int main() {
    benchmark(512);
    benchmark(1024);
    return 0;
}
