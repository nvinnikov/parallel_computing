//#define N 10000000
#define MAX_INT 2147483647
#define SCALE 10000

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void vector_add(float *out, float *a, float *b, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) {
        out[i]  = a[i] + b[i];
    }
}


int main() {
    int N = 1000000000;
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    float time;
    cudaEvent_t start, stop;

    a = (float*)malloc(sizeof(float)*N);
    b = (float*)malloc(sizeof(float)*N);
    out = (float*)malloc(sizeof(float)*N);

    for (int i = 0; i < N; i++) {
        float base = ((float)rand())/MAX_INT;
        a[i] = SCALE*base;
        b[i] = a[i]/5.5;
    }

    cudaMalloc((void**)&d_a, sizeof(float)*N);
    cudaMalloc((void**)&d_b, sizeof(float)*N);
    cudaMalloc((void**)&d_out, sizeof(float)*N);

    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    vector_add <<<(N/128)+1,128>>> (d_out, d_a, d_b, N); //16,32

    cudaEventRecord(stop, 0);


    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&time, start, stop);

    printf("Time spent executing by the GPU: %5.5f ms \n", time);
    printf("First element is: %5.5f\n", out[0]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);

    return 0;


}
