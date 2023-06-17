#include <iostream>
#include <vector>
#include <queue>
#include <cuda_runtime.h>

__global__ void breadthFirstSearch(const int* graph, bool* visited, int startNode, int numNodes, double* executionTime) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == startNode) {
        visited[tid] = true;
        printf("Visiting node %d\n", tid);


        for (int i = 0; i < numNodes; ++i) {
            if (graph[tid * numNodes + i] == 1 && !visited[i]) {
                visited[i] = true;
                printf("Visiting node %d\n", i);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        *executionTime = clock() / static_cast<double>(CLOCKS_PER_SEC);
    }
}

int main() {
    int numNodes = 6;
    std::vector<int> graph = {
        0, 1, 1, 0, 0, 0,
        1, 0, 0, 1, 0, 0,
        1, 0, 0, 1, 1, 0,
        0, 1, 1, 0, 1, 1,
        0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 0, 0
    };

    int startNode = 0;

    int* deviceGraph;
    bool* deviceVisited;
    double* deviceExecutionTime;

    cudaMalloc((void**)&deviceGraph, numNodes * numNodes * sizeof(int));
    cudaMemcpy(deviceGraph, graph.data(), numNodes * numNodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&deviceVisited, numNodes * sizeof(bool));
    cudaMemset(deviceVisited, false, numNodes * sizeof(bool));

    cudaMalloc((void**)&deviceExecutionTime, sizeof(double));
    cudaMemset(deviceExecutionTime, 0, sizeof(double));

    breadthFirstSearch<<<1, numNodes>>>(deviceGraph, deviceVisited, startNode, numNodes, deviceExecutionTime);

    double executionTime;
    cudaMemcpy(&executionTime, deviceExecutionTime, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Execution Time: " << executionTime << " seconds" << std::endl;

    cudaFree(deviceGraph);
    cudaFree(deviceVisited);
    cudaFree(deviceExecutionTime);

    return 0;
}
