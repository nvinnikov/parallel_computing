#include <iostream>
#include <vector>
#include <queue>

__global__ void breadthFirstSearch(int* graph, bool* visited, int numVertices, int startVertex) {
    visited[startVertex] = true;
    printf("%d ", startVertex);

    std::queue<int> vertexQueue;
    vertexQueue.push(startVertex);

    while (!vertexQueue.empty()) {
        int currentVertex = vertexQueue.front();
        vertexQueue.pop();

        for (int neighbor = 0; neighbor < numVertices; ++neighbor) {
            if (graph[currentVertex * numVertices + neighbor] && !visited[neighbor]) {
                visited[neighbor] = true;
                printf("%d ", neighbor);
                vertexQueue.push(neighbor);
            }
        }
    }
}

int main() {
    int numVertices, numEdges;
    std::cout << "Enter the number of vertices and edges: ";
    std::cin >> numVertices >> numEdges;

    std::vector<int> graph(numVertices * numVertices, 0);

    std::cout << "Enter the edges:\n";
    for (int i = 0; i < numEdges; ++i) {
        int src, dest;
        std::cin >> src >> dest;

        graph[src * numVertices + dest] = 1;
        // Uncomment the line below for an undirected graph
        // graph[dest * numVertices + src] = 1;
    }

    int startVertex;
    std::cout << "Enter the starting vertex: ";
    std::cin >> startVertex;

    int* d_graph;
    bool* d_visited;
    cudaMalloc((void**)&d_graph, numVertices * numVertices * sizeof(int));
    cudaMalloc((void**)&d_visited, numVertices * sizeof(bool));

    cudaMemcpy(d_graph, graph.data(), numVertices * numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_visited, 0, numVertices * sizeof(bool));

    std::cout << "Breadth-first traversal: ";
    breadthFirstSearch<<<1, 1>>>(d_graph, d_visited, numVertices, startVertex);

    cudaFree(d_graph);
    cudaFree(d_visited);

    return 0;
}
