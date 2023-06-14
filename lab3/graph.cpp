#include <iostream>
#include <vector>
#include <queue>
#include <mpi.h>

using namespace std;

void BFS(vector<vector<int>>& graph, int startNode, int rank, int numProcesses) {
    int numNodes = graph.size();
    vector<bool> visited(numNodes, false);

    queue<int> queue;
    visited[startNode] = true;
    queue.push(startNode);

    while (!queue.empty()) {
        int currentNode = queue.front();
        queue.pop();
        cout << "Process " << rank << " visiting node " << currentNode << endl;

        for (int i = 0; i < numNodes; ++i) {
            if (graph[currentNode][i] == 1 && !visited[i]) {
                visited[i] = true;
                queue.push(i);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, numProcesses;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    int numNodes = 6;
    vector<vector<int>> graph = {
        {0, 1, 1, 0, 0, 0},
        {1, 0, 0, 1, 0, 0},
        {1, 0, 0, 1, 1, 0},
        {0, 1, 1, 0, 1, 1},
        {0, 0, 1, 1, 0, 0},
        {0, 0, 0, 1, 0, 0}
    };

    int startNode = 0;

    cout << "Process " << rank << " starting BFS" << endl;
    BFS(graph, startNode, rank, numProcesses);

    MPI_Finalize();

    return 0;
}
