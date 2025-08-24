#include <chrono>
#include <cuda.h>
#include <iostream>
#include <string>
#include <stdlib.h>  

#define MOD 1000000007   // defining the modulus value for the MST cost output
#define BLOCK_SIZE 1024     // block size for kernel launches
using namespace std;

struct Edge {
    int src, dest, weight;
};

__device__ int Swap_Function(int* address, int val) { // This swap function (using atomicCAS) exchanges the value at the given address with 'val' and returns the old value.
    int oldAddress = *address;
    int assumed;
    do {
        assumed = oldAddress;
        oldAddress = atomicCAS(address, assumed, val);
    } while (assumed != oldAddress);
    return oldAddress;
}

__device__ bool Union_Operation(int* parent, int Node1, int Node2) {   // performs the union of two disjoint sets. Returns true if the union was successful, false if both nodes were already in the same set.
    for(;;) {     // loop until an 'union' operation is performed or the nodes are found to be in the same set
        for(;;) {    // performing path compression for Node1
            int p = parent[Node1];
            if (p == Node1) {
                break;
            }
            int newp = parent[p];
            Swap_Function(&parent[Node1], newp);
            Node1 = newp;
        }
    
        for(;;) {     // performing path compression for Node2
            int p = parent[Node2];
            if (p == Node2) {
                break;
            }
            int newp = parent[p];
            Swap_Function(&parent[Node2], newp);
            Node2 = newp;
        }
        
        if (Node1 == Node2) {  
            return false;
        }

        int high = (Node1 > Node2) ? Node1 : Node2;
        int low = (Node1 < Node2) ? Node1 : Node2;
        int old = atomicCAS(&parent[high], high, low);  // link the higher-indexed node to the lower-indexed node.
        if (old == high){
            return true;
        }
        // if another thread have updated the parent, then repeat the process
    }
}

__device__ int adjust_weight(long long base, int factor) {
    int mod_result = (int)((base * factor) % MOD);
    return mod_result;
}

__global__ void EdgeWeight(int* deviceEdges, int* deviceType, int E) {   // kernel to adjust the weight of each edge (based on the terrain type) in parallel
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < E) 
    {
        int index = tid * 3;    // starting index
        int baseWeight = deviceEdges[index + 2];
        long long longBaseweight = baseWeight * 1LL;   // long long to avoid overflow during multiplication operation

        // adjusting weight based on terrain type below
        if (deviceType[tid] == 1) {        // terrain is green
            deviceEdges[index + 2] = adjust_weight(longBaseweight, 2);
        } 
        else if (deviceType[tid] == 2) {   // terrain is traffic
            deviceEdges[index + 2] = adjust_weight(longBaseweight, 5);
        }
        else if (deviceType[tid] == 3) {   // terrain is dept
            deviceEdges[index + 2] = adjust_weight(longBaseweight, 3);
        } 
        else {                             // terrain is normal
            deviceEdges[index + 2] = adjust_weight(longBaseweight, 1);
        }
    }
}

__global__ void precompute(int* deviceParent, unsigned long long* lowestValue, int V, int choice) {     // kernel to initialize two arrays based on the choice (either 0 or 1)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < V) {
        if (choice == 0)
            deviceParent[tid] = tid;     // if choice is 0, the deviceParent array is initialized ina way such that each vertex is its own parent
        else
            lowestValue[tid] = 0xFFFFFFFFFFFFFFFFULL;    // if choice is 1, the lowestValue array for each vertex is initialized to the maximum value (i.e., 0xFFFFFFFFFFFFFFFFULL)
    }
}


__global__ void CheapestOutgoingEdge(int* deviceEdges, int E, int* deviceParent, unsigned long long* lowestValue) {    // kernel to scan each edge and update the lowestValue array (using atomicMin) for the components of its endpoints(if the endpoints belong to different sets)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < E) {
        int index = tid * 3;     // starting index for this edge
        int source = deviceEdges[index];           // Source vertex
        int destination = deviceEdges[index + 1];  // Destination vertex
        int weight = deviceEdges[index + 2];       // Weight of the edge

        int Updated_Source;
        for(;;) {          // path compression for the source vertex
            int p = deviceParent[source];
            if (p == source) {
                Updated_Source = source;
                break;
            }
            int newp = deviceParent[p];
            Swap_Function(&deviceParent[source], newp);
            source = newp;
        }
       
        int Updated_Destination;
        for(;;) {      // path compression for destination vertex
            int p = deviceParent[destination];
            if (p == destination) {
                Updated_Destination = destination;
                break;
            }
            int newp = deviceParent[p];
            Swap_Function(&deviceParent[destination], newp);
            destination = newp;
        }

        if (Updated_Source != Updated_Destination) {
            unsigned long long newWeight = ((unsigned long long) weight) << 32;
            unsigned long long res = ((unsigned int) tid) | newWeight;

            // updating the lowest value for both source and destination 
            atomicMin(&lowestValue[Updated_Source], res);
            atomicMin(&lowestValue[Updated_Destination], res);
        }
    }
}

__global__ void setMinimumCostEdge(int* deviceEdges, int* deviceParent, unsigned long long* lowestValue, int V, int* numTrees, int* totalCost) {     // Kernel to apply the cheapest edge for each vertex. If a candidate is not INF in lowestValue array, it will extract the edge index and retrieve the edge details
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < V) {
        unsigned long long candidate = lowestValue[tid];
        if (candidate != 0xFFFFFFFFFFFFFFFFULL) {
            int edgeIndex = (int)(0xFFFFFFFF & candidate);
            int edgeindex = edgeIndex * 3;            // starting index
            int source = deviceEdges[edgeindex];           // Source vertex
            int destination = deviceEdges[edgeindex + 1];  // Destination vertex
            int weight = deviceEdges[edgeindex + 2];       // Edge weight

            int Updated_Source;
            for(;;) {           // path compression for the source vertex
                int p = deviceParent[source];
                if (p == source) {
                    Updated_Source = source;
                    break;
                }
                int newp = deviceParent[p];
                Swap_Function(&deviceParent[source], newp);
                source = newp;
            }
            
            int Updated_Destination;
            for(;;) {           //path compression for the destination vertex
                int p = deviceParent[destination];
                if (p == destination) {
                    Updated_Destination = destination;
                    break;
                }
                int newp = deviceParent[p];
                Swap_Function(&deviceParent[destination], newp);
                destination = newp;
            }
            
            if (Updated_Source != Updated_Destination) {       // if the endpoints belong to different components, perform union operatoin
                if (Union_Operation(deviceParent, Updated_Source, Updated_Destination)) {
                    atomicAdd(totalCost, weight);   // if union successful, then add the edge's weight to total cost
                    atomicAdd(numTrees, -1);        // if union successful, then decrement the number of components
                }
            }
        }
    }
}

void kernelLaunches(int* deviceEdges, int E, int* deviceParent, unsigned long long* lowestValue, int V, int* numTrees, int* totalCost, int numBlocksV, int blocksE) {
    precompute<<<numBlocksV, BLOCK_SIZE>>>(deviceParent, lowestValue, V, 1);    // kernel launch to reinitialize the 'lowestvalue' parent array based on the choice value
    CheapestOutgoingEdge<<<blocksE, BLOCK_SIZE>>>(deviceEdges, E, deviceParent, lowestValue);    // kernel launch to find the minimum cost outgoing edge for each component
    setMinimumCostEdge<<<numBlocksV, BLOCK_SIZE>>>(deviceEdges, deviceParent, lowestValue, V, numTrees, totalCost);     // kernel launch to apply the selected edges to merge components and update the MST total cost
}

int main() {
    int V, E;
    cin >> V >> E;
    
    int* hostEdges = (int*)malloc(E * 3 * sizeof(int));  // each edge is represented by 3 integers (source, destination, weight)
    int* types = (int*)malloc(E * sizeof(int));
    
    // taking input below (edges and terrain type for each edge)
    for (int i = 0; i < E; i++) {
        int u, v, wt;
        string terrainType;
        cin >> u >> v >> wt >> terrainType;
        int type = (terrainType == "green") ? 1 : ((terrainType == "traffic") ? 2 : ((terrainType == "dept") ? 3 : 0));  // Terrain types considered: 0 for normal, 1 for green, 2 for traffic and 3 for dept
        types[i] = type;
        //storing edge information in the 'hostEdges' array.
        hostEdges[i * 3 + 0] = u;
        hostEdges[i * 3 + 1] = v;
        hostEdges[i * 3 + 2] = wt;
    }
    
    // allocating memory for device (edges, terrain types, parent array, lowest edge-value array, and counters for number of trees and total cost)
    int* deviceEdges;
    int* deviceType;
    int* deviceParent;
    unsigned long long* lowestValue;
    int* numTrees;
    int* totalCost;
    
    cudaMalloc(&deviceEdges, E * 3 * sizeof(int));
    cudaMalloc(&deviceType, E * sizeof(int));
    cudaMalloc(&deviceParent, V * sizeof(int));
    cudaMalloc(&lowestValue, V * sizeof(unsigned long long));
    cudaMalloc(&numTrees, sizeof(int));
    cudaMalloc(&totalCost, sizeof(int));
    
    // calculating grid dimensions for the vertices and edges
    int numBlocksV = ceil((float)V / BLOCK_SIZE);
    int blocksE = ceil((float)E / BLOCK_SIZE);
    
    // copying data from host to device 
    cudaMemcpy(deviceEdges, hostEdges, E * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceType, types, E * sizeof(int), cudaMemcpyHostToDevice);
    
    int host_numTrees = V;
    int host_Actual_Cost = 0;
    cudaMemcpy(numTrees, &host_numTrees, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(totalCost, 0, sizeof(int));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EdgeWeight<<<blocksE, BLOCK_SIZE>>>(deviceEdges, deviceType, E);          // kernel launch to adjust edge weights based on terrain types
    precompute<<<numBlocksV, BLOCK_SIZE>>>(deviceParent, lowestValue, V, 0);  // kernel launch to initialize the 'lowestvalue' parent array, since choice is 0, so each vertex will be its own parent
    
    while(host_numTrees > 1) {  // loop until all vertices are merged into a single component
        kernelLaunches(deviceEdges, E, deviceParent, lowestValue, V, numTrees, totalCost, numBlocksV, blocksE);
        cudaMemcpy(&host_numTrees, numTrees, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(&host_Actual_Cost, totalCost, sizeof(int), cudaMemcpyDeviceToHost);   // copying back the final MST cost from device to host
    
    cout << host_Actual_Cost % MOD << endl;
    //cout << "Execution Time: " << elapsed.count() << "s\n";
    
    // free all the allocated device and host memory
    cudaFree(deviceEdges);
    cudaFree(deviceType);
    cudaFree(deviceParent);
    cudaFree(lowestValue);
    cudaFree(numTrees);
    cudaFree(totalCost);
    free(hostEdges);
    free(types);
    
    return 0;
}
