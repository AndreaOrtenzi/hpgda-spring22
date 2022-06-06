// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <sstream>
#include <assert.h>
#include "personalized_pagerank.cuh"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

//////////////////////////////
//////////////////////////////

// Write GPU kernel here!

__global__ void gpu_calculate_ppr_0(
    int *cols_idx, 
    int* ptr, 
    double* val,
    double* p,
    int* dangling,
    double* result,
    int pers_ver,
    double alpha,
    int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   
    int start = ptr[idx];
    int end = ptr[idx + 1];

    double prod_fact = 0, dang_fact = 0, pers_fact = 0;

    for (int i = start; i < end; i++) {
        prod_fact += val[i] * p[cols_idx[i]];        
    }

    for (int i = 0; i < V; i++){
        dang_fact += dangling[i] * p[i];
    }

    prod_fact *= alpha;
    dang_fact *= alpha / V;
    if (pers_ver == idx)//for the future preprocess pers_ver in a vector check condition
        pers_fact = (1 - alpha);
    
    //__syncthreads();    atomicAdd(res, sum);  

    result[idx] = prod_fact + dang_fact + pers_fact;   
}

//////////////////////////////
//////////////////////////////

// CPU Utility functions;

// Read the input graph and initialize it;
void PersonalizedPageRank::initialize_graph() {
    // Read the graph from an MTX file;
    int num_rows = 0;
    int num_columns = 0;
    read_mtx(graph_file_path.c_str(), &x, &y, &val,
        &num_rows, &num_columns, &E, // Store the number of vertices (row and columns must be the same value), and edges;
        true,                        // If true, read edges TRANSPOSED, i.e. edge (2, 3) is loaded as (3, 2). We set this true as it simplifies the PPR computation;
        false,                       // If true, read the third column of the matrix file. If false, set all values to 1 (this is what you want when reading a graph topology);
        debug,                 
        false,                       // MTX files use indices starting from 1. If for whatever reason your MTX files uses indices that start from 0, set zero_indexed_file=true;
        true                         // If true, sort the edges in (x, y) order. If you have a sorted MTX file, turn this to false to make loading faster;
    );
    if (num_rows != num_columns) {
        if (debug) std::cout << "error, the matrix is not squared, rows=" << num_rows << ", columns=" << num_columns << std::endl;
        exit(-1);
    } else {
        V = num_rows;
    }
    if (debug) std::cout << "loaded graph, |V|=" << V << ", |E|=" << E << std::endl;

    // Compute the dangling vector. A vertex is not dangling if it has at least 1 outgoing edge;
    dangling.resize(V);
    std::fill(dangling.begin(), dangling.end(), 1);  // Initially assume all vertices to be dangling;
    for (int i = 0; i < E; i++) {
        // Ignore self-loops, a vertex is still dangling if it has only self-loops;
        if (x[i] != y[i]) dangling[y[i]] = 0;
    }
    // Initialize the CPU PageRank vector;
    pr.resize(V);
    pr_golden.resize(V);
    // Initialize the value vector of the graph (1 / outdegree of each vertex).
    // Count how many edges start in each vertex (here, the source vertex is y as the matrix is transposed);
    int *outdegree = (int *) calloc(V, sizeof(int));
    for (int i = 0; i < E; i++) {
        outdegree[y[i]]++;
    }
    // Divide each edge value by the outdegree of the source vertex;
    for (int i = 0; i < E; i++) {
        val[i] = 1.0 / outdegree[y[i]];  
    }
    free(outdegree);
}

//convert COO in CSR
void PersonalizedPageRank::converter(){
    std::vector<int> xPtr;
    int ptr=0,previousX;

    // Matrix:
    // 10 20  0  0  0  0
    //  0 30  0 40  0  0
    //  0  0 50 60 70  0
    //  0  0  0  0  0 80

    // coo data:
    //double coo_val[nnz] = { 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 };
    //int    coo_x[nnz] = { 0, 0, 1, 1, 2, 2, 2, 3 };
    //int    coo_col[nnz] = { 0, 1, 1, 3, 2, 3, 4, 5 };

    // Expected output:
    // csr_val: 10 20 30 40 50 60 70 80
    // csr_col:  0  1  1  3  2  3  4  5
    // csr_x:  0  2  4  7  8

    if(E==0)
        return;
    
    previousX = 0;
    xPtr.push_back(0);

    for (int i =0; i< E; i++) {
        
        while(x[i]!=previousX){
            xPtr.push_back(ptr);
            previousX++;
        }
        ptr++;
    }
    
    for (int i =0; i< V-x[E-1]; i++) {
        xPtr.push_back(ptr); 
    }


    //if (debug){
    //    std::cout << "vettore ptr: ";
    //    for (int i =0; i< xPtr.size(); i++){
    //        std::cout << xPtr[i] << " ";
    //    }
    //    std::cout << "\n";
    //}
    
    x=xPtr;

}

void PersonalizedPageRank::alloc_to_gpu() {
    
    cudaMalloc(&d_x, sizeof(double) * x.size());
    cudaMalloc(&d_y, sizeof(double) * y.size());
    cudaMalloc(&d_val, sizeof(double) * val.size());
    cudaMalloc(&d_dangling, sizeof(double) * dangling.size());
    cudaMalloc(&d_pr, sizeof(double) * V);
    cudaMalloc(&d_newPr, sizeof(double) * V);

    cudaMemcpy(d_x, &x[0], sizeof(double) * x.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y[0], sizeof(double) *  y.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, &val[0], sizeof(double) *  val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dangling, &dangling[0], sizeof(double) * dangling.size(), cudaMemcpyHostToDevice);
    
}


//////////////////////////////
//////////////////////////////

// Allocate data on the CPU and GPU;
void PersonalizedPageRank::alloc() {
    // Load the input graph and preprocess it;
    initialize_graph();
    
    //convert COO in CSR
    converter();

    // Compute the number of blocks for implementations where the value is a function of the input size;
    BlockNum = (V + block_size - 1) / block_size;

    // Allocate any GPU data here;
    alloc_to_gpu();
}

// Initialize data;
void PersonalizedPageRank::init() {
    // Do any additional CPU or GPU setup here;
    // TODO!
}

// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset() {
    // Do any GPU reset here, and also transfer data to the GPU;

    // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);
    pr.clear();
    for (int i=0; i<V;i++){
        pr.push_back(1.0 / V);
    }
    
    
    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V; 
    if (debug) std::cout << "personalization vertex=" << personalization_vertex << std::endl;

    // Reset the result in GPU and Transfer data to the GPU (cudaMemset(d_pr, 1.0 / V, sizeof(double) * V));
    //if it's so stupid we don't need to copy but just set it or even find a way to begin without passing thisdata
    cudaMemcpy(d_pr, &pr[0], sizeof(double) * V, cudaMemcpyHostToDevice);
    
}

void PersonalizedPageRank::personalized_page_rank_0(int iter){
    auto start_tmp = clock_type::now();
    double *d_temp;
    bool converged = false;

    int = 0;
    while ((!converged && i < max_iterations) || i == 30) {
        // Call the GPU computation.
        gpu_calculate_ppr_0<<<1, 17>>>(d_y, d_x, d_val, d_pr, d_dangling, d_newPr, personalization_vertex, alpha, V);
        
        d_temp=d_pr;
        d_pr=d_newPr;
        d_newPr=d_temp;

        //ensure entire pr is calculated
        cudaDeviceSynchronize();

        double err = euclidean_distance(d_pr, d_newPr, V);
        converged = err <= convergence_threshold;
        i++;
    }
    

    // Print performance of GPU, not accounting for transfer time;
    if (debug) {
        // Synchronize computation by hand to measure GPU exec. time;
        cudaDeviceSynchronize();
        auto end_tmp = clock_type::now();
        auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
        std::cout << "  pure GPU execution(" << iter << ")=" << double(exec_time) / 1000 << " ms, " << std::endl;
    }

    // Copy the result from the GPU to the CPU;
    //for the future try order values in GPU and trasfer only first 20 
    cudaMemcpy(&pr[0], d_pr, sizeof(double) * V, cudaMemcpyDeviceToHost);
}

double PersonalizedPageRank::euclidean_distance(double *x, double *y, int N) {
    double result = 0;
    for (int i = 0; i < N; i++) {
        double tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return std::sqrt(result);
}

// Do the GPU computation here, and also transfer results to the CPU;
void PersonalizedPageRank::execute(int iter) {

    switch (implementation)
    {
    case 0:
        personalized_page_rank_0(iter);
        break;
    
    default:
        break;
    }    
    
}

void PersonalizedPageRank::cpu_validation(int iter) {

    // Reset the CPU PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr_golden.begin(), pr_golden.end(), 1.0 / V);

    // Do Personalized PageRank on CPU;
    auto start_tmp = clock_type::now();
    personalized_pagerank_cpu(x.data(), y.data(), val.data(), V, E, pr_golden.data(), dangling.data(), personalization_vertex, alpha, 1e-6, 100);
    auto end_tmp = clock_type::now();
    auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
    std::cout << "exec time CPU=" << double(exec_time) / 1000 << " ms" << std::endl;

    // Obtain the vertices with highest PPR value;
    std::vector<std::pair<int, double>> sorted_pr_tuples = sort_pr(pr.data(), V);
    std::vector<std::pair<int, double>> sorted_pr_golden_tuples = sort_pr(pr_golden.data(), V);

    // Check how many of the correct top-20 PPR vertices are retrieved by the GPU;
    std::unordered_set<int> top_pr_indices;
    std::unordered_set<int> top_pr_golden_indices;
    int old_precision = std::cout.precision();
    std::cout.precision(4);
    int topk = std::min(V, topk_vertices);
    for (int i = 0; i < topk; i++) {
        int pr_id_gpu = sorted_pr_tuples[i].first;
        int pr_id_cpu = sorted_pr_golden_tuples[i].first;
        top_pr_indices.insert(pr_id_gpu);
        top_pr_golden_indices.insert(pr_id_cpu);
        if (debug) {
            double pr_val_gpu = sorted_pr_tuples[i].second;
            double pr_val_cpu = sorted_pr_golden_tuples[i].second;
            if (pr_id_gpu != pr_id_cpu) {
                std::cout << "* error in rank! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            } else if (std::abs(sorted_pr_tuples[i].second - sorted_pr_golden_tuples[i].second) > 1e-6) {
                std::cout << "* error in value! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            }
        }
    }
    std::cout.precision(old_precision);
    // Set intersection to find correctly retrieved vertices;
    std::vector<int> correctly_retrieved_vertices;
    set_intersection(top_pr_indices.begin(), top_pr_indices.end(), top_pr_golden_indices.begin(), top_pr_golden_indices.end(), std::back_inserter(correctly_retrieved_vertices));
    precision = double(correctly_retrieved_vertices.size()) / topk;
    if (debug) std::cout << "correctly retrived top-" << topk << " vertices=" << correctly_retrieved_vertices.size() << " (" << 100 * precision << "%)" << std::endl;
}

std::string PersonalizedPageRank::print_result(bool short_form) {
    if (short_form) {
        return std::to_string(precision);
    } else {
        // Print the first few PageRank values (not sorted);
        std::ostringstream out;
        out.precision(3);
        out << "[";
        for (int i = 0; i < std::min(20, V); i++) {
            out << pr[i] << ", ";
        }
        out << "...]";
        return out.str();
    }
}

void PersonalizedPageRank::clean() {
    // Delete any GPU data or additional CPU data;
    
    //free(cpu_data);
    cudaFree(d_dangling);
    cudaFree(d_pr);
    cudaFree(d_newPr);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y); 
}
