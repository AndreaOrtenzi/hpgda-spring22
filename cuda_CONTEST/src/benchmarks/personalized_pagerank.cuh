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

#pragma once
#include <set>
#include <list>
#include <mutex>
#include <iterator>
#include "../benchmark.cuh"

// CPU Utility functions;

inline void spmv_coo_cpu(const int *x, const int *y, const double *val, const double *vec, double *result, int N) {
    for (int i = 0; i < N; i++) {
        result[x[i]] += val[i] * vec[y[i]];
    }
}

inline double dot_product_cpu(const int *a, const double *b, const int N) {
    double result = 0;
    for (int i = 0; i < N; i++) {
        result += a[i] * b[i];
    }
    return result;
}

inline void axpb_personalized_cpu(
    double alpha, double *x, double beta,
    const int personalization_vertex, double *result, const int N) {
    double one_minus_alpha = 1 - alpha;
    for (int i = 0; i < N; i++) {
        result[i] = alpha * x[i] + beta + ((personalization_vertex == i) ? one_minus_alpha : 0.0);
    }
}

inline double euclidean_distance_cpu(const double *x, const double *y, const int N) {
    double result = 0;
    for (int i = 0; i < N; i++) {
        double tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return std::sqrt(result);
}

inline void personalized_pagerank_cpu(
    const int *x,
    const int *y,
    const double *val,
    const int V, 
    const int E,
    double *pr,
    const int *dangling_bitmap, 
    const int personalization_vertex,
    double alpha=DEFAULT_ALPHA,
    double convergence_threshold=DEFAULT_CONVERGENCE,
    const int max_iterations=DEFAULT_MAX_ITER) {

    // Temporary PPR result;
    double *pr_tmp = (double *) malloc(sizeof(double) * V);

    int iter = 0;
    bool converged = false;
    while (!converged && iter < max_iterations) {    
        memset(pr_tmp, 0, sizeof(double) * V);
        spmv_coo_cpu(x, y, val, pr, pr_tmp, E);
        double dangling_factor = dot_product_cpu(dangling_bitmap, pr, V); 
        axpb_personalized_cpu(alpha, pr_tmp, alpha * dangling_factor / V, personalization_vertex, pr_tmp, V);

        // Check convergence;
        double err = euclidean_distance_cpu(pr, pr_tmp, V);
        converged = err <= convergence_threshold;

        // Update the PageRank vector;
        memcpy(pr, pr_tmp, sizeof(double) * V);
        iter++;
    }
    free(pr_tmp);
}

inline std::vector<std::pair<int, double>> sort_pr(double *pr, int V) {
	std::vector<std::pair<int, double>> sorted_pr;
    // Associate PR values to the vertex indices;
	for (int i = 0; i < V; i++) {
		sorted_pr.push_back( { i, pr[i] });
	}
    // Sort the tuples (vertex, PR) by decreasing value of PR;
	std::sort(sorted_pr.begin(), sorted_pr.end(), [](const std::pair<int, double> &l, const std::pair<int, double> &r) {
		if (l.second != r.second) return l.second > r.second;
		else return l.first > r.first;
	});
	return sorted_pr;
}

class PersonalizedPageRank : public Benchmark {
   public:
    PersonalizedPageRank(Options &options) : Benchmark(options) {
        alpha = options.alpha;
        max_iterations = options.maximum_iterations;
        convergence_threshold = options.convergence_threshold;
        graph_file_path = options.graph;
    }
    void alloc();
    void init();
    void reset();
    void execute(int iter);
    void clean();
    void cpu_validation(int iter);
    std::string print_result(bool short_form = false);

   private:
    int V = 0;
    int E = 0;
    int BlockNum = 0;

    std::vector<int> x;       // Source coordinate of edges in graph;
    std::vector<int> y;       // Destination coordinate of edges in graph;
    std::vector<double> val;  // Used for matrix value, initially all values are 1;
    std::vector<int> dangling;
    std::vector<double> pr;   // Store here the PageRank values computed by the GPU;
    std::vector<double> pr_golden;  // PageRank values computed by the CPU;

    // Implementation 0
    std::vector<int> convertedX;
    std::vector<double> newPr;

    // Implementation 1
    std::vector<float> val_f;
    std::vector<float> pr_f;
    std::vector<float> newPr_f;

    // Implementation 2
    std::vector<int> processedX;
    std::vector<int> processedXShared;
    std::vector<int> processedY;
    std::vector<float> processedVal; //change in float
    //std::vector<int> beginning_of_warp_data;
    std::vector<int> beginning_of_blocks;
    std::vector<int> last_write_length;
    std::vector<int> writings_of_blocks;
    std::list<int> writings_of_blocks_list;
    int remaining_places_in_shared_mem;
    //int num_of_warp_in_block;

    int personalization_vertex = 0;
    double convergence_threshold = DEFAULT_CONVERGENCE;
    double alpha = DEFAULT_ALPHA;
    int max_iterations = DEFAULT_MAX_ITER;
    int topk_vertices = 20;   // Number of highest-ranked vertices to look for;
    double precision = 0;     // How many top-20 vertices are correctly retrieved;
    std::string graph_file_path = DEFAULT_GRAPH;

    ////kernel pointers
    // Implementation 0
    int *d_x, *d_y;
    double *d_val,*d_pr,*d_newPr;

    // Implementation 1
    float *d_val_f, *d_pr_f, *d_newPr_f, *d_diff_f, *d_err_sum;
    float err_sum;

    // Implementation 2
    int *d_beginning_of_blocks,*d_x_shared,*d_writings_of_blocks,*d_last_write_length;

    // Implementation 3
    int *d_dangling;
    float* d_dang_res;
    

    void initialize_graph();
    void converter();
    void pre_processing_coo_graph();

    void alloc_to_gpu_0();
    void alloc_to_gpu_1();
    void alloc_to_gpu_2();
    void alloc_to_gpu_3();
    void alloc_to_gpu_4();

    float euclidean_distance_float(float *x, float *y, int N);
    
    // Implementations of the algorithm;
    void personalized_page_rank_0(int iter);
    void personalized_page_rank_1(int iter);
    void personalized_page_rank_2(int iter);
    void personalized_page_rank_3(int iter);
    void personalized_page_rank_4(int iter);

    void clean_0();
    void clean_1();
    void clean_2();
    void clean_3();
    void clean_4();
 
};