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

#define WARP_SIZE 32
#define BLOCKNUM 151

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

//////////////////////////////
//////////////////////////////

// Write GPU kernel here!

// Used to sum the values in a warp;
__inline__ __device__ double warp_reduce(double val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Note: atomicAdd on double is present only in recent GPU architectures.
// If you don't have it, change the benchmark to use doubles;
__global__ void gpu_vector_sum(float *x, float *res, int N) {
    double sum = double(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i];
    }
    sum = warp_reduce(sum);                    // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0)  // Same as (threadIdx.x % WARP_SIZE) == 0 but faster
        atomicAdd(res, sum);                   // The first thread in the warp updates the output;
}

__global__ void gpu_vector_power_sum (float *pprOld, float *res, float *pprNew,int N,float alpha,double dang_fact, int pers_ver){
    double sum = double(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        pprNew[i] = pprNew[i]*alpha + dang_fact + (!(pers_ver-i))*(1-alpha);
        sum += (pprNew[i]-pprOld[i])*(pprNew[i]-pprOld[i]);
    }
    sum = warp_reduce(sum);
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0)
        atomicAdd(res, sum);     
}

__global__ void gpu_vector_prod(int *x, float *y, float *res, int N) {
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }
    sum = warp_reduce(sum);
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0)
        atomicAdd(res, sum);  
}

__global__ void gpu_calculate_ppr_0(
    int *cols_idx,
    int* ptr,
    double* val,
    double* p,
    double dang_fact,
    double* result,
    int pers_ver,
    double alpha,
    int V)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start = ptr[idx];
    int end = ptr[idx + 1];

    double prod_fact = 0;
    for (int i = start; i < end; i++) {
        prod_fact += val[i] * p[cols_idx[i]];
    }
    prod_fact *= alpha;

    //__syncthreads();    atomicAdd(res, sum);
    result[idx] = prod_fact + dang_fact + (!(pers_ver-idx))*(1-alpha);
}

__global__ void gpu_calculate_ppr_1(
    int *cols_idx,
    int* ptr,
    float* val,
    float* p,
    float dang_fact,
    float* result,
    int pers_ver,
    float alpha,
    int V,
    float* diff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start = ptr[idx];
    int end = ptr[idx + 1];

    float prod_fact = 0;
    for (int i = start; i < end; i++) {
        prod_fact += val[i] * p[cols_idx[i]];
    }
    prod_fact *= alpha;

    
    result[idx] = prod_fact + dang_fact + (!(pers_ver-idx))*(1-alpha);
    diff[idx] = (result[idx] - p[idx]) * (result[idx] - p[idx]);
}

__global__ void gpu_calculate_ppr_2(
    int* cols,
    int* rows,
    float* vals,
    float* ppr,
    float* results,
    int* beginning_of_warp_data)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int warp_num =  static_cast<int>(blockIdx.x * blockDim.x/WARP_SIZE) + static_cast<int>(threadIdx.x/WARP_SIZE);
    const int start = beginning_of_warp_data[warp_num]*WARP_SIZE + threadIdx.x; //*WARP_SIZE because I can overlow int, if it happen we'll change easily here in long
    const int end = beginning_of_warp_data[warp_num+1]*WARP_SIZE;
    int result_index=-1,previous_row,iter_this_warp=beginning_of_warp_data[warp_num+1]-beginning_of_warp_data[warp_num];
    float *partial_results;//a thread can change a line every iteration, this is max allocation. To allocate less space in private mem put in global a vector of num of row processed by each thread
    int *partial_results_row;

    extern __shared__ float shared_mem[];
    float *ppr_shared=shared_mem;// 500*4 = 2 000
    float *results_shared=&shared_mem[blockDim.x];//3 500 000 * 4 = 14 000 000 max 48kb too big

    //copy ppr chunk in shared mem
    //assert(ppr[idx]!=0);//NON LO PASSA?!?!? forse perchè float approx
    ppr_shared[threadIdx.x]=ppr[idx];

    __syncthreads();
    
    //dynamic allocation but can be done previously with cudaMalloc size: 2*sizeof(float)*beginning_of_warp_data[last]*WarpSize and addressed with &vector[beginning_of_warp_data[warp_num]]
    partial_results = (float*)malloc(2*sizeof(float)*iter_this_warp);
    partial_results_row = (int*)&partial_results[iter_this_warp];
    
    //these loops have same lenght for all warp

    //initialize partial_results after __syncthreads because it' longer for warps with more data to process
    for (int i = 0; i<iter_this_warp;i++){
        partial_results[i]=0;
    }
    //compute
    previous_row = rows[start]+1;
    for (int i = start; i < end; i+=WARP_SIZE) {
        result_index += (rows[i]!=previous_row);//avoid if
        partial_results_row[result_index]=rows[i];
        partial_results[result_index] += vals[i] * ppr_shared[cols[i] % blockDim.x];
        previous_row=rows[i];
        
    }

    //__syncthreads();

    //write on shared mem block_size values in order and then copy them on global mem
    
    for (int i=0;i<gridDim.x;i++){    
        int j=0;
        results_shared[threadIdx.x]=0;
        __syncthreads(); 

        while (j<=result_index){
          if(partial_results_row[j] >= i*blockDim.x && partial_results_row[j] < (i+1)*blockDim.x){
            results_shared[ partial_results_row[j] % blockDim.x] = partial_results[j];
          }
          j++;
        }
        __syncthreads();        
        
        //32 atomicadd in 1 shot using coalescing
        //__syncwarp(); //useless I think
        atomicAdd(&results[i*blockDim.x+threadIdx.x], results_shared[threadIdx.x]);
    }

    free(partial_results);
    
}

void PersonalizedPageRank::cpu_calculate_ppr_2(
    int* cols,
    int* rows,
    float* vals,
    float* ppr,
    float* results,
    int* beginning_of_warp_data,
    float dang_fact,
    int pers_ver,
    float alpha,
    int thx,
    int blkx,
    int blkDim)
{
    int idx = thx + blkx * blkDim;
    int warp_num =  static_cast<int>(blkx * blkDim/WARP_SIZE) + static_cast<int>(thx/WARP_SIZE);
    int start = beginning_of_warp_data[warp_num]*WARP_SIZE + thx; //*WARP_SIZE because I can overlow int, if it happen we'll change easily here in long
    int end = beginning_of_warp_data[warp_num+1]*WARP_SIZE;
    int result_index=-1,previous_row,iter_this_warp=beginning_of_warp_data[warp_num+1]-beginning_of_warp_data[warp_num];
    float *partial_results;//a thread can change a line every iteration, this is max allocation. To allocate less space in private mem put in global a vector of num of row processed by each thread
    int *partial_results_row;

    float *ppr_shared=&ppr[blkx * blkDim];
    //float *results_shared=shared_mem+sizeof(float)*blkDim;

    //copy fr chunk in shared mem
    //ppr_shared[thx]=ppr[idx];
    //results_shared[thx]=0;

    //dynamic allocation but can be done previously with cudaMalloc size: 2*sizeof(float)*beginning_of_warp_data[last]*WarpSize and addressed with &vector[beginning_of_warp_data[warp_num]]
    partial_results = (float*)malloc(2*sizeof(float)*iter_this_warp);
    partial_results_row = (int*)&partial_results[iter_this_warp];
    
    //these loops have same lenght for all warp

    //initialize partial_results after __syncthreads because it' longer for warps with more data to process
    for (int i = 0; i<iter_this_warp;i++){
        partial_results[i]=0;
    }
    //compute
    previous_row = rows[start]+1;
    for (int i = start; i < end; i+=WARP_SIZE) {
        result_index += (rows[i]!=previous_row);//avoid if
        partial_results_row[result_index]=rows[i];
        partial_results[result_index] += vals[i] * ppr_shared[cols[i] % blkDim];
        previous_row=rows[i];
        
    }

    std::cout<<"\n Partial results of block "<<blkx<<" thread "<<thx<<":\n";
    for (int i=0;i<=result_index;i++){
        std::cout<< "(r:"<< partial_results_row[i] <<",v:"<< partial_results[i]<<")   ";
    }
    std::cout<<"\n shared mem usage:\n";
    int j=0;
    for (int i=0;i< BLOCKNUM;i++){

        while (j<=result_index && partial_results_row[j] >= i*blkDim && partial_results_row[j] < (i+1)*blkDim ){
            std::cout<< "(block:"<<i<<",sha_r:"<< partial_results_row[j] % blkDim <<",v:"<< partial_results[j]<<")   ";
            j++;
        }
    }
    //__syncthreads();

    //write on shared mem block_size values in order and then copy them on global mem
    /* 
    int j=0;
    for (int i=0;i< BLOCKNUM;i++){    

        results_shared[thx]=0;
        __syncthreads(); 

        while (j<=result_index && partial_results_row[j] > i*blkDim && partial_results_row[j] < (i+1)*blkDim ){
            j++;
            results_shared[ partial_results_row[j] % blkDim] = partial_results[j];
        }
        __syncthreads();        
        
        //32 atomicadd in 1 shot using coalescing
        //__syncwarp(); //useless I think
        atomicAdd(&results[i*blkDim+thx], results_shared[thx]);
    }
 */
    free(partial_results);

}
 
__global__ void gpu_calculate_ppr_3(
    int *cols_idx,
    int* ptr,
    float* val,
    float* p,
    float dang_fact,
    float* result,
    int pers_ver,
    float alpha,
    int V,
    float* diff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start = ptr[idx];
    int end = ptr[idx + 1];

    float prod_fact = 0;
    for (int i = start; i < end; i++) {
        prod_fact += val[i] * p[cols_idx[i]];
    }
    prod_fact *= alpha;

    result[idx] = prod_fact + dang_fact + (!(pers_ver-idx))*(1-alpha);
    diff[idx] = (result[idx] - p[idx]) * (result[idx] - p[idx]);
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
    pr_f.resize(V);
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
        val_f.push_back(1.0 / outdegree[y[i]]);
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

    if(E==0) //pay attention here
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

    convertedX=xPtr;

}


//it returns num of iterations that each block must do
//it needs convertedX (free here), x, y, val, pr and fill the processedX,processedY, processedVal,processedPr
void PersonalizedPageRank::pre_processing_coo_graph(){
    //blocksize<=notEmptyLines blocksize=n*32 SUM_blockNum(blocksize*numIter[i])=len processedVal,processedX,Y blockNum=V/blockSize
    assert(E>0 && V>=block_size);
    assert(block_size%WARP_SIZE==0);
    
    /* processedPr.assign(pr.begin(), pr.end());
    //because p is splitted in block_size segments but no one will access these values
    for (int i=0;i<V%block_size;i++) {
        processedPr.push_back(0);
    } */

    long total_no_op=0;

    num_of_warp_in_block = block_size/WARP_SIZE;
    beginning_of_warp_data.push_back(0);
    
    //
    for(int thread_block_num=0;thread_block_num<BlockNum;thread_block_num++){  
        std::vector<int> thread_start;
        std::vector<int> semiprocessedX;
        std::vector<int> semiprocessedY;
        std::vector<float> semiprocessedVal;
        int dataLen[2][block_size];   
        
        for (int count=0;count<convertedX.size()-1;count++){
            
            if (thread_start.size()<block_size){
                bool not_empty_line_section=false;
                int thread_start_pos=semiprocessedX.size();
                
                for(int i=convertedX[count];i<convertedX[count+1]&&y[i]<block_size*(thread_block_num+1);i++){
                    if (y[i]<block_size*thread_block_num)
                        continue;
                    semiprocessedX.push_back(x[i]);
                    semiprocessedY.push_back(y[i]);
                    semiprocessedVal.push_back(val_f[i]);
                    not_empty_line_section=true;                    
                }
                if (not_empty_line_section)
                    thread_start.push_back(thread_start_pos);
            }else{
                //concat new data to the shortest thread execution
                int insert_position=0,thread_num=0,length_segment_added=0;

                for(int i=convertedX[count];i<convertedX[count+1]&&y[i]<block_size*(thread_block_num+1);i++){                    
                    if (y[i]<block_size*thread_block_num)
                        continue;
                    //first time inside for u have to find where to put this segment
                    if (i==convertedX[count]){                        
                        //find shortest section
                        int j,shortest_exe = 2147483647;
                        for (j=0;j<block_size-1;j++){
                            if (thread_start[j+1]-thread_start[j]<shortest_exe){                                
                                insert_position=thread_start[j+1];                                
                                shortest_exe=thread_start[j+1]-thread_start[j];
                                thread_num=j;
                            }
                        }
                        //check last thread seq:
                        if (semiprocessedX.size()-thread_start[j]<shortest_exe){                                
                            insert_position=semiprocessedX.size();                                
                            shortest_exe=semiprocessedX.size()-thread_start[j];
                            thread_num=j;
                        }
                        assert(insert_position!=0);
                    }

                    semiprocessedX.insert(semiprocessedX.begin() + insert_position,x[i]);
                    semiprocessedY.insert(semiprocessedY.begin() + insert_position,y[i]);
                    semiprocessedVal.insert(semiprocessedVal.begin() + insert_position,val_f[i]);

                    length_segment_added++;

                }

                //change next threads starting
                for (int j=thread_num+1;j<block_size && length_segment_added!=0;j++){
                    thread_start[j]=thread_start[j]+length_segment_added;
                }
            }
            
        }
        //assert(thread_start.size()%WARP_SIZE==0); //if it's impossible to tune block_size comment this assert but you'll lose performance. (some threads in a warp won't work and add a lot of 0s)

        assert(thread_start.size()!=0);//if this is triggered one entire block is useless. Now try to decrease blocksize but for the future merge this data with the next
        bool not_good_params=false;
        while (thread_start.size()!=block_size){
            thread_start.push_back(semiprocessedX.size());
            not_good_params = true;
        }

        if (not_good_params)
            std::cout << "\nATTENZIONE alcuni warp fanno 0 istruzioni";

        //sort segments to waste less memory with 0s
        
        //create dataLen matrix [[threadNumOfBlock][lenOfDataToProcess]]
        int j;
        for (j=0;j<block_size-1;j++){
            dataLen[0][j]=j;
            dataLen[1][j]=thread_start[j+1]-thread_start[j];
        }
        dataLen[0][j]=j;
        dataLen[1][j]=semiprocessedX.size()-thread_start[j];

        //sort dataLen
        bool sorted=false;
        while(!sorted){
            sorted=true;
            for (int a=1;a<block_size;a++){
                int tempL,tempT;
                if (dataLen[1][a-1]<dataLen[1][a]){
                    sorted=false;
                    tempL=dataLen[1][a];
                    tempT=dataLen[0][a];
                    dataLen[1][a]=dataLen[1][a-1];
                    dataLen[0][a]=dataLen[0][a-1];
                    dataLen[1][a-1]=tempL;
                    dataLen[0][a-1]=tempT;
                }
            }
        }

        //append to processed variables in better order
        int no_op_in_this_block=0;

        for (int warp_num_in_block=0;warp_num_in_block < num_of_warp_in_block; warp_num_in_block++){

            
            beginning_of_warp_data.push_back(beginning_of_warp_data[beginning_of_warp_data.size()-1]+dataLen[1][warp_num_in_block*WARP_SIZE]);
            
            
            for (int iteration=0;iteration<dataLen[1][warp_num_in_block*WARP_SIZE];iteration++){
                for (int thread_of_block=warp_num_in_block*WARP_SIZE;thread_of_block<(warp_num_in_block+1)*WARP_SIZE;thread_of_block++){
                    
                    if (iteration < dataLen[1][thread_of_block]){
                        processedX.push_back(semiprocessedX[thread_start[dataLen[0][thread_of_block]]+iteration]);
                        processedY.push_back(semiprocessedY[thread_start[dataLen[0][thread_of_block]]+iteration]);
                        processedVal.push_back(semiprocessedVal[thread_start[dataLen[0][thread_of_block]]+iteration]);
                    }else{
                        //copy last x and y with value=0
                        processedX.push_back(semiprocessedX[thread_start[dataLen[0][thread_of_block]]+dataLen[1][thread_of_block]-1]);
                        processedY.push_back(semiprocessedY[thread_start[dataLen[0][thread_of_block]]+dataLen[1][thread_of_block]-1]);
                        processedVal.push_back(0);
                        no_op_in_this_block++;
                    }
                }
            }
        }

        total_no_op+=no_op_in_this_block;
        
        std::cout<<"\nno op in block "<<thread_block_num<<": "<<no_op_in_this_block<<" that is avg: "<< no_op_in_this_block/block_size;

        thread_start.clear();
        semiprocessedX.clear();
        semiprocessedY.clear();
        semiprocessedVal.clear();
    }
    
    //Print avg no operations in blocks
    std::cout<< "\n\navg no operations in blocks: " << total_no_op/BlockNum << ". Each thread: " << total_no_op/(block_size * BlockNum)<<"\n\n";

    convertedX.clear();

}


float PersonalizedPageRank::euclidean_distance_float(float *x, float *y, const int N) {
    float result = 0;
    for (int i = 0; i < N; i++) {
        float tmp = x[i] - y[i];
        result += tmp * tmp;
    }
    return std::sqrt(result);
}

//////////////////////////////
//////////////////////////////


void PersonalizedPageRank::alloc_to_gpu_0() {

    cudaMalloc(&d_x, sizeof(double) * convertedX.size());
    cudaMalloc(&d_y, sizeof(double) * y.size());
    cudaMalloc(&d_val, sizeof(double) * val.size());
    //cudaMalloc(&d_dangling, sizeof(int) * dangling.size());
    cudaMalloc(&d_pr, sizeof(double) * V);
    cudaMalloc(&d_newPr, sizeof(double) * V);

    cudaMemcpy(d_x, &convertedX[0], sizeof(double) * convertedX.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y[0], sizeof(double) *  y.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, &val[0], sizeof(double) *  val.size(), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_dangling, &dangling[0], sizeof(int) * dangling.size(), cudaMemcpyHostToDevice);

}

void PersonalizedPageRank::alloc_to_gpu_1() {

    cudaMalloc(&d_x, sizeof(int) * convertedX.size());
    cudaMalloc(&d_y, sizeof(int) * y.size());
    cudaMalloc(&d_val_f, sizeof(float) * val_f.size());
    //cudaMalloc(&d_dangling, sizeof(int) * dangling.size());
    cudaMalloc(&d_pr_f, sizeof(float) * V);
    cudaMalloc(&d_newPr_f, sizeof(float) * V);
    cudaMalloc(&d_diff_f, sizeof(float) * V);
    cudaMalloc(&d_err_sum, sizeof(float) * BlockNum);

    cudaMemcpy(d_x, &convertedX[0], sizeof(int) * convertedX.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y[0], sizeof(int) *  y.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_f, &val_f[0], sizeof(float) *  val_f.size(), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_dangling, &dangling[0], sizeof(int) * dangling.size(), cudaMemcpyHostToDevice);

}

void PersonalizedPageRank::alloc_to_gpu_2() {

    //malloc vectors first to have x32 addresses
    cudaMalloc(&d_x, sizeof(int) * processedX.size());
    cudaMalloc(&d_y, sizeof(int) * processedY.size());
    cudaMalloc(&d_val_f, sizeof(float) * processedVal.size());
    cudaMalloc(&d_pr_f, sizeof(float) * block_size*BlockNum); //alloc more to avoid if in accessing memory in kernel usid thread.x
    cudaMalloc(&d_newPr_f, sizeof(float) * block_size*BlockNum); //for this reason TRY TO HAVE V=block_size*BlockNum
    
    cudaMalloc(&d_beginning_of_warp_data, sizeof(int) * beginning_of_warp_data.size()); //end_of_warp_data.size() is not a x32 so leave as last vector
    cudaMalloc(&d_err_sum, sizeof(float) * BlockNum);

    cudaMemcpy(d_x, &processedX[0], sizeof(int) * x.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &processedY[0], sizeof(int) *  y.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_f, &processedVal[0], sizeof(float) *  val_f.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beginning_of_warp_data, &beginning_of_warp_data[0], sizeof(int) *  beginning_of_warp_data.size(), cudaMemcpyHostToDevice);    
    cudaMemset(d_pr_f, 0.0, sizeof(float) * block_size*BlockNum);
    cudaMemset(d_newPr_f, 0.0, sizeof(float) * block_size*BlockNum);
}

void PersonalizedPageRank::alloc_to_gpu_3() {

    cudaMalloc(&d_x, sizeof(int) * convertedX.size());
    cudaMalloc(&d_y, sizeof(int) * y.size());
    cudaMalloc(&d_val_f, sizeof(float) * val_f.size());
    cudaMalloc(&d_dangling, sizeof(int) * dangling.size());
    cudaMalloc(&d_dang_res, sizeof(float) * dangling.size());
    cudaMalloc(&d_pr_f, sizeof(float) * V);
    cudaMalloc(&d_newPr_f, sizeof(float) * V);
    cudaMalloc(&d_diff_f, sizeof(float) * V);
    cudaMalloc(&d_err_sum, sizeof(float) * BlockNum);

    cudaMemcpy(d_x, &convertedX[0], sizeof(int) * convertedX.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y[0], sizeof(int) *  y.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_f, &val_f[0], sizeof(float) *  val_f.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dangling, &dangling[0], sizeof(int) * dangling.size(), cudaMemcpyHostToDevice);

}

// Allocate data on the CPU and GPU;
void PersonalizedPageRank::alloc() {
    // Load the input graph and preprocess it;
    initialize_graph();

    //convert COO in CSR
    converter();

    // Compute the number of blocks for implementations where the value is a function of the input size;    
    BlockNum = (V - V%block_size) / block_size;
    if (V%block_size!=0)
        BlockNum ++;

    switch (implementation)
    {
    case 0:
        alloc_to_gpu_0();
        break;
    case 1:
        alloc_to_gpu_1();
        break;
    case 2:
        pre_processing_coo_graph();
        alloc_to_gpu_2();
        break;
    case 3:
        alloc_to_gpu_3();
    default:
        break;
    }
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
    pr.clear();
    if (implementation<1){
        // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);        
        newPr.clear();
        for (int i=0; i<V;i++){
            pr.push_back(1.0 / V);
            newPr.push_back(1.0 / V);
        }
        // Reset the result in GPU and Transfer data to the GPU (cudaMemset(d_pr, 1.0 / V, sizeof(double) * V));
        //cudaMemcpy(d_pr, &pr[0], sizeof(double) * V, cudaMemcpyHostToDevice);
        cudaMemset(d_pr, 1.0 / V, V*sizeof(double));
    }else{
        pr_f.clear();
        newPr_f.clear();
        for (int i=0; i<V;i++){
            pr_f.push_back(1.0 / V);
            newPr_f.push_back(1.0 / V);
        }
        // Reset the result in GPU and Transfer data to the GPU (cudaMemset(d_pr, 1.0 / V, sizeof(double) * V));
        cudaMemset(d_pr_f, (float)1/ (float)V, V*sizeof(float));
    }

    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V;
    if (debug) std::cout << "personalization vertex=" << personalization_vertex << std::endl;

}

void PersonalizedPageRank::personalized_page_rank_0(int iter){
    bool converged = false;
    double *d_temp;
    int i = 0;

    while (!converged && i < max_iterations) {

        double dang_fact = 0;
        for (int j = 0; j < V; j++){
            dang_fact += dangling[j] * pr[j];
        }
        dang_fact *= alpha / V;

        // Call the GPU computation.
        gpu_calculate_ppr_0<<< BlockNum, block_size>>>(d_y, d_x, d_val, d_pr, dang_fact, d_newPr, personalization_vertex, alpha, V);

        d_temp=d_pr;
        d_pr=d_newPr;
        d_newPr=d_temp;

        cudaMemcpy(&pr[0],d_pr, sizeof(double) * V, cudaMemcpyDeviceToHost);
        cudaMemcpy(&newPr[0],d_newPr, sizeof(double) * V, cudaMemcpyDeviceToHost);

        //ensure entire pr is calculated
        cudaDeviceSynchronize();

        double err = euclidean_distance_cpu(&newPr[0], &pr[0], V);
        converged = err <= convergence_threshold;
        i++;
    }
}

void PersonalizedPageRank::personalized_page_rank_1(int iter){
    bool converged = false;
    float *d_temp;
    int i = 0;

    while (!converged && i < max_iterations) {

        float dang_fact = 0;
        for (int j = 0; j < V; j++){
            dang_fact += dangling[j] * pr_f[j];
        }
        dang_fact *= alpha / V;

        // Call the GPU computation.
        gpu_calculate_ppr_1<<<BlockNum, block_size>>>(d_y, d_x, d_val_f, d_pr_f, dang_fact, d_newPr_f, personalization_vertex, static_cast<float>(alpha), V, d_diff_f);
        cudaDeviceSynchronize();

        gpu_vector_sum<<<BlockNum, block_size>>>(d_diff_f, d_err_sum, V);
        cudaMemcpy(&err_sum, d_err_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&pr_f[0],d_pr_f, sizeof(float) * V, cudaMemcpyDeviceToHost);//d_pr_f or d_newPr_f?

        d_temp=d_pr_f;
        d_pr_f=d_newPr_f;
        d_newPr_f=d_temp;      

        converged = std::sqrt(err_sum) <= convergence_threshold;
        i++;
    }

    
    //copy results on pr
    for (int j=0;j<V;j++){
        pr.push_back(static_cast<double>(pr_f[j]));
    }
}

void PersonalizedPageRank::personalized_page_rank_2(int iter){
    bool converged = false;
    float *d_temp;
    int i = 0;

    //put all FLOAT variables
    while (!converged && i < max_iterations) {

        double dang_fact = 0;
        for (int j = 0; j < V; j++){
            dang_fact += dangling[j] * pr_f[j];
        }
        dang_fact *= alpha / V;
        

        //set d_newPr full of 0
        cudaMemset(d_newPr_f, 0.0, block_size*BlockNum*sizeof(float));
        cudaDeviceSynchronize();

        // Call the GPU computation.
        gpu_calculate_ppr_2<<< BlockNum, block_size,2*sizeof(float)*block_size>>>(d_y, d_x, d_val_f, d_pr_f, d_newPr_f, d_beginning_of_warp_data);
        /* 
        d_temp=d_pr_f;
        d_pr_f=d_newPr_f;
        d_newPr_f=d_temp;

        cudaMemcpy(&pr_f[0],d_pr_f, sizeof(float) * V, cudaMemcpyDeviceToHost);
        cudaMemcpy(&newPr_f[0],d_newPr_f, sizeof(float) * V, cudaMemcpyDeviceToHost);

        //ensure entire pr is calculated
        cudaDeviceSynchronize();

        float err = euclidean_distance_float(&newPr_f[0], &pr_f[0], V);
        converged = err <= convergence_threshold;
        i++;
         */
  
        cudaDeviceSynchronize();

        gpu_vector_power_sum<<<BlockNum, block_size>>>(d_pr_f, d_err_sum, d_newPr_f,V, static_cast<float>(alpha), dang_fact, personalization_vertex);//it should copy d_newPr_f in d_diff_f after the computation
        cudaMemcpy(&err_sum, d_err_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&pr_f[0],d_newPr_f, sizeof(float) * V, cudaMemcpyDeviceToHost);
/* 
        std::cout<<"Dang: "<<dang_fact<<" 1-aplha:"<<1-alpha<<"\n";
        for (int j=0;j<V;j++){
            std::cout<<pr_f[j]<<"   ";
        }
        std::cout<<"\n";
 */

        d_temp=d_pr_f;
        d_pr_f=d_newPr_f;
        d_newPr_f=d_temp;

        converged = std::sqrt(err_sum) <= convergence_threshold;
        i++;
         
    }

    
    //copy results on pr
    for (int j=0;j<V;j++){
        pr.push_back(static_cast<double>(pr_f[j]));
    }

}

void PersonalizedPageRank::personalized_page_rank_3(int iter){
    bool converged = false;
    float *d_temp;
    int i = 0;

    while (!converged && i < max_iterations) {

        float dang_fact = 0;
        gpu_vector_prod<<<BlockNum, block_size>>>(d_dangling, d_pr_f, d_dang_res, V);
        cudaMemcpy(&dang_fact, d_dang_res, sizeof(float), cudaMemcpyDeviceToHost);
        dang_fact *= alpha / V;

        // Call the GPU computation.
        gpu_calculate_ppr_3<<<BlockNum, block_size>>>(d_y, d_x, d_val_f, d_pr_f, dang_fact, d_newPr_f, personalization_vertex, static_cast<float>(alpha), V, d_diff_f);
        cudaDeviceSynchronize();

        gpu_vector_sum<<<BlockNum, block_size>>>(d_diff_f, d_err_sum, V);
        cudaMemcpy(&err_sum, d_err_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&pr_f[0],d_pr_f, sizeof(float) * V, cudaMemcpyDeviceToHost);

        d_temp=d_pr_f;
        d_pr_f=d_newPr_f;
        d_newPr_f=d_temp;

        converged = std::sqrt(err_sum) <= convergence_threshold;
        i++;
    }

    
    //copy results on pr
    for (int j=0;j<V;j++){
        pr.push_back(static_cast<double>(pr_f[j]));
    }
}

// Do the GPU computation here, and also transfer results to the CPU;
void PersonalizedPageRank::execute(int iter) {

    switch (implementation) {
        case 0:
            personalized_page_rank_0(iter);
            break;
        case 1:
            //use float
            personalized_page_rank_1(iter);
            break;
        case 2:
            //use shared mem but it needs coo
            //test_pre_processing();
            personalized_page_rank_2(iter);
            //improve memory-coalescing changing order of processedX,Y,Val and writing results on shared mem (no atomic add) and after block sync on global (only 1 atomic add for each warp using full bandwidth of global mem)
            //adding at the end of a block data (if len(data)%32!=32) 32-len(data)%32 empty slots or find a way to begin blocks data in x32 addresses
            break;
        case 3:
            //euclidean and dangling in gpu
            personalized_page_rank_3(iter);
            break;
        case 4: //euclidean 1 every 2 cycles
        case 5: //kernel to find top 20 pages and store their value in an array. So that you can copy from gpu only those values with their position (40*4Byte)
        //even the casting is done only on these 20 values
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
    std::set<int> top_pr_indices;
    std::set<int> top_pr_golden_indices;
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

}

void PersonalizedPageRank::test_pre_processing(){
    int i;
    std::cout<< "\nconvertedX:";
    for (i=0; i<convertedX.size()&&i<40;i++){
        std::cout<< "  " <<convertedX[i];
    }
    std::cout<< "\nx:";
    for (i=0; i<x.size()&&i<40;i++){
        std::cout<< "  " << x[i];
    }
    std::cout<< "\ny:";
    for (i=0; i<y.size()&&i<40;i++){
        std::cout<< "  " << y[i];
    }
    std::cout<< "\nval:";
    for (i=0; i<val_f.size()&&i<40;i++){
        std::cout<< "  " << val_f[i];
    }

    std::cout<< "\n\nprocessed vectors size: "<<processedX.size();
    std::cout<< "\nprocessedX:";
    for (i=289; i<354&&processedX.size();i++){
        std::cout<< "  " << processedX[i];
        if (i%WARP_SIZE==0)
          std::cout<< "x";
    }
    std::cout<< "\nordered?:";
    int f=1;
    for (i=0; i<processedX.size()-WARP_SIZE;i++){
        if (static_cast<int>(i/32)+1==beginning_of_warp_data[f]){
            if (i%32==31)
              f++;
            continue;
        }
        if (processedX[i]>processedX[i+WARP_SIZE])
            std::cout<< " !"<<i;
    }
    std::cout<< "\nprocessedY:";
    for (i=289; i<354&&processedY.size();i++){
        std::cout<< "  " << processedY[i];
        if (i%WARP_SIZE==0)
          std::cout<< "x";
    }
    std::cout<< "\nprocessedVal:";
    for (i=289; i<354&&processedVal.size();i++){
        std::cout<< "  " << processedVal[i];
        if (i%WARP_SIZE==0)
          std::cout<< "x";
    }

    std::cout<< "\n\nblock_iterations:";
    for (i=0; i<beginning_of_warp_data.size()&&i<40;i++){
        std::cout<< "  " << beginning_of_warp_data[i];
    }

    std::cout<< "\npr:"<<pr_f[21];

    for (i=0;i<4;i++){

        for(int ax=0;ax<4;ax++){

            cpu_calculate_ppr_2(&processedY[0],&processedX[0],&processedVal[0], &pr_f[0],&newPr_f[0],&beginning_of_warp_data[0],
                0.1,1, 0.15,ax,i,block_size);
        }
    }
}
