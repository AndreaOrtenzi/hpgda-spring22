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
#include <thread>
#include <mutex>
#include <map>
#include "personalized_pagerank.cuh"

#define WARP_SIZE 32
#define SHARED_DIM 49152//48KB
#define UNROLL_PREPROC 20

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
    int* shared_rows,
    float* vals,
    float* ppr,
    float* results,
    int* beginning_of_block_data,
    int* writing_of_blocks,
    int* last_write_lenght)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int start = writing_of_blocks[beginning_of_block_data[blockIdx.x]]*WARP_SIZE + threadIdx.x; //*WARP_SIZE because I can overlow int, if it happen we'll change easily here in long
    int writing_moment = writing_of_blocks[beginning_of_block_data[blockIdx.x]+1]*WARP_SIZE,writing_index=2;
    const int end = writing_of_blocks[beginning_of_block_data[blockIdx.x+1]]*WARP_SIZE;
    

    extern __shared__ float shared_mem[];
    float *ppr_shared=shared_mem;// 500*4 = 2 000
    float *results_shared=&shared_mem[blockDim.x];//3 500 000 * 4 = 14 000 000 max 48kb too big
    const int max_rows_in_shared=(SHARED_DIM-sizeof(float)*blockDim.x)/(sizeof(float)*2);
    int *results_glob_mem_rows=(int*) &results_shared[max_rows_in_shared];

    //copy ppr chunk in shared mem
    ppr_shared[threadIdx.x]=ppr[idx];

    for(int j=0+threadIdx.x;j<max_rows_in_shared;j+=blockDim.x){
        results_shared[j]=0;
        results_glob_mem_rows[j]=j;
    }

    __syncthreads();    

    //compute
    //previous_row = rows[start]+1;
    for (int i = start; i < end; i+=blockDim.x) {
        if (i>=writing_moment){
            __syncthreads();
            writing_moment=writing_of_blocks[beginning_of_block_data[blockIdx.x]+writing_index]*WARP_SIZE;
            writing_index++;
            //shared mem full, copy on global mem and clear it
            for(int j=0+threadIdx.x;j<max_rows_in_shared;j+=blockDim.x){
                atomicAdd(&results[results_glob_mem_rows[j]], results_shared[j]);
                results_shared[j]=0;
                results_glob_mem_rows[j]=j;
            }            
            
            __syncthreads();//wait for it to be empty before continuing
        }
        results_glob_mem_rows[shared_rows[i]]=rows[i];
        atomicAdd(&results_shared[shared_rows[i]], vals[i] * ppr_shared[cols[i] % blockDim.x]);        
    }

    //write for the last time
     __syncthreads();
    
    for(int j=0+threadIdx.x;j<last_write_lenght[blockIdx.x];j+=blockDim.x){
        atomicAdd(&results[results_glob_mem_rows[j]], results_shared[j]);
    }
    
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

    convertedX=xPtr;
}


//it returns num of iterations that each block must do
//it needs x, y, val_f and fill the processedX,processedY, processedVal,processedPr
void PersonalizedPageRank::pre_processing_coo_graph(){
    //blocksize<=notEmptyLines blocksize=n*32 SUM_blockNum(blocksize*numIter[i])=len processedVal,processedX,Y blockNum=V/blockSize
    assert(E>0 && V>=block_size);
    assert(block_size%WARP_SIZE==0);
    assert(sizeof(int)==sizeof(float));
    assert(UNROLL_PREPROC<=BlockNum);
    
    /* processedPr.assign(pr.begin(), pr.end());
    //because p is splitted in block_size segments but no one will access these values
    for (int i=0;i<V%block_size;i++) {
        processedPr.push_back(0);
    } */
    convertedX.clear(); //don't need in this implementation
    long no_op=0;

    remaining_places_in_shared_mem=(SHARED_DIM-sizeof(float)*block_size)/(sizeof(float)*2); //save in shared row and result

    beginning_of_blocks.resize(BlockNum+1,0);

    //read entire col and count num of elements in each column section
    for (int i=0;i<y.size();i++){
        beginning_of_blocks[static_cast<int>(y[i]/block_size)+1]++;
    }

    //round num of elements in each column section as nx32 and save only n and accumulate
    int total_32blocks=0;
    for (int i=1;i<BlockNum+1;i++){
        if (beginning_of_blocks[i]%WARP_SIZE!=0){
            no_op+=WARP_SIZE-beginning_of_blocks[i]%WARP_SIZE;
            total_32blocks+=static_cast<int>(beginning_of_blocks[i]/WARP_SIZE)+1;
            beginning_of_blocks[i]=static_cast<int>(beginning_of_blocks[i]/WARP_SIZE)+1+beginning_of_blocks[i-1];
            
        }else{
            total_32blocks+=static_cast<int>(beginning_of_blocks[i]/WARP_SIZE);
            beginning_of_blocks[i]=static_cast<int>(beginning_of_blocks[i]/WARP_SIZE)+beginning_of_blocks[i-1];            
        }
    }   
  
    processedX.assign(total_32blocks*WARP_SIZE,0);
    processedXShared.assign(total_32blocks*WARP_SIZE,0);
    processedY.assign(total_32blocks*WARP_SIZE,0);
    processedVal.assign(total_32blocks*WARP_SIZE,0);
    last_write_length.resize(BlockNum,0);
    //writings_of_blocks_list.push_back(0);
    writings_of_blocks_list.assign(beginning_of_blocks.begin(),beginning_of_blocks.end());

    ///////////////////////////////////
    
    //std::vector<int> num_of_warp_data_block(total_32blocks,0);
    std::vector<int> last_row(BlockNum,-1);
    std::vector<int> first_free_warp(BlockNum,0);
    
    
    std::vector<std::vector<int>> data_in_warp_slot(BlockNum);
    std::list<int> index_to_replace[BlockNum];
    int warp_ex[BlockNum];
    
    for(int i=0;i<BlockNum;i++){
        index_to_replace[i]= std::list<int>();
        warp_ex[i]=(beginning_of_blocks[i+1]-beginning_of_blocks[i])>1;
        
        std::vector<int> temp(beginning_of_blocks[i+1]-beginning_of_blocks[i],0);
        data_in_warp_slot[i].assign(temp.begin(),temp.end()); 
        temp.clear();

        
    }
    
    for (int i=0;i<E;i++){
        int block_number=y[i]/block_size;        
        
        if (x[i]==last_row[block_number]){
            if (beginning_of_blocks[block_number]*WARP_SIZE + warp_ex[block_number]*WARP_SIZE>=beginning_of_blocks[block_number+1]*WARP_SIZE){
                index_to_replace[block_number].push_back(i);
                continue;
            }
            processedX[beginning_of_blocks[block_number]*WARP_SIZE+warp_ex[block_number]*WARP_SIZE+data_in_warp_slot[block_number][warp_ex[block_number]]]=x[i];
            processedY[beginning_of_blocks[block_number]*WARP_SIZE+warp_ex[block_number]*WARP_SIZE+data_in_warp_slot[block_number][warp_ex[block_number]]]=y[i];
            processedVal[beginning_of_blocks[block_number]*WARP_SIZE+warp_ex[block_number]*WARP_SIZE+data_in_warp_slot[block_number][warp_ex[block_number]]]=val_f[i];
            
            data_in_warp_slot[block_number][warp_ex[block_number]]++;
            if (data_in_warp_slot[block_number][warp_ex[block_number]]==WARP_SIZE)
                first_free_warp[block_number]++; 
            warp_ex[block_number]++;
            continue;
        }
        last_row[block_number]=x[i];
        processedX[beginning_of_blocks[block_number]*WARP_SIZE+first_free_warp[block_number]*WARP_SIZE+data_in_warp_slot[block_number][first_free_warp[block_number]]]=x[i];
        processedY[beginning_of_blocks[block_number]*WARP_SIZE+first_free_warp[block_number]*WARP_SIZE+data_in_warp_slot[block_number][first_free_warp[block_number]]]=y[i];
        processedVal[beginning_of_blocks[block_number]*WARP_SIZE+first_free_warp[block_number]*WARP_SIZE+data_in_warp_slot[block_number][first_free_warp[block_number]]]=val_f[i];

        warp_ex[block_number]=first_free_warp[block_number]+1;

        data_in_warp_slot[block_number][first_free_warp[block_number]]++;
        if (data_in_warp_slot[block_number][first_free_warp[block_number]]==WARP_SIZE)
            first_free_warp[block_number]++;        
    }

    //place index with problems: put first index (the one with lower row in the last slot hoping that is not serialized in shared)
    std::vector<bool> decreasing(BlockNum,true);
    int new_pos[BlockNum],future_first_free_warp[BlockNum];

    for (int block_number=0;block_number<BlockNum;block_number++){
        new_pos[block_number]=beginning_of_blocks[block_number+1]-beginning_of_blocks[block_number]-1;
        future_first_free_warp[block_number]=first_free_warp[block_number];

        

        for (std::list<int>::iterator i=index_to_replace[block_number].begin();i!=index_to_replace[block_number].end();i++){
            //bounce between first_free_warp<=a<num_of_warp_data_block (incremental free slots or equal)
            //assert(start+new_pos*WARP_SIZE+data_in_warp_slot[new_pos]<end);
            
            processedX[beginning_of_blocks[block_number]*WARP_SIZE+new_pos[block_number]*WARP_SIZE+data_in_warp_slot[block_number][new_pos[block_number]]]=x[*i];
            processedY[beginning_of_blocks[block_number]*WARP_SIZE+new_pos[block_number]*WARP_SIZE+data_in_warp_slot[block_number][new_pos[block_number]]]=y[*i];
            processedVal[beginning_of_blocks[block_number]*WARP_SIZE+new_pos[block_number]*WARP_SIZE+data_in_warp_slot[block_number][new_pos[block_number]]]=val_f[*i];

            
            data_in_warp_slot[block_number][new_pos[block_number]]++;
            if (data_in_warp_slot[block_number][new_pos[block_number]]==WARP_SIZE&&future_first_free_warp[block_number]<=new_pos[block_number]){
                future_first_free_warp[block_number]=new_pos[block_number]+1;
            }

            if (decreasing[block_number]){
                if (first_free_warp[block_number]>new_pos[block_number]-1){
                    first_free_warp[block_number]=future_first_free_warp[block_number];
                    new_pos[block_number]=future_first_free_warp[block_number];//here +1 could be better but too complicated to track empty slots
                    decreasing[block_number]=false;
                }else{
                    new_pos[block_number]--;
                }
            }else{
                if (beginning_of_blocks[block_number+1]-beginning_of_blocks[block_number]<=new_pos[block_number]+1){
                    //new_pos--;//here could be better but too complicated to track empty slots
                    decreasing[block_number]=true;
                }else{
                    new_pos[block_number]++;
                }
            }   

        }
    

        //map each row to a shared mem row
        std::map<int, int> shared_global_row;
        int index_to_use=0;
        for (int i=beginning_of_blocks[block_number]*WARP_SIZE;i<beginning_of_blocks[block_number+1]*WARP_SIZE;i++){
            if (index_to_use>=remaining_places_in_shared_mem){
                //not enough shared mem so come back at warp beginning and restart mapping from 0, before this warp kernel will write on global mem
                i-=i%WARP_SIZE-1;
                index_to_use=0;
                shared_global_row.clear();
                //add to vector index for writes
                
                writings_of_blocks_list.push_back((i+1)/WARP_SIZE);
                continue;
            }

            if (processedVal[i]==0){
                //this is a place holder to get x32 positions per block, so change row and sharedRow to not collide with same warp
                //find an intelligent value: not same as others in the same warp, close to them but not more than remaining_places_in_shared_mem
                //but it can use a shared mem slot, can't find a solution quicly so keep it for now
                int val=0,j;
                for (j=i-i%WARP_SIZE;j<i+(WARP_SIZE-i%WARP_SIZE)-1;j++){
                    if (processedX[j+1]-processedX[j]>1&&processedX[j]!=0){
                        val=processedX[j]+1;
                        break;
                    }
                }
                if (val==0){
                    val=processedX[j]+1;
                }
                processedX[i]=val%(block_size*BlockNum);
                
            }

            if (shared_global_row.find(processedX[i])==shared_global_row.end()){
                shared_global_row[processedX[i]]=index_to_use;
                index_to_use++;
            }

            processedXShared[i]=shared_global_row[processedX[i]];
            
        }

        last_write_length[block_number]=index_to_use;
    }

    //sort writings because with threads they are random O(NlogN)
    writings_of_blocks_list.sort();//should be already ordered

    writings_of_blocks.assign(writings_of_blocks_list.begin(),writings_of_blocks_list.end());
    writings_of_blocks_list.clear();
    

    //change beginning_of_blocks to not be redundant so it contains the index of the first and the last value in writings_of_blocks
    int beginning_of_blocks_index=0;
    for (int i=0;i<writings_of_blocks.size();i++){
        if(beginning_of_blocks[beginning_of_blocks_index]==writings_of_blocks[i]){
            beginning_of_blocks[beginning_of_blocks_index]=i;
            beginning_of_blocks_index++;
        }
    }

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
    cudaMalloc(&d_pr, sizeof(double) * V);
    cudaMalloc(&d_newPr, sizeof(double) * V);

    cudaMemcpy(d_x, &convertedX[0], sizeof(double) * convertedX.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y[0], sizeof(double) *  y.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, &val[0], sizeof(double) *  val.size(), cudaMemcpyHostToDevice);
}

void PersonalizedPageRank::alloc_to_gpu_1() {

    cudaMalloc(&d_x, sizeof(int) * convertedX.size());
    cudaMalloc(&d_y, sizeof(int) * y.size());
    cudaMalloc(&d_val_f, sizeof(float) * val_f.size());
    cudaMalloc(&d_pr_f, sizeof(float) * V);
    cudaMalloc(&d_newPr_f, sizeof(float) * V);
    cudaMalloc(&d_diff_f, sizeof(float) * V);
    cudaMalloc(&d_err_sum, sizeof(float) * BlockNum);

    cudaMemcpy(d_x, &convertedX[0], sizeof(int) * convertedX.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y[0], sizeof(int) *  y.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_f, &val_f[0], sizeof(float) *  val_f.size(), cudaMemcpyHostToDevice);
}

void PersonalizedPageRank::alloc_to_gpu_2() {

    //malloc vectors first to have x32 addresses
    cudaMalloc(&d_x, sizeof(int) * processedX.size());
    cudaMalloc(&d_x_shared, sizeof(int) * processedXShared.size());
    cudaMalloc(&d_y, sizeof(int) * processedY.size());
    cudaMalloc(&d_val_f, sizeof(float) * processedVal.size());
    cudaMalloc(&d_pr_f, sizeof(float) * block_size*BlockNum); //alloc more to avoid if in accessing memory in kernel usid thread.x
    cudaMalloc(&d_newPr_f, sizeof(float) * block_size*BlockNum); //for this reason TRY TO HAVE V=block_size*BlockNum
    
    cudaMalloc(&d_writings_of_blocks, sizeof(int) * writings_of_blocks.size());
    cudaMalloc(&d_beginning_of_blocks, sizeof(int) * beginning_of_blocks.size()); //end_of_warp_data.size() is not a x32 so leave as last vector
    cudaMalloc(&d_last_write_length, sizeof(int) * last_write_length.size());
    cudaMalloc(&d_err_sum, sizeof(float) * BlockNum);

    cudaMemcpy(d_x, &processedX[0], sizeof(int) * processedX.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_shared, &processedXShared[0], sizeof(int) * processedXShared.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &processedY[0], sizeof(int) *  processedY.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_f, &processedVal[0], sizeof(float) *  processedVal.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beginning_of_blocks, &beginning_of_blocks[0], sizeof(int) *  beginning_of_blocks.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_writings_of_blocks, &writings_of_blocks[0], sizeof(int) *  writings_of_blocks.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_last_write_length, &last_write_length[0], sizeof(int) *  last_write_length.size(), cudaMemcpyHostToDevice);
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

void PersonalizedPageRank::alloc_to_gpu_4() {

    //malloc vectors first to have x32 addresses
    cudaMalloc(&d_x, sizeof(int) * processedX.size());
    cudaMalloc(&d_x_shared, sizeof(int) * processedXShared.size());
    cudaMalloc(&d_y, sizeof(int) * processedY.size());
    cudaMalloc(&d_val_f, sizeof(float) * processedVal.size());
    cudaMalloc(&d_pr_f, sizeof(float) * block_size*BlockNum); //alloc more to avoid if in accessing memory in kernel usid thread.x
    cudaMalloc(&d_newPr_f, sizeof(float) * block_size*BlockNum); //for this reason TRY TO HAVE V=block_size*BlockNum
    
    cudaMalloc(&d_writings_of_blocks, sizeof(int) * writings_of_blocks.size());
    cudaMalloc(&d_beginning_of_blocks, sizeof(int) * beginning_of_blocks.size()); //end_of_warp_data.size() is not a x32 so leave as last vector
    cudaMalloc(&d_last_write_length, sizeof(int) * last_write_length.size());
    cudaMalloc(&d_dangling, sizeof(int) * dangling.size());
    cudaMalloc(&d_err_sum, sizeof(float));
    cudaMalloc(&d_dang_res, sizeof(float));

    cudaMemcpy(d_x, &processedX[0], sizeof(int) * processedX.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_shared, &processedXShared[0], sizeof(int) * processedXShared.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &processedY[0], sizeof(int) *  processedY.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_f, &processedVal[0], sizeof(float) *  processedVal.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beginning_of_blocks, &beginning_of_blocks[0], sizeof(int) *  beginning_of_blocks.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_writings_of_blocks, &writings_of_blocks[0], sizeof(int) *  writings_of_blocks.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_last_write_length, &last_write_length[0], sizeof(int) *  last_write_length.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dangling, &dangling[0], sizeof(int) * dangling.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_pr_f, 0.0, sizeof(float) * block_size*BlockNum);
    cudaMemset(d_newPr_f, 0.0, sizeof(float) * block_size*BlockNum);    
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
        break;
    case 4:
        pre_processing_coo_graph();
        alloc_to_gpu_4();
    default:
        break;
    }
}

// Initialize data;
void PersonalizedPageRank::init() {
    // Do any additional CPU or GPU setup here;
    switch (implementation)
    {
    case 2:
    case 4:
        pre_processing_coo_graph();
        break;
    default:
        break;
    }
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
        cudaMemset(d_err_sum, 0.0, sizeof(float));
        cudaDeviceSynchronize();

        // Call the GPU computation.
        gpu_calculate_ppr_2<<< BlockNum, block_size,SHARED_DIM>>>(d_y, d_x,d_x_shared, d_val_f, d_pr_f, d_newPr_f, d_beginning_of_blocks,d_writings_of_blocks,d_last_write_length);
          
        cudaDeviceSynchronize();

        gpu_vector_power_sum<<<BlockNum, block_size>>>(d_pr_f, d_err_sum, d_newPr_f,V, static_cast<float>(alpha), dang_fact, personalization_vertex);//it should copy d_newPr_f in d_diff_f after the computation
        cudaMemcpy(&err_sum, d_err_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&pr_f[0],d_newPr_f, sizeof(float) * V, cudaMemcpyDeviceToHost);

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

        cudaMemset(d_newPr_f, 0.0, V*sizeof(float));
        cudaMemset(d_err_sum, 0.0, sizeof(float));
        cudaMemset(d_dang_res, 0.0, sizeof(float));

        float dang_fact = 0;
        gpu_vector_prod<<<BlockNum, block_size>>>(d_dangling, d_newPr_f, d_dang_res, V);
        cudaMemcpy(&dang_fact, d_dang_res, sizeof(float), cudaMemcpyDeviceToHost);
        dang_fact *= alpha / V;

        // Call the GPU computation.
        gpu_calculate_ppr_3<<<BlockNum, block_size>>>(d_y, d_x, d_val_f, d_pr_f, dang_fact, d_newPr_f, personalization_vertex, static_cast<float>(alpha), V, d_diff_f);
        cudaDeviceSynchronize();

        gpu_vector_sum<<<BlockNum, block_size>>>(d_diff_f, d_err_sum, V);
        cudaMemcpy(&err_sum, d_err_sum, sizeof(float), cudaMemcpyDeviceToHost);
        

        d_temp=d_pr_f;
        d_pr_f=d_newPr_f;
        d_newPr_f=d_temp;

        converged = std::sqrt(err_sum) <= convergence_threshold;
        i++;
    }

    cudaMemcpy(&pr_f[0],d_newPr_f, sizeof(float) * V, cudaMemcpyDeviceToHost);
    //copy results on pr
    for (int j=0;j<V;j++){
        pr.push_back(static_cast<double>(pr_f[j]));
    }
}

void PersonalizedPageRank::personalized_page_rank_4(int iter){
    bool converged = false;
    float *d_temp, dang_fact = 0;
    int i = 0;

    while (!converged && i < max_iterations) {

        cudaMemset(d_newPr_f, 0.0, V*sizeof(float));
        cudaMemset(d_err_sum, 0.0, sizeof(float));
        cudaMemset(d_dang_res, 0.0, sizeof(float));
        cudaDeviceSynchronize();
        
        gpu_vector_prod<<<BlockNum, block_size>>>(d_dangling, d_pr_f, d_dang_res, V);

        // Call the GPU computation.
        gpu_calculate_ppr_2<<< BlockNum, block_size,SHARED_DIM>>>(d_y, d_x,d_x_shared, d_val_f, d_pr_f, d_newPr_f, d_beginning_of_blocks,d_writings_of_blocks,d_last_write_length);

        cudaDeviceSynchronize();
        cudaMemcpy(&dang_fact, d_dang_res, sizeof(float), cudaMemcpyDeviceToHost);
        dang_fact *= alpha / V;//put this in gpu vector power_sum so don't need to memcpy, is faster?

        gpu_vector_power_sum<<<BlockNum, block_size>>>(d_pr_f, d_err_sum, d_newPr_f,V, static_cast<float>(alpha), dang_fact, personalization_vertex);
        cudaMemcpy(&err_sum, d_err_sum, sizeof(float), cudaMemcpyDeviceToHost);
        

        d_temp=d_pr_f;
        d_pr_f=d_newPr_f;
        d_newPr_f=d_temp;

        converged = std::sqrt(err_sum) <= convergence_threshold;
        i++;
    }

    cudaMemcpy(&pr_f[0],d_pr_f, sizeof(float) * V, cudaMemcpyDeviceToHost);
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
            personalized_page_rank_2(iter);
            //improve memory-coalescing changing order of processedX,Y,Val and writing results on shared mem (no atomic add) and after block sync on global (only 1 atomic add for each warp using full bandwidth of global mem)
            //adding at the end of a block data (if len(data)%32!=32) 32-len(data)%32 empty slots or find a way to begin blocks data in x32 addresses
            break;
        case 3:
            //euclidean and dangling in gpu
            personalized_page_rank_3(iter);
            break;
        case 4: 
            personalized_page_rank_4(iter);
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

void PersonalizedPageRank::clean_0() {
    free(&x[0]);
    free(&y[0]);  
    free(&val[0]);  
    free(&dangling[0]);
    free(&pr[0]);
    free(&pr_golden[0]);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_val);
    cudaFree(d_pr);
    cudaFree(d_newPr);
}

void PersonalizedPageRank::clean_1() {
    free(&x[0]);
    free(&y[0]);  
    free(&val[0]);  
    free(&dangling[0]);
    free(&pr[0]);
    free(&pr_golden[0]);

    cudaFree(d_val_f);
    cudaFree(d_pr_f);
    cudaFree(d_newPr_f);
    cudaFree(d_diff_f);
    cudaFree(d_err_sum);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_val);
    cudaFree(d_pr);
    cudaFree(d_newPr);
}

void PersonalizedPageRank::clean_2() {
    free(&x[0]);
    free(&y[0]);  
    free(&val[0]);  
    free(&dangling[0]);
    free(&pr[0]);
    free(&pr_golden[0]);

    cudaFree(d_beginning_of_blocks);
    cudaFree(d_x_shared);
    cudaFree(d_writings_of_blocks);
    cudaFree(d_last_write_length);
    cudaFree(d_val_f);
    cudaFree(d_pr_f);
    cudaFree(d_newPr_f);
    cudaFree(d_diff_f);
    cudaFree(d_err_sum);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_val);
    cudaFree(d_pr);
    cudaFree(d_newPr);
}

void PersonalizedPageRank::clean_3() {
    free(&x[0]);
    free(&y[0]);  
    free(&val[0]);  
    free(&dangling[0]);
    free(&pr[0]);
    free(&pr_golden[0]);

    cudaFree(d_beginning_of_blocks);
    cudaFree(d_x_shared);
    cudaFree(d_writings_of_blocks);
    cudaFree(d_last_write_length);
    cudaFree(d_val_f);
    cudaFree(d_pr_f);
    cudaFree(d_newPr_f);
    cudaFree(d_diff_f);
    cudaFree(d_err_sum);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_val);
    cudaFree(d_pr);
    cudaFree(d_newPr);
}

void PersonalizedPageRank::clean_4() {
    free(&x[0]);
    free(&y[0]);  
    free(&val[0]);  
    free(&dangling[0]);
    free(&pr[0]);
    free(&pr_golden[0]);

    cudaFree(d_beginning_of_blocks);
    cudaFree(d_x_shared);
    cudaFree(d_writings_of_blocks);
    cudaFree(d_last_write_length);
    cudaFree(d_val_f);
    cudaFree(d_pr_f);
    cudaFree(d_newPr_f);
    cudaFree(d_diff_f);
    cudaFree(d_err_sum);
    cudaFree(d_dangling);
    cudaFree(d_dang_res);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_val);
    cudaFree(d_pr);
    cudaFree(d_newPr);
}

void PersonalizedPageRank::clean() {
    // Delete any GPU data or additional CPU data;
    switch (implementation) {
        case 0:
            clean_0();
            break;
        case 1:
            clean_1();
            break;
        case 2:
            clean_2();
            break;
        case 3:
            clean_3();
            break;
        case 4: 
            clean_4();
            break;
        default:
            break;
    }
}