//******************************************************************************
// cuda includes
//******************************************************************************

#include <cub/cub.cuh>


//******************************************************************************
// local includes
//******************************************************************************

#include "scan.hpp"


//******************************************************************************
// interface operations
//******************************************************************************

// compute an exclusive sum scan of d_in in O(log_2 n) steps
void ex_sum_scan
(
 int *d_out, // pointer to an output device array with space for n integers 
 int *d_in,  // pointer to an input device array containing n integers
 int n       // n - the number of elements in d_in and d_out
)
{
  void     *d_tmp = NULL; // pointer to temporary storage used by the the scan 
  size_t   tmp_bytes = 0; // the number of bytes of temporary storage needed

  // determine how many bytes of temporary storage are needed
  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n);

  // allocate temporary storage
  cudaMalloc(&d_tmp, tmp_bytes);

  // compute the exclusive prefix sum of d_in into d_out using d_tmp as 
  // temporary storage
  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n);

  // feee the temporary storage
  cudaFree(d_tmp);
}


const int BLOCK_SIZE = 1024; 
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))


__global__ void prescan(int *d_out, int *d_in, long n){
  extern __shared__ int temp[];  
  long thread_id = threadIdx.x;
  long global_id = blockIdx.x * blockDim.x + threadIdx.x;

  //shared memory
  int bankOffset = CONFLICT_FREE_OFFSET(thread_id);
  //temp[thread_id + bankOffset] = (global_id < n) ? d_in[global_id] : 0;
  if (global_id < n) {
    temp[thread_id + bankOffset] = d_in[global_id];
  } else {
    temp[thread_id + bankOffset] = 0;
  }

  temp[thread_id + blockDim.x + bankOffset] = 0;

  __syncthreads();

  // Up-sweep (reduce) phase
  int offset = 1;
  for (int d = blockDim.x >> 1; d > 0; d >>= 1){
    __syncthreads();
    if (thread_id < d){
      int ai = offset * (2 * thread_id + 1) - 1 + CONFLICT_FREE_OFFSET(offset * (2 * thread_id + 1) - 1);
      int bi = offset * (2 * thread_id + 2) - 1 + CONFLICT_FREE_OFFSET(offset * (2 * thread_id + 2) - 1);
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  // Down-sweep phase
  if (thread_id == 0){
    //d_block_sum[blockIdx.x] = temp[blockDim.x - 1 + CONFLICT_FREE_OFFSET(blockDim.x - 1)];
    temp[blockDim.x - 1 + CONFLICT_FREE_OFFSET(blockDim.x - 1)] = 0;
  }

  for (int d = 1; d < blockDim.x; d *= 2){
    offset >>= 1;
    __syncthreads();
    if (thread_id < d){
      int ai = offset * (2 * thread_id + 1) - 1 + CONFLICT_FREE_OFFSET(offset * (2 * thread_id + 1) - 1);
      int bi = offset * (2 * thread_id + 2) - 1 + CONFLICT_FREE_OFFSET(offset * (2 * thread_id + 2) - 1);
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

    __syncthreads();
  if (global_id < n){
    d_out[global_id] = temp[thread_id + bankOffset];
  }
}
// handle error
void handle(cudaError_t status) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
    exit(EXIT_FAILURE);
  }
}


// summation of blocks
__global__ void sumb(int *d_out, const int *block_sums, long n) {
  long global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_id >= n) return;
  if (blockIdx.x > 0) {
    d_out[global_id] += block_sums[blockIdx.x - 1];
  }
}


// new scan method
void newex_sum_scan(int *d_out, int *d_in, int n) {
  int *d_block_sum;
  int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int shared_mem_size = 2 * BLOCK_SIZE * sizeof(int);
  handle(cudaMalloc(&d_block_sum, num_blocks * sizeof(int)));

  // scanning
  prescan<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(d_out, d_in, n);

  // recursive call for more blocks
  if (num_blocks > 1) {
    int *d_scanned_block_sums;
    handle(cudaMalloc(&d_scanned_block_sums, num_blocks * sizeof(int)));
    newex_sum_scan(d_scanned_block_sums, d_block_sum, num_blocks);

    // summation of blocks
    sumb<<<num_blocks, BLOCK_SIZE>>>(d_out, d_scanned_block_sums, n);
    cudaFree(d_scanned_block_sums);
  }

  cudaFree(d_block_sum);
}
