//******************************************************************************
// system includes
//******************************************************************************

// output
#include <stdio.h>

// option processing
#include <unistd.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
//******************************************************************************
// local includes
//******************************************************************************

#include "scan.hpp"
#include "alloc.hpp"


//******************************************************************************
// macros 
//******************************************************************************

#define DEFAULT_N 100
#define H 10

#define NBLOCKS(n, block_size) ((n + block_size - 1) / block_size)

//******************************************************************************
// global variables 
//******************************************************************************

int debug_dump = 0;
int verification = 0;


//******************************************************************************
// compute a random vector of integers on the host, with about 1/H of the values
// negative.  negative values represent holes.
//******************************************************************************

void init_input(int *array, long n)
{
  for (long i = 0; i < n; i++) {
    int value = rand() % 10000;  
    if (value % H == 0) value = -value;
    array[i] = value;
  }
}


//******************************************************************************
// functions to dump a host array
//******************************************************************************

void dump(const char *what, int *array, long n)
{
  printf("*** %s ***\n", what);
  if (debug_dump == 0) {
    printf("<<output omitted: -d not specified>>\n\n");
    return;
  }
  for (long i = 0; i < n; i++) {
    printf("%5d ", array[i]);
  }
  printf("\n\n");
}


//******************************************************************************
// functions to dump a device array
//******************************************************************************

void dump_device(const char *what, int *d_array, long n)
{
  HOST_ALLOCATE(h_temp, n, int);
  cudaMemcpy(h_temp, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
  dump(what, h_temp, n);
  HOST_FREE(h_temp);
}
 

//******************************************************************************
// verification code
//******************************************************************************

// a comparison function for qsort to sort values into descending order 
int less(const void *left, const void *right)
{
  int *ileft = (int *) left; 
  int *iright = (int *) right; 
  return *iright - *ileft;
}


// sort a sequence of values in descending order, which puts all of the 
// negative values at the end
void sort(int *h_in, long n)
{
  qsort(h_in,  n, sizeof(int), less);
}


// verify that h_out contains the positive values of h_in
void verify(int *h_in, int *h_out, long n, long n_holes)
{
  if (verification == 0) {
    printf("<<verification omitted: -v not specified>>\n\n");
    return;
  }
  long non_holes = n - n_holes;
  int verified = 0;

  // sort h_in in descending order so that the positive values are at the front
  sort(h_in, n);

  // sort h_out in descending order
  sort(h_out, non_holes);

  // if any value of h_out is not equal to the corresponding value in h_in, the
  // hole compaction is wrong
  for (long i = 0; i < non_holes; i++) {
    if (h_in[i] != h_out[i]) {
      printf("verification failed: h_out[%d] (%d) != h_in[%d] (%d)\n", i, h_out[i], i, h_in[i]);  
    } else {
      verified++;
    }
  }
  if (verified == non_holes) printf("verification succeeded; %d non holes!\n", non_holes);
  printf("\n");
}
     

//******************************************************************************
// data-parallel code
//******************************************************************************

// example cuda function to add 1 to a vector
__global__ void
add_one(int *d_out, int *d_in, long n)
{
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_out[i] = d_in[i] + 1;
}


// example cuda function to set elements of a vector to 1
__global__ void
set_one(int *d_inout, long n)
{
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_inout[i] = 1;
}


//identify holes
__global__ void identify(int *input, int *output, long n){
  long idx = blockIdx.x * blockDim.x + threadIdx.x;  
  if (idx < n) {  
    if (input[idx] < 0) {
      output[idx] = 1;  
    } else {
      output[idx] = 0;  
    }
  }
}


// compact non-Hole Values
__global__ void generateback(int *input, int *output, long compact, long n_holes, int *scan){
  long idx = blockIdx.x * blockDim.x + threadIdx.x;  
  if (idx < n_holes) {  
    int val = input[compact + idx];  
    if (val >= 0) {  
      long newIdx = idx - scan[compact + idx] + scan[compact];  
      output[newIdx] = val;  
    }
  }
}


//fill holes
__global__ void parallel_fill_holes(int *input, int *output, long n, int *nums, int *hole, int *scan){
  long idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    if (hole[idx]) {
      output[idx] = nums[scan[idx]];
    } else {
      output[idx] = input[idx];
    }
  }
}

//******************************************************************************
// a serial hole filling algorithm
//******************************************************************************

long
serial_fill_holes(int *output, int *input, long n)
{
  long n_holes = 0;
  
  long right = n - 1;   // right cursor in the input vector
  long left = 0;        // left cursor in the input and output vectors
  for (; left <= right; left++) {
    if (input[left] > 0) {
      output[left] = input[left];
    } else {
      n_holes++; // count a hole in the prefix that needs filling
      while (right > left && input[right] < 0) {
        right--; n_holes++; // count a hole in the suffix backfill
      }
      if (right <= left) break; 
      output[left] = input[right--]; // fill a hole at the left cursor
    }
  }

  return n_holes;
}


long count_holes(int *in, long n)
{
  long n_holes = 0;
  for (long i = 0; i < n; i++) {
    if (in[i] < 0) n_holes++;
  }
  return n_holes;
} 

//******************************************************************************
// argument processing
//******************************************************************************

void getargs(int argc, char **argv, long *N)
{
  int opt;

  while ((opt = getopt(argc, argv, "dvn:")) != -1) {
    switch (opt) {
    case 'd':
      debug_dump = 1;
      break;

    case 'v':
      verification = 1;
      break;

    case 'n':
      *N = atol(optarg);
      break;

    default: /* '?' */
      fprintf(stderr, "Usage: %s [-n size] [-d] [-v]\n", argv[0]);
      exit(-1);
    }
  }
}


//******************************************************************************
// main function
//******************************************************************************
// int main(int argc, char **argv)
// {
//   long N = DEFAULT_N;
//   long n_holes;

//   getargs(argc, argv, &N);

//   printf("fill: N = %ld, debug_dump = %d\n", N, debug_dump);

//   HOST_ALLOCATE(h_input, N, int);
//   HOST_ALLOCATE(h_output, N, int);
//   DEVICE_ALLOCATE(d_input, N, int);
//   DEVICE_ALLOCATE(d_output, N, int);
//   DEVICE_ALLOCATE(d_tmp, N, int);

//   init_input(h_input, N);

//   dump("input", h_input, N);

//   printf("count_holes returns n_holes = %ld\n\n", count_holes(h_input, N));

//   n_holes = serial_fill_holes(h_output, h_input, N);

//   printf("serial_fill_holes returns n_holes = %ld\n\n", n_holes);

//   dump("h_output after serial_fill_holes", h_output, N - n_holes);

//   verify(h_input, h_output, N, n_holes);

//   //*********************************************************************
//   // copy the input data to the GPU
//   //*********************************************************************
//   cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

//   //*********************************************************************
//   // some example data parallel operations on data in GPU memory
//   //*********************************************************************

//   // set each element of d_tmp to 1 + the the value of the 
//   // corresponding value of d_input
//   add_one<<<NBLOCKS(N, 1024), 1024>>>(d_tmp, d_input, N);

//   dump_device("d_tmp after add_one", d_tmp, N);

//   // increment d_tmp in place
//   add_one<<<NBLOCKS(N, 1024), 1024>>>(d_tmp, d_tmp, N);

//   dump_device("d_tmp after second add_one", d_tmp, N);

//   // set the values in d_tmp
//   set_one<<<NBLOCKS(N, 1024), 1024>>>(d_tmp, N);

//   dump_device("d_tmp after set_one" , d_tmp, N);

//   ex_sum_scan(d_output, d_tmp, N);

//   dump_device("d_output after exscan of d_tmp" , d_output, N);

//   //*********************************************************************
//   // copy the output data from the GPU
//   //*********************************************************************
//   cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

//   dump("h_output on host", h_output, N);

//   HOST_FREE(h_input);
//   HOST_FREE(h_output);

//   DEVICE_FREE(d_input);
//   DEVICE_FREE(d_output);
//   DEVICE_FREE(d_tmp);
// }

int main(int argc, char **argv)
{
  long N = DEFAULT_N;
  long n_holes;
  getargs(argc, argv, &N);
  printf("fill: N = %ld, debug_dump = %d\n", N, debug_dump);
  // Allocation
  HOST_ALLOCATE(h_input, N, int);
  HOST_ALLOCATE(h_output, N, int);
  DEVICE_ALLOCATE(d_input, N, int);
  DEVICE_ALLOCATE(d_output, N, int);
  DEVICE_ALLOCATE(d_hole, N, int);
  DEVICE_ALLOCATE(d_scan, N, int);
  DEVICE_ALLOCATE(d_nums, N, int);

  init_input(h_input, N);
  cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
  // dump_device("d_input", d_input, N);

  // clock_t cstart = clock();
  // n_holes = serial_fill_holes(h_output, h_input, N);
  // clock_t cend = clock();
  // double elapsed_secs = double(cend - cstart) / CLOCKS_PER_SEC * 1000.0;
  // printf("Serial Time: %.5f ms\n", elapsed_secs);
  //CPU time
  auto cstart = std::chrono::high_resolution_clock::now();
  n_holes = serial_fill_holes(h_output, h_input, N);
  auto cend = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ctime = cend - cstart;
  printf("Serial Time: %.5f ms\n", ctime.count()); 
  //record gpu time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  //GPU steps
  identify<<<NBLOCKS(N, 1024), 1024>>>(d_input, d_hole, N);
  ex_sum_scan(d_scan, d_hole, N);
  //newex_sum_scan(d_scan, d_hole, N);
  cudaMemcpy(&n_holes, d_scan + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
  generateback<<<NBLOCKS(n_holes, 1024), 1024>>>(d_input, d_nums, N - n_holes, n_holes, d_scan);
  parallel_fill_holes<<<NBLOCKS(N - n_holes, 1024), 1024>>>(d_input, d_output, N - n_holes, d_nums, d_hole, d_scan);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float runt = 0;
  cudaEventElapsedTime(&runt, start, stop);
  printf("Parallel Time: %.5f ms\n", runt);

  // dump_device("h_output", h_output, N - n_holes);
  // dump_device("d_output", d_output, N - n_holes);
  //step for verify
  cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
  verify(h_input, h_output, N, n_holes);
  //Free memory
  HOST_FREE(h_input);
  HOST_FREE(h_output);
  DEVICE_FREE(d_input);
  DEVICE_FREE(d_output);
  DEVICE_FREE(d_hole);
  DEVICE_FREE(d_scan);
  DEVICE_FREE(d_nums);
  return 0;
}

