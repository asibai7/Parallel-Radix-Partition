/*  Programmer: Ahmad Sibai
    Project 3: Parallel Radix Partition (CUDA)
    To compile: nvcc PRP.c -o PRP
*/

#include <assert.h>
#include <cuda_runtime.h> //header for cudaMalloc, cudaFree, cudaMemcpy
#include <device_launch_parameters.h> //header for blockIdx, blockDim, threadIdx
// cuda_runtime.h and device_launch_parameters.h are not needed because of nvcc, i'm keeping them for practice
#include <stdio.h>
#include <stdlib.h>
#include <limits.h> // need in order to use INT_MAX to avoid overflow

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))
#define BLOCK_SIZE 1024 // number of threads per block
#define MAX_PARTITIONS 1024 // max number of partitions

// data generator which take in number of elements
void dataGenerator(int* data, int count, int first, int step)
{
    assert(data != NULL);
    for(int i = 0; i < count; ++i)
        data[i] = first + i * step;
    srand(time(NULL));
    for(int i = count-1; i>0; i--) // knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}

/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}

// histogram kernel that computes histogram of partitions
__global__ void histogram(int* input, int* histLow, int* histHigh, int arraySize, int numPartitions, int bitsPerPartition)
{
    // shared mem array to hold low and high histogram counts
    __shared__ int sharedHistLow[MAX_PARTITIONS];
    __shared__ int sharedHistHigh[MAX_PARTITIONS]; 
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // calculates global thread id
    int localTid = threadIdx.x; // calculates thread id within block
    if (localTid < numPartitions) { // sets histograms to 0 only if local thread id is less than number of partitions
        sharedHistLow[localTid] = 0;
        sharedHistHigh[localTid] = 0;
    }
    __syncthreads(); // sync threads before starting field extraction

    // loop through elements asigned to thread
    for (int i = tid; i < arraySize; i += gridDim.x * blockDim.x) {
        int partition = bfe(input[i], 0, bitsPerPartition); // get partition index using bit field extraction 
        if (atomicAdd(&sharedHistLow[partition], 1) == INT_MAX) { // update low histogram
        /* if value before addition reaches the max an int can hold then
        to avoid overflow, increment high histogram and reset low to 0
        */ 
            atomicAdd(&sharedHistHigh[partition], 1);
            sharedHistLow[partition] = 0;
        }
    }
    __syncthreads();
    if (localTid < numPartitions) { // update global histogram with data from shared mem if local thread id is less than number of partitions
        atomicAdd(&histLow[localTid], sharedHistLow[localTid]);
        atomicAdd(&histHigh[localTid], sharedHistHigh[localTid]);
    }
}

// prefix scan kernel that computes exclusive prefix sums of histogram
__global__ void prefixScan(int* histLow, int* histHigh, int* prefixSum, int numPartitions)
{
    __shared__ int temp[2 * BLOCK_SIZE]; // declare shared mem to hold temporary data for scan
    int tid = threadIdx.x; // thread id within block 
    if (tid < numPartitions) // if thread index is less than number of partitions
        temp[tid] = (int)histHigh[tid] * INT_MAX + histLow[tid]; // load combined histogram value into shared mem
    else 
        temp[tid] = 0; // mark 0 for threads outside of range
    __syncthreads(); // sync threads to make sure data is ready before scan

    // inclusive scan using reduction
    for (int stride = 1; stride < numPartitions; stride *= 2)
    { 
        int index = (tid + 1) * 2 * stride - 1; // calculate index for current thread in stride
        if (index < 2 * BLOCK_SIZE) // check if within the boundary 
            temp[index] += temp[index - stride]; // if so add value from previous stride
        __syncthreads(); // sync to check all calculations for stride are done
    }

    // perform distribution to convert values from reduction to prefix sums
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        int index = (tid + 1) * 2 * stride - 1; // calculate index for current thread in stride
        if ((index + stride) < 2 * BLOCK_SIZE) // check if index and next stride are within boundary
            temp[index + stride] += temp[index]; // if so add value from previous stride
    }
    __syncthreads(); // sync to check prefix sum is done
    if (tid < numPartitions) // check if thread id is within range
        prefixSum[tid] = (int)((tid == 0) ? 0 : temp[tid - 1]); // if so then load prefix sum into global mem
}

// reorder kernel reorders keys based on prefix sum
__global__ void Reorder(int* input, int* output, int* prefixSum, int arraySize, int numPartitions, int bitsPerPartition)
{
    __shared__ int localOffsets[MAX_PARTITIONS]; // declare shared mem for local offsets
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // calculate gloabl thread id
    int localTid = threadIdx.x; // get local thread id within block
    if (localTid < numPartitions) // if local thread id is within range
        localOffsets[localTid] = prefixSum[localTid]; // initialize local offset from prefix sum
    __syncthreads(); // sync to check all offsets are initialized

    // loop through elements assigned to thread 
    for (int i = tid; i < arraySize; i += gridDim.x * blockDim.x) {
        int partition = bfe(input[i], 0, bitsPerPartition); // extract partition index using bfe
        int offset = atomicAdd(&localOffsets[partition], 1); // increment offset atomically so partition can get next position
        if (offset < arraySize) // if offset is within array size
            output[offset] = input[i]; // place input key in correct position in output array
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 3) { // check user inputs correct number of arguments
        printf("Usage: %s <array_size> <num_partitions>\n", argv[0]);
        return 1;
    }
    int arraySize = atoi(argv[1]); // get array size from argv and convert to int
    int numPartitions = atoi(argv[2]); // get number of partitions from argv and convert to int

    // validate input parameters checking that number of elements is 1,000,000 or greater, that number of partitions is within 2 to 1024, and that number of partitions is a power of 2
    if (arraySize < 1000000 || numPartitions < 2 || numPartitions > 1024 || (numPartitions & (numPartitions - 1)) != 0) {
        printf("Invalid input parameters\n");
        return 1;
    }
    // calculate number of bits to represent number of partitions
    int bitsPerPartition = 0;
    while ((1 << bitsPerPartition) < numPartitions)
        bitsPerPartition++;
    // declare pointers for host and device arrays (r_h should be named h_input but I'm following the provided template)
    int *r_h, *h_output, *h_histLow, *h_histHigh, *h_prefixSum;
    int *d_input, *d_output, *d_histLow, *d_histHigh, *d_prefixSum;
    // allocate host memory
    cudaMallocHost((void**)&r_h, arraySize * sizeof(int));
    cudaMallocHost((void**)&h_output, arraySize * sizeof(int));
    cudaMallocHost((void**)&h_histLow, numPartitions * sizeof(int));
    cudaMallocHost((void**)&h_histHigh, numPartitions * sizeof(int));
    cudaMallocHost((void**)&h_prefixSum, numPartitions * sizeof(int));
    // generate data based on number of elements inputted
    dataGenerator(r_h, arraySize, 0, 1);
    // allocate device memory
    cudaMalloc((void**)&d_input, arraySize * sizeof(int));
    cudaMalloc((void**)&d_output, arraySize * sizeof(int));
    cudaMalloc((void**)&d_histLow, numPartitions * sizeof(int));
    cudaMalloc((void**)&d_histHigh, numPartitions * sizeof(int));
    cudaMalloc((void**)&d_prefixSum, numPartitions * sizeof(int));
    // copy input data from host to device
    cudaMemcpy(d_input, r_h, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    // initialize arrays on device to 0
    cudaMemset(d_histLow, 0, numPartitions * sizeof(int));
    cudaMemset(d_histHigh, 0, numPartitions * sizeof(int));
    // calculate grid size
    int gridSize = (arraySize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // events for timing properly
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // begin timing
    cudaEventRecord(start, 0);
    // launch the 3 kernels
    histogram<<<gridSize, BLOCK_SIZE>>>(d_input, d_histLow, d_histHigh, arraySize, numPartitions, bitsPerPartition);
    prefixScan<<<1, BLOCK_SIZE>>>(d_histLow, d_histHigh, d_prefixSum, numPartitions);
    Reorder<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_prefixSum, arraySize, numPartitions, bitsPerPartition);
    // stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // calculate elapsed time
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // copy results from device to host
    cudaMemcpy(h_output, d_output, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prefixSum, d_prefixSum, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histLow, d_histLow, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histHigh, d_histHigh, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
    // print partition information
    printf("Partition Information:\n");
    for (int i = 0; i < numPartitions; i++) {
        int offset = h_prefixSum[i];
        int count = (int)h_histHigh[i] * INT_MAX + h_histLow[i];
        printf("partition %d: offset %d, number of keys %lld\n", i, offset, count);
    }
    // print run time
    printf("******** Total Running Time of All Kernels = %.4f sec *******\n", elapsedTime / 1000.0);
    // free mem
    cudaFreeHost(r_h);
    cudaFreeHost(h_output);
    cudaFreeHost(h_histLow);
    cudaFreeHost(h_histHigh);
    cudaFreeHost(h_prefixSum);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_histLow);
    cudaFree(d_histHigh);
    cudaFree(d_prefixSum);
    return 0;
}
