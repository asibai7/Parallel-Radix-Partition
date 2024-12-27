# Parallel-Radix-Partition

This project implements a Parallel Radix Partition algorithm designed to efficiently partition datasets using a radix-based approach. The implementation leverages CUDA to accelerate partitioning by processing data in parallel on a GPU.

The project demonstrates the advantages of GPU-based partitioning over a sequential CPU implementation, with significant performance gains for large datasets. The algorithm includes optimizations such as memory coalescing, shared memory utilization, and efficient workload distribution across threads.

Requirements:<br>
To compile and run the program, you need access to a CUDA-enabled machine. Ensure you have nvcc installed to compile CUDA code.

To install and use:
1. Clone the repository:
```
git clone https://github.com/asibai7/Parallel-Radix-Partition.git
cd Parallel-Radix-Partition
```
2. Compile the code using nvcc on a CUDA-enabled machine:
```
nvcc PRP.cu -o PRP
```
3. Run program:
```
./PRP <inputSize> <numPartitions>
```
Example: 
```
./PRP 1000000 512
```
## Testing Results  

| **Test** | **Number of Elements** | **Number of Partitions** | **Time (Seconds)** | **Time (Milliseconds)** |
|----------|-------------------------|--------------------------|--------------------|--------------------------|
| 1        | 50,000,000              | 256                      | 0.0181             | 18.1                     |
| 2        | 1,000,000               | 512                      | 0.0002             | 0.2                      |
| 3        | 100,000,000             | 1024                     | 0.0786             | 78.6                     |

<br>(inputSize) The number of elements in input array (must be >= 1,000,000.  
(numPartitions) The number of partitions (must be a power of 2, between 2 and 1024).
