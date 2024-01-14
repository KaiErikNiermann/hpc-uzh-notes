
------------------------------------------------------------------------

CUDA Introduction
-----------------

Compute Unified Device Architecture or CUDA for short is a closed-source
parallel computing platform and application programming interface that
enables computing using GPUs.

### CUDA Architecture
The CUDA architecture exposes the GPU parallelism for general purpose
computing while retaining performance.

### CUDA C/C++
CUDA Provides a C/C++ language extension ending in `.cu` that enables
the easy offloading of a traditional C/C++ code to the GPU for
computation with some minor alterations to the code.

### CUDA Terminology
**Host**
The host refers to the CPU and main memory.

**Device**
The device is the GPU along with its memory.

### CUDA Processing flow
The method of how data is processed using the GPU can be explained by
the following steps

1.  Copy input data from CPU memory to GPU memory

    Specifically the data is copied from the CPU memory to the GPU’s
    DRAM using the PCI Bus interface between the host and the device.

2.  Load the GPU program and execute it, caching data on chip for
    performance.

    Here we are loading the data and then using the GPU’s SMs to process
    the data in parallel by breaking it down into chunks and then
    distributing it amongst gangs. The GPUs memory is also used as a
    cache to increase performance, as it would be alot more costly to
    try and fetch the data from the main memory.

3.  Copy the result back to the CPU memory

    Here we copy the now processed data back from the GPU’s DRAM to the
    CPU Memory.

### CUDA Hello World
As stated before CUDA C/C++ enables GPU programming with just some minor
changes to the code, to demonstrate this we can compare traditional C
code with some CUDA C code.

`hello_world.cu`
``` code
#include <stdio.h>

int main(void) {
    printf("Hello world!\n");
    return 0;
}
```

**compile**
``` code
nvcc hello_world.cu ; ./a.out
```

Notes

Since CUDA C/C++ is just an extension of traditional C you can write and
compile regular C with the `nvcc` compiler.

`hello_world.cu`
``` code
#include <stdio.h>

__global__ void mykernel(void) {
}

int main(void) {
    mykernel<<<1, 1>>>();
    cudaDeviceSynchronize();  // Wait for the GPU to finish
    printf("Hello World!\n");
    return 0;
}
```

From this you might notice 2 new syntactical elements introduced in the
code.

**`__global__`** **keyword**

This is used to indicate that you want want the function `mykernel` to
run on the device ( GPU ) and that this function is being called from
the host ( CPU ).

The reason for this separation is that the host code, so the regular
code is passed to the gcc compiler which you would just with regular C
code anyways for example. Whereas the device code, so anything marked
with `__global__` in this instance is give to the NVIDIA compiler to
handle since this cannot be compiled in the same manner as device code.

**`<<< >>>`** **brackets**

This is used to mark a call from the host to the device code. In other
words its an indicator that a specific call we are making is coming from
the CPU and is calling a function meant to run on the GPU.

This is called a **kernel launch** and the numbers inside correspond to
the number of **blocks,** and **threads per block**. In this case we are
using 1 block and 1 thread per block.

Currently we have no actual code in the `mykernel` function so nothing
happens. The `printf` as you might have already guessed is executing on
the host.

------------------------------------------------------------------------

Basics of CUDA programming
--------------------------
We can start by going through a few basic examples, at the heart of it
GPU programming is above massive parallelism of specific tasks, a common
basic example of this is vector addition.

### Memory Management
A key thing to note when it comes to memory management in computations
that use both the CPU and GPU is that either compute uni has its own
memory, which by extension means that pointers point to *their own
memory*, which in turn means you must manage the memory separately on
either unit.

**Device memory**
May not be passed between host functions ( since its in the device
memory not host )

May not be dereferences in host functions ( again same pretty clear
reasoning )

**Host memory**
May be passed to/from device code

May not be dereferenced in the device code.

### Example - Scalar addition on the GPU
There are a few important things that need to be considered on a
fundamental level when you are working with GPUs. Specifically about how
we manage the memory.

-   We need to somehow allocate the memory on the GPU
-   We need to copy over the data to the GPU so it can work with it
-   We need to copy back and free the data on the GPU so we write safe
    code
-   We need to also manage the memory on the host

**Addition function**
Starting with the core of things with have the GPU addition function

``` code
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}
```

Nothing to special, one thing to note is that we are passing the
reference to the variables not the values, and then we are dereferencing
the variables for the addition process using the `*` . That is, we
accessing and modifying the value at the allocated GPU memory location.

**Main function**
We start off by doing some initializing, we want to add 3 numbers, so we
need space for this on the host and the device. We also want to have a
variable that expresses the size of the units we are working with. In
this case integers.

``` code
// Host variables
int a, b, c;
// Device variables
int *d_a, *d_b, *d_c;
// Size variable 
int size = sizeof(int)
```

We then wanna allocate the memory on the GPU, we created the pointers to
point to this memory, we then use this along with the `size` variable to
allocate the necessary memory at the locations specified by the device
variables.

`cudaMalloc`
``` code
cudaMalloc((void**)&d_a, size);
cudaMalloc((void**)&d_b, size);
cudaMalloc((void**)&d_c, size);
```

Some notes here are that the `cudaMalloc` function returns the address
on the device, so we are allocating `size` bytes and then the location
of these bytes on the device is assigned to the device variable pointers
we created.

`cudaMalloc` is also a relatively slow function which means its
generally used at startup, so a good rule is to try your best to
allocate all memory before you begin doing your computations.

You then want to setup the input values and copy the inputs tot he
device using the `cudaMemcpy` function.

`cudaMemcpy`
``` code
a = 2
b = 7
cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
```

Function is pretty self explanatory, we are copying `a` into the GPU
memory pointed to by `d_a` and we are specifying the direction host →
device using `cudaMemcpyHostToDevice` .

We then launch the `add()` kernel function using the `<<< >>>` notation
specifying 1 block and 1 thread long with the arguments.

``` code
add<<<1, 1>>>(d_a, d_b, d_c);
```

After which we use `cudaMemcpy` to copy the result back to the host
variable `c` from the GPU this time using the `cudaMemcpyDeviceToHost`
direction specifier.

`cudaMemcpy`
``` code
cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
```

Finally to clean up we use the `cudaFree` memory and optionally we can
now do whatever we want with the result as its back to being stored on
the host memory.

`cudaFree`
``` code
// Free device memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);

// Print the result
printf("Result: %d\n", c);
```

**full code**
The full code would then look something like this

[`add.cu`](http://add.cu)

``` code
#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main(void) {
    // Host variables
    int a, b, c;
    // Device variables
    int *d_a, *d_b, *d_c;

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    // Initialize host variables
    a = 2;
    b = 3;

    // Copy data from host to device
    cudaMemcpy(d_a, a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with a single block and single thread
    add<<<1, 1>>>(d_a, d_b, d_c);
            
        // Waiting for GPU to finish before executing
        cudaDeviceSynchronize();    

    // Copy the result from device to host
    cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

        // Print the result
    printf("Result: %d\n", *c);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c); 

    return 0;
}
```

### Error checking

Because CUDA functions generally don’t exit when an error occurs but
simply continue onwards it might be good to implement some sort of error
checking. For this you can use the `cudaGetErrorString(err_code)`
function which takes the `err_code` returned from a call like
`cudaMemcpy` and gives the string which explains what the error actually
means. You can implement this as a function or more simply as a macro
like this.

``` code
#define CUDA_CHECK_ERROR(err) \
    do { \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
```

And you would use this macro by simply wrapping it around any CUDA calls

``` code
CUDA_CHECK_ERROR(cudaMalloc((void**)&d_a, sizeof(int)));
```

### Terminology

**Block**

A set of blocks is called a **grid**.

Blocks are numbered sequentially from zero to some defined value, for
example if we say we want `N = 512` blocks then the blocks are numbered
0 → N - 1.

Each thread can use `blockIdx.x` to get its own block number, that is,
the block its contained within. And we can use these indexes to express
that each block should handle a specific index in for example an array
like when we do vector addition.

These blocks can naturally be executed in parallel by specifying this
when we launch a kernel function. So when we do

``` code
func_name<<<N, 1>>>( ... )
```

We are launching `N` blocks in parallel.

**Thread**

A block can be split into parallel threads.

To find the index of the thread we are currently in we use `threadIdx.x`

When we want to launch say `N` parallel threads we again specify this in
the kernel launch parameters

``` code
func_name<<<1, N>>>( ... )
```

### Example - vector addition

We can make some slight modifications to the code above to introduce
some parallelism with vector addition as opposed to basic scalar
addition.

Lets start by looking at the vector addition function, we can show two
variations using blocks or threads.

**Blocks**

``` code
__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
```

This has the corresponding launch kernel

``` code
add<<<N, 1>>>(d_a, d_b, d_c); // lauinching N parallel blocks
```

**Threads**

``` code
__global__ void add(int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
```

This has the corresponding launch kernel

``` code
add<<<1, N>>>(d_a, d_b, d_c); // lauinching N parallel blocks
```

**The rest of the code**

This stays largely the same, we just allocate in terms of arrays now
instead of scalar values, so it looks like this

``` code
// Host copies of a, b, c
int *a, *b, *c;

// Device copies
int *a_d, *b_d, *c_d;

int size = N * sizeof(int);

// Allocate device copies of a, b, c
cudaMalloc((void **)&a_d, size);
cudaMalloc((void **)&b_d, size);
cudaMalloc((void **)&c_d, size);

// Allocate host copies of a, b, c
a = (int *)malloc(size);
b = (int *)malloc(size);
c = (int *)malloc(size);

// Initialize host arrays (random_ints function assumed)
random_ints(a, N);
random_ints(b, N);
```

Were `random_ints(a, N);` is just some arbitrary function that populates
the arrays with some random values, you can create this yourself or
presumably find something online that implements this.

**The full code (using blocks)**

So the full code assuming we go with blocks for now would look something
like this

`vec_add.cu`

``` code
#include <stdio.h>

#define N 512

// Kernel function for vector addition on the device
__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main(void) {
    // Host copies of a, b, c
    int *a, *b, *c;

    // Device copies
    int *a_d, *b_d, *c_d;

    int size = N * sizeof(int);

    // Allocate device copies of a, b, c
    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, size);

    // Allocate host copies of a, b, c
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Initialize host arrays (random_ints function assumed)
    random_ints(a, N);
    random_ints(b, N);

    // Copy inputs to device
    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks and 1 thread per block
    add<<<N, 1>>>(a_d, b_d, c_d);

    // Copy result back to host
    cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}
```

------------------------------------------------------------------------

Combining Blocks and Threads
----------------------------

💡

For optimal performance you want to be combining blocks and threads in
your computations. But there are certain hardware limitations that have
to be considered with their application. It also comes at an increasing
cost of complexity with writing the code.

### Kernel launch

Before we get into combining threads and blocks it might make sense to
define how a kernel is actually launched. Specifically in terms of its
parallelism properties.

A kernel is launched a **single grid** which contains some number of
**blocks** which themselves contain some number of **threads**.

There are certain built in variables that allow you to look at the
dimensions and indices of everything, these are helpful in defining the
dimensions of your problem.

**Index ranges**

The thread indexes are defined from `0` → `blockDim.x` - 1

The block indexes go from `0` → `gridDim.x` - 1

**Indices**

The index of the block is defined by `blockIdx.x`

The index of the thread is defined by `threadIdx.x`

### Indexing arrays

A basic example to demonstrate how this hierarchy of grid → block →
thread is important is simply indexing arrays in parallel vector
addition which uses both blocks and threads.

Considering an example with 4 blocks and 8 threads, so programmatically

``` code
add<<<4, 8>>>(d_a, d_b, d_c); 
```

We can visualize the layout as follows

![](CUDA%204dedf8d793414be2be7ba00a74e5d5b3/Untitled.png)

Here we know that

-   `gridDim.x == 4` So we have 4 blocks

 

-   `blockDim.x == 8` So we have 8 threads

So if we want to define an index to a specific variable in our vector
then we have to use the following expression

$${\texttt{int}\ \texttt{index}} = \texttt{threadIdx.x} + \texttt{blockIdx.x} \times \texttt{blockDim.x}$$int index=threadIdx.x+blockIdx.x×blockDim.x

**Example - what thread will operate on the red element ?**

![](CUDA%204dedf8d793414be2be7ba00a74e5d5b3/Untitled%201.png)

We know that the block index is 2 the thread index is 5 and the
dimensions of the blocks are 8 hence we have

`int index = 5 + 2 * 8 = 21`

So we know that the 5th thread of second block will represents the 21st
index into the array.

### Example - vector addition with both blocks and indexes

The resulting function simply uses the expression we defined above the
calculate this new index

``` code
__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}
```

Now if we change the size of the array to

$2024^{2}$20242 we have to determine the number of blocks and threads
to specify. If we assume a block size of 512 for our problem then we can
use the following formulas to express the thread and block counts for
the kernel function call.

-   thread counts =
    $\texttt{block\_size}$block\_size

 

-   block counts =
    $N/\texttt{block\_size}$N/block\_size

Which means our kernel function call will look as follows

**Kernel launch**

``` code
add<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_a, d_b, d_c);
```

### Arbitrary vector sizes

If `N` is not a multiple of the chosen block size then we have we have
to use a different more general formula to account for this case to
avoid any out of bounds array accesses in the kernel function.

-   block counts =
    $(N + \texttt{block\_size})/\texttt{block\_size}$(N+block\_size)/block\_size

We also have to add an additional bounds check to the kernel function,
which makes it look as follows.

``` code
__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) c[index] = a[index] + b[index];
}
```

### Example - combing blocks and threads

So combining all the information above this is what our final code would
look like

``` code
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 560
#define BLOCKSIZE 512

// Function which populatates array of `size` with random integers
void random_ints(int *array, int size) {
    srand(time(NULL)); // Seed the random number generator with current time

    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 100; // Generate random integers between 0 and 99 (you can adjust as needed)
    }
}

// Kernel function for vector addition on the device
__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) c[index] = a[index] + b[index];
}

int main(void) {
    // Host copies of a, b, c
    int *a, *b, *c;

    // Device copies
    int *a_d, *b_d, *c_d;

    int size = N * sizeof(int);

    // Allocate device copies of a, b, c
    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, size);

    // Allocate host copies of a, b, c
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Initialize host arrays (random_ints function assumed)
    random_ints(a, N);
    random_ints(b, N);

    // Copy inputs to device
    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks and 1 thread per block
    add<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}
```

------------------------------------------------------------------------

Compute Capability
------------------

💡

While this is not necessarily that relevant at runtime it is something
that might be beneficial to specify during compilation.

### Example of compute level

![](CUDA%204dedf8d793414be2be7ba00a74e5d5b3/Untitled%202.png)

These are some common examples of the compute levels of recent cards,
these levels describe the hardware restrictions when writing CUDA code
for a specific GPU to run on.

### Compute level restrictions

![](CUDA%204dedf8d793414be2be7ba00a74e5d5b3/Untitled%203.png)

There are the restrictions associated with the different compute levels,
for example if you are writing CUDA code to be executed on a single GTX
1080ti then the maximum number of threads per block is 1024.

### Terminology Recap

To recap some of the terminology, specifically about **thread, warp,
block,** and **grid**

  -------- ---------------------------------------------------------------------------------------------
  thread   Each thread is executed by one core.  
           Cores have multiple threads resident at one time.  
           Only one thread is executing at one time.  

  warp     A warp consists of 32 cores that share register memory and operate on the same instruction.

  block    Each block is assigned to an SM which consists of multiple warps.  
           Each SM shares 64KB of memory that can be used to share data between threads.  

  grid     The kernel is run on a grid of blocks.
  -------- ---------------------------------------------------------------------------------------------

------------------------------------------------------------------------

Synchronizing between Host and Device
-------------------------------------

💡

A function you might’ve already seen here in the code is
`cudaDeviceSynchronize` . Kernel launches are async by default which
means that the host can continue processing while the kernel is
launched. So if we need a value from the GPU we have to wait for the
results to be ready to synchronize.

### Synchronizing

There are a few functions in particular that are relevant when we talk
about synchronizing.

  --------------------------- --------------------------------------------------------------------
  `cudaMemcpy()`              **Blocks** the host until the copy is complete.  
                              The copy begins when all preceding kernel calls have completed.  

  `cudaMemcpyAsync()`         **Does not block** the host.  
                              The copy occurs after all preceding kernel calls have completed.  

  `cudaDeviceSynchronize()`   Waits for all transfers and kernel calls to complete.
  --------------------------- --------------------------------------------------------------------

------------------------------------------------------------------------

Detecting Errors
----------------

💡

As already mentioned previous detecting errors is something always
important. The CUDA API calls all return an error code of type
`cudaError_t`

### Error code

This code can be either

-   An error in the API call, so for example when we call `cudaMemcpy()`

 

-   An error in an earlier asynchronous operation, such as a kernel
    launch

### Retrieving the last error

Sometimes you might not be able to easily get the return value of a
function you just called, in which case you can use the
`cudaGetLastError(void);` function which returns the last CUDA error
that was triggered.

### Converting error code

Since both the different CUDA operations (API calls, async operations)
just return error codes, that is, codes which represent errors these
aren’t really that human interpretable. Which is why there is the
function `cudaGetErrorString(cudaError_t)` which takes a `cudaError_t`
and returns a `char *` so a pointer to a string of characters which
express what the error means in a human readable format.

Using the aforementioned macro we can print the error as follows

``` code
#define CUDA_CHECK_ERROR(err) \
    do { \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
```

Here we are printing the error to the `stderr` stream and exiting the
program.

------------------------------------------------------------------------

Multiple Device Management
--------------------------

💡

A host may have more than one connected device, that is, a CPU may have
multiple GPUs it can interface with, so CUDA has functions that assist
in the management of these devices.

### Device management functions

|                                  |                                                                             |
|----------------------------------|-----------------------------------------------------------------------------|
| `cudaGetDeviceCount(int *count)` | Write the number of connected devices into the memory pointed to by `count` |
| `cudaSetDevice(int device)`      | Set the device we are using to the one specified by the identifier `device` |
| `cudaGetDevice(int *device)`     | Write the id of the current device to the memory pointed to by `device`     |

**Sharing a device**

Multiple host threads can share a device, this is a thread safe
operation.

------------------------------------------------------------------------

3D Indexing
-----------

💡

A kernel is launched as a grid of blocks and threads. Before we were
only using the `x` component of this grid but the builtin variables also
have a `y` and `z` component. Which is a useful feature for problems
that map into higher dimensions easily.

### Total number of threads

The total number of threads can be defined by

![](CUDA%204dedf8d793414be2be7ba00a74e5d5b3/Untitled%204.png)

------------------------------------------------------------------------

Sharing Data Between threads
----------------------------

💡

How memory is shared between threads is another important thing to
consider.

### Notes on shared properties

Within a block

-   Threads can share data via shared memory

Different blocks

-   Data is not visible to threads in different blocks

`__shared__`

-   Used to declare shared allocates between threads on the same blocks

`__syncthreads()`

-   Used to sync threads within a thread block, since threads can
    execute in any order

### Shared memory reduction

The parallel reduction operation is a good example of how we can use
shared memory in the process of optimizing parallel reduction.

------------------------------------------------------------------------
