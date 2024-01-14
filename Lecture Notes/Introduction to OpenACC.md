
------------------------------------------------------------------------

OpenACC structure
-----------------

💡

All programs which make use of OpenACC consist of 3 main parts :
**compilation directives and clauses**, **functions and routines**,
**environment variables**

### Compilation directives and clauses

**Main functions**

-   Parallelize loops

 

-   Define work and data sharing strategy

 

-   Synchronize different threads/cores

### Functions and routines

OpenACC contains several dedicated functions (similar to MPI) which are
a part of the OpenACC library and can be linked together along with your
program during the build process.

These functions manage different aspects of the parallel environment.

### Environment variables

OpenACC has several environment variables that can be set at execution
time and change the parallel computing behavior.

------------------------------------------------------------------------

GPU Model and Operations
------------------------

💡

OpenACC can target both CPU and GPU architectures so one important thing
to understand is how OpenACC actually interacts with the GPU.

### GPU Model

![](Introduction%20to%20OpenACC%200686e789539744f78902f0ee33207073/Untitled.png)

**Observations**

The GPU has significantly more threads per core allowing for higher
parallelism.

The connection between the GPU and its memory is faster than that
between the CPU and the main memory. Though of note is that access to
the lower cache levels occurs the fastest due to the embedded nature.

The communication between the main memory, that is, the RAM, and the GPU
is slow.

### GPU Operations

When OpenACC wants to execute something on the GPU it follows the
following steps

1.  Allocate or free some GPU memory for the executing code

 

1.  Copy data from the “host” to the GPU memory

 

1.  Launch a “kernel” which just means that you flag a routine to run on
    the GPU, so its compiled for the GPU to be executed on there instead
    of the CPU

 

1.  Copy data from the GPU to the host

**Notes**

OpenACC handles most of these operations for you, so it abstracts away
alot of the underlying complexity that comes with GPU programming.

Transfers and Kernels can overlap, in other words data transfers between
the main memory and the GPU memory can occur at the same time as the
Kernels are executing which can help improver performance.

------------------------------------------------------------------------

SAXPY
-----

💡

SAXPY refers to **single-precision** **aX+YaX + YaX+Y**, which is just
an acronym that refers to expressions which include both scalar
multiplication

$aX$aX and vector addition

$X + Y$X+Y where

$X,Y \in \mathbb{R}^{n}$X,Y∈Rn . Additionally the result of this
specific type of expression is **stored in** **YYY**

### Implemented SAXPY

A basic implementation of this type of operation in C would look
something like this

``` code
void saxpy(int n, float a, float * restrict x, float * restrict y) {
    // Loop through each element of the vectors
    for (int i = 0; i < n; ++i) {
        // Perform the SAXPY operation: y[i] = a*x[i] + y[i]
        y[i] = a * x[i] + y[i];
    }
}
```

This function is accessible in the Basic Linear Algebra Subroutines
(BLAS) Library.

**Float to Byte ratio**

The Float to Byte (FBR) ratio is defined as the number of floating point
operations per second divided by the number of bytes transferred per
second.

This is an important metric of performance because it can convey the
situation of a bottleneck, for example if the throughput of data is not
fast enough then it can limit certain systems with a very high number of
FLOPS.

Since the SAXPY operation performs many floating point operations its a
key type of operation that can be effected by this type of bottleneck.

### Different approaches to computing

Since with SAXPY we are adding multiple things at a time we can either
have serial or concurrent approaches to computing this scalar
multiplication and vector addition.

**Serial approach**

Here we are just consecutively computing each entry into the vector

$Y$Y for a total of

$n$n loop iterations.

$$Y_{0} = aX_{0} + Y_{0} Y_{1} = aX_{1} + Y_{1}\ldots Y_{n} = aX_{n} + Y_{n}$$Y0​=aX0​+Y0​Y1​=aX1​+Y1​…Yn​=aXn​+Yn​

**OpenMP approach**

Since these are logically independent calculations we can divide the
work up evenly amongst

$k$k threads for example.

Here we separated the work for a vector of size

$n = 4000$n=4000 evenly amongst 4 threads

$$\left. k = 0:Y_{0}\rightarrow Y_{999} k = 1:Y_{1000}\rightarrow Y_{1999} k = 2:Y_{2000}\rightarrow Y_{2999} k = 3:Y_{3000}\rightarrow Y_{3999} \right.$$k=0:Y0​→Y999​k=1:Y1000​→Y1999​k=2:Y2000​→Y2999​k=3:Y3000​→Y3999​

**GPU approach**

Because the massive level of parallelism a GPU allows for we follow the
same principle of dividing up the work amongst the different GPU threads
but just to a much greater extent.

$$k = 0:Y_{0} = aX_{0} + Y_{0} k = 1:Y_{1} = aX_{1} + Y_{1}\ldots k = 3999:Y_{4000} = aX_{4000} + Y_{4000}$$k=0:Y0​=aX0​+Y0​k=1:Y1​=aX1​+Y1​…k=3999:Y4000​=aX4000​+Y4000​

------------------------------------------------------------------------

Basic Principles of OpenACC
---------------------------

💡

There are four key principles to remember with OpenACC, that being the
use of **directives**, the **creation of parallel regions and offloading
work to GPU, data transfer can be implicit or explicit**, **kernel
invocations are expensive.**

### Directives

As mentioned before the fundamental way most programmers are going to
use OpenACC is through directives, the same principle as with OpenMP,
but in this case its generally meant for parallel programming using the
GPU.

### Parallel regions

Again similar to OpenMP we use directives to indicate to the machine
that we want to create a parallel region here which then leads to the
work being executed by new spun up threads. In the case of OpenACC it
works the same way but the work in the parallel regions is offloaded to
the GPU threads instead of CPU threads.

### Data transfer

Data transfer between the main memory and the GPU can be implicit, that
is, its handled without any explicit instruction from the programmer,
just implicitly as an inherent feature of some directives. Data transfer
can also be explicit so the programmer explicitly specifies that they
want to transfer certain data to the GPU through the use of certain
instructions.

### Kernel invocations

Similar to the latency caused by starting up a parallel region using MPI
the cost of invoking the GPU kernel to execute some routine is an
expensive processes whos trade-off should be well considered.

### CPU and GPU exchange diagram

This demonstrates an abstract example where we have some CPU code we are
executing then we do a kernel invocation which initiates some type of
data transfer where the code is executed in a parallel manner after
which the execution returns back to the CPU.

![](Introduction%20to%20OpenACC%200686e789539744f78902f0ee33207073/Untitled%201.png)

------------------------------------------------------------------------

Directive Syntax
----------------

💡

Similar to OpenMP directives are specified with the `#pragma` mechanism
and all follow a similar syntax

![](Introduction%20to%20OpenACC%200686e789539744f78902f0ee33207073/Untitled%202.png)

------------------------------------------------------------------------

OpenACC setup and basic compiling
---------------------------------

💡

To start working with OpenACC you need to ensure that you have the
proper compiler to work with GPU code setup and you need to link to
OpenACC in the compile command.

### Compiler setup

The way to get OpenACC setup on your home computer is to work with the
NVIDEA High Performance Computing Software Development Kit

![](https://dirms4qsy6412.cloudfront.net/nv/assets/favicon-81bff16cada05fcff11e5711f7e6212bdc2e0a32ee57cd640a8cf66c87a6cbe6.ico)

Once you have everything setup you should be able to run
`nvc++ --version` and get something like the following output

``` code
nvc++ 21.3-0 LLVM 64-bit target on x86-64 Linux -tp haswell
NVIDIA Compilers and Tools
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
```

### Compiling with OpenACC

The basic compile command for OpenACC generally consists of the
following parts

``` code
CC \ 
-O3 \ 
-acc \
-Minfo=acc \
-o saxpy \
saxpy.cpp
```

-   the compiler you are using

 

-   the optimization level

 

-   enabling OpenACC

 

-   enabling OpenACC info

 

-   the output executable

 

-   the input source code

So the final command just written out again would be something like this

``` code
CC -O3 -acc -Minfo=acc -o saxpy saxpy.cpp
```

------------------------------------------------------------------------

Kernels construct
-----------------

💡

The kernel refers to a region of code you specify that **may** contain
parallelism. The compiler will analyze the block and **if appropriate**
generate one or more kernels and data transfer operations.

### General syntax

The general syntax of a kernel block is as follows

``` code
#pragma acc kernels 
{
    // parallel block 
}
```

### Example - kernel in SAXPY operation

Here we define a kernel for two loops, we can compile this using the
previous command which should give us the following output.

`saxpy.cpp`

``` code
#include <iostream>

int main() {
    const int N = 1'000'000'000;
    float* x = new float[N];
    float* y = new float[N];

    #pragma acc kernels
    {
        // Initialize vectors x and y
        for (int i = 0; i < N; i++) {
            y[i] = 0.0f;
            x[i] = static_cast<float>(i + 1);
        }

        // Perform SAXPY operation: y = 2.0f * x + y
        for (int i = 0; i < N; i++) {
            y[i] = 2.0f * x[i] + y[i];
        }
    }

    // Cleanup: release allocated memory
    delete[] x;
    delete[] y;

    return 0;
}
```

`output`

``` code
main:
      9, Generating implicit copyout(y[:1000000000],x[:1000000000]) [if not already present]
     11, Complex loop carried dependence of y-> prevents parallelization
         Loop carried dependence of x-> prevents parallelization
         Loop carried backward dependence of x-> prevents vectorization
         Accelerator serial kernel generated
         Generating Tesla code
         11, #pragma acc loop seq
     11, Loop carried dependence of x-> prevents parallelization
     17, Complex loop carried dependence of x-> prevents parallelization
         Loop carried dependence of y-> prevents parallelization
         Loop carried backward dependence of y-> prevents vectorization
         Accelerator serial kernel generated
         Generating Tesla code
         17, #pragma acc loop seq
     17, Loop carried dependence of y-> prevents parallelization
         Loop carried backward dependence of y-> prevents vectorization
```

Breaking down this output one by one

`copyout`

``` code
9, Generating implicit copyout(y[:1000000000],x[:1000000000]) [if not already present]
```

This refers to OpenACC creating a region to copy the data back from the
GPU once the computation has finished.

**`loop carried dependence`**

This just refers to the fact that the compiler is assuming that the two
variables `x` and `y` are *aliased* that is, its assuming they both
point to the same/overlapping memory. This assumption means that because
we are updating both of these pointers in the same loop the compiler
assumes that the modification of one pointer affects the other which
creates the loop carried dependence.

`Accelerator serial kernel code generated`

This refers to the fact that due to the loop dependence preventing
parallelization the work is still offloaded to the GPU but not as a
parallel kernel but a serial one, so we get a serial execution of the
work on the GPU.

### Example - SAXPY using `restrict` keyword

`saxpy.cpp`

``` code
// ...
float* restrict x = new float[N];
float* restrict y = new float[N];

// ...
```

`output`

``` code
main:
      9, Generating implicit copyout(x[:1000000000],y[:1000000000]) [if not already present]
     11, Loop is parallelizable
         Generating Tesla code
         11, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
     17, Loop is parallelizable
         Generating Tesla code
         17, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
```

**Why are the loops suddenly parallelizable ?**

Because the use of the `restrict` keyword is just the programmer telling
the compiler that the two pointers `x` and `y` definitely point to
*different* regions of memory which can be updated independently of one
another without effecting the other.

------------------------------------------------------------------------

Parallel and Loop construct
---------------------------

💡

The `parallel` construct defines a block of code that will be
parallelized. Some points are that **its the programmers responsibility
to ensure safety, best used with** `loop` **directive.**

### Parallel construct syntax

``` code
#pragma acc parallel
{
    // parallel block body
}
```

Generally combined with the loop construct in the parallel block body

### Loop construct syntax

``` code
#pragma acc parallel
{
    #pragma acc loop 
    for (...) { 
        // loop body    
    }
}
```

### Example - parallel and loop construct in SAXPY operation

If we use the parallel and loop construct we can tell the compiler that
it is safe to parallelize the loops which gives us the following output.

`saxpy.cpp`

``` code
#include <iostream>

int main() {
    const int N = 1'000'000'000;
    float* x = new float[N];
    float* y = new float[N];

    #pragma acc parallel
    {
        #pragma acc loop
        for (auto i = 0; i < N; i++) {
            y[i] = 0.0f;
            x[i] = static_cast<float>(i + 1);
        }

        #pragma acc loop
        for (auto i = 0; i < N; i++) 
            y[i] = 2.0f * x[i] + y[i];
    }

    // Cleanup: release allocated memory
    delete[] x;
    delete[] y;

    return 0;
}
```

`output`

``` code
main:
      9, Generating Tesla code
         11, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
         17, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
      9, Generating implicit copyout(y[:1000000000],x[:1000000000]) [if not already present]
```

Here we can see the directive generated by the compiler is the same as
for using the `restrict` keyword. So we can use this as an alternative
method to clearly tell the compiler that it is safe to parallelize the
loop.

⚠️ It should be noted that same as with using the restrict keyword if
its not actually safe then the parallelized code will exhibit undefined
behavior, so this should be taken into account.

### Example - **Profiling our example code**

To profile our code we can use the `nvprof` utility by just passing the
name of the executable as an argument. As follows

``` code
nvprof ./saxpy 
```

------------------------------------------------------------------------

Data construct
--------------

💡

The data construct is used to explicitly modify and transfer data
between the host CPU and GPU, there are various operations associated
with the data construct including `copy`**,** `copyin`**,**
`copyout`**,** `create`**,** `present`**,** `present_or_*`

### Data construct syntax

``` code
#pragma acc data clauses 
{
    // parallel block(s)
}
```

### Data construct operations

|                |                                                                              |
|----------------|------------------------------------------------------------------------------|
| `copy`         | Allocate and copy variable to the device and copy it back at the end         |
| `copyin`       | Allocate and copy to the device                                              |
| `copyout`      | Allocate space but do not initialize. Copy to host at the end.               |
| `create`       | Allocate space but do not initialize or copy back to the host.               |
| `present`      | The variable is already present on the device (when data regions are nested) |
| `present_or_*` | An example of this would be `pcreate`                                        |

### Example - data construct used for SAXPY

We can use data constructs to explicitly pass the necessary arrays to
and from the GPU for the SAXPY computation.

`saxpy.cpp`

``` code
#include <iostream>

int main() {
    const int N = 1'000'000'000;
    float* x = new float[N];
    float* y = new float[N];

    #pragma acc data pcreate(x[0:N]) pcopyout(y[:N])
    {
        #pragma acc parallel loop
        for (auto i = 0; i < N; i++) {
            y[i] = 0.0f;
            x[i] = static_cast<float>(i + 1);
        }

        #pragma acc parallel loop
        for (auto i = 0; i < N; i++) {
            y[i] = 2.0f * x[i] + y[i];
        }
    }

    delete[] x;
    delete[] y;

    return 0;
}
```

Here we used the data construct and two of the data construct operations

`pcreate(x[0:N])`

This clause is used to tell the compiler that you want to create a
device pointer for the array `x` going from 0 → N on the accelerator
device (GPU) and have it managed on this device in the subsequence
parallel loop.

`pcopyout(y[:N])`

This clause tells the compiler that we want to copy back the data at
pointer `y` from range 0 → N *after* the parallel loop, that is, after
everything has completed executing. Which I think makes sense because
`y` is where the results from our computation end up.

### Example - nested data regions with SAXPY

Since we don’t always want all our code to just be in one file we might
sometimes run into situations where we are using data constructs inside
data constructs, that is, when we are using data constructs in functions
we are calling in data constructs for example.

`saxpy.cpp`

``` code
#include <iostream>

void init(int N, float* x, float* y) {
    #pragma acc data pcopyout(x[0:N]) pcopyout(y[0:N])
    {
        #pragma acc parallel loop
        for (auto i = 0; i < N; i++) {
            y[i] = 0.0f;
            x[i] = static_cast<float>(i + 1);
        }
    }
}

void saxpy(int N, float a, float* restrict x, float* restrict y) {
    #pragma acc data pcopyin(x[0:N]) pcopy(y[0:N])
    {
        #pragma acc parallel loop
        for (auto i = 0; i < N; i++) {
            y[i] = a * x[i] + y[i];
        }
    }
}

int main() {
    const int N = 1'000'000'000;
    auto x = new float[N];
    auto y = new float[N];

    #pragma acc data pcreate(x[0:N]) pcopyout(y[0:N])
    {
        init(N, x, y);
        saxpy(N, 2.0f, x, y);
    }

    // Here you would use the result ‘y’

    // Don't forget to free the allocated memory
    delete[] x;
    delete[] y;

    return 0;
}
```

`output`

``` code
init(int, float *, float *):
      5, Generating copyout(y[:N],x[:N]) [if not already present]
         Generating Tesla code
          7, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
saxpy(int, float, float *, float *):
     16, Generating copy(y[:N]) [if not already present]
         Generating copyin(x[:N]) [if not already present]
         Generating Tesla code
         18, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
main:
     30, Generating copyout(y[:1000000000]) [if not already present]
         Generating create(x[:1000000000]) [if not already present]
```

We can see here just as before that for each function in the format
`funcname(arguments):` has certain code which is generated by the
compiler to preform data transfer operations with the GPU.

### Updating in the data region

One common type of situation you might run into is that you want to do
some work on the GPU and some work on the CPU.

It would make sense that you don’t want to start up new data sections
each time since that would just lead to more boilerplate, if you have
some sequence of methods that perform computations on the CPU and GPU
you want to create a single data section and then transfer the data
between the two devices.

`example.cpp`

``` code
#pragma acc data ...
{
    do_some_gpu_work(x, ...);
    #pragma acc update self(x)
    do_something_on_the_cpu(x);
    #pragma acc update device(x)
    do_more_gpu_work(x, ...);
}
```

The two key tools we use here are `update self(x)` which ensures that
the CPU has the latest version of the value `x` to work on.

Likewise then when we want the GPU to modify this value we use
`update device(x)` such that the GPU has the updated value.

**Synchronicity**

The main idea behind `update` is that we synchronize the data between
the two devices.

### Unstructured data regions

Unstructured refers to data regions that don’t follow the traditional
syntax of `#pragma acc data {}`

but instead can be more dynamically placed.

Broadly speaking unstructured data regions are often needed in scenarios
where the data lifetime is not directly died to a clear block of code.

An example of this a constructor and destructor in a C++ class.

`unstructured_data_example.cpp`

``` code
class foo {
    float* v;

public:
    foo(int n) {
        v = new float[n];
        #pragma acc enter data create(v[:n])
    }

    ~foo() {
        #pragma acc exit data delete(v)
        delete[] v;
    }
};
```

`#pragma acc enter data create(v[:n])`

This line tells the compiler that

-   We are entering the data region

 

-   The data pointed to by `v` is entering the data region for all
    elements up to `n`

`#pragma acc exit data delete(v)`

This line tells the compiler that

-   The associated data `v` is leaving the data region

 

-   The copy `v` should be deleted from the device

------------------------------------------------------------------------

Data scope
----------

💡

Data scope is important for managing how variables are shared or private
across different threads, similar to the OpenMP counterpart for CPU
threads.

### Scalar and Loop index variables

Scalar values and loop index values are private by default, so each
thread gets their own copy. Obviously this implies that changes to such
values in one thread do not effect any other thread.

**Differences from OpenMP**

This is fundamentally different as from OpenMP where these variables are
shared by default.

### Arrays shared by default

Arrays, in contrast to scalar values and loop indexes, are shared by
default where all threads have access to the same memory locations of
the array.

### Overriding default scoping - explicit data clauses

You can override the automatic scoping decisions through the use of
explicit data clauses that define the scoping rules.

|                             |                                                                                                                       |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `private`                   | Clause which can be used with directives like `parallel`, `kernels`, `loop` to explicitly specify private variables.  |
| `copyin`, `copyout`, `copy` | Used for controlling data movement between the host and the device (GPU)                                              |
| `default(none)`             | This is used to enforce specific scoping rules for all variables the compile will show an issue if you fail to do so. |

------------------------------------------------------------------------

Blocks, Gangs, Vectors, and Workers
-----------------------------------

💡

To understand how parallelism works there are 4 main concepts that
should be understood, this being **blocks, gangs, vectors,** and
**workers.**

### Blocks (Thread Block)

Blocks in the context of NVIDIA GPU’s refers to a group of threads that
execute together on a single streaming multiprocessor. That is, all
threads within a particular thread block must reside on a single SM.

### Levels of granularity

In terms of how parallel code is executed vectors, workers, and gangs
basically just describe the levels of granularity. With vectors
corresponding to the SIMD-like execution of an instruction, workers
corresponding to a grouping of one or more vectors and a gang being a
grouping of workers.

### Vector

A vector corresponds to some grouping of threads of a thread block on
which a single instruction is executed in an SIMD-like fashion.

One way a vector is implemented is in NVIDIA through the concept of a
WARP which same as the idea of a vector executes an instruction as a
group of 32 threads belonging to some thread block.

**Vector/Warp length**

You should generally set the vector length to either 32 or a multiple of
32 threads for efficiency sake.

### Worker

Again worker is an abstraction that’s meant to refer to a grouping of
one or more vectors but in the context of NVIDIA GPUs workers refer to
the Y-axis layout of a block.

Put in another way a worker refers to some code which operates on one or
more vectors of data using vector operations, in the context of GPUs
these are single instruction multiple thread operations, analogous to
SIMD in the case of the CPU.

### Gang

This roughly refers to a grouping of workers. They are independent and
may operate in parallel or even at different times. They can also share
resources.

Gangs are scheduled by the GPU for the available resources, this being
the SMs.

------------------------------------------------------------------------

Using Gangs, Vectors, and Workers
---------------------------------

💡

In OpenACC you can manually specify the specific way in which a loop
should be parallelized.

### Example - Gang and Vector loop

``` code
#pragma acc parallel loop gang
for (int i = 0; i < n; ++i)
    #pragma acc loop vector
    for (int j = 0; j < n; ++j)
        // Code inside the loop
```

Here we are expressing that we want the outer loop to be parallelized
using gangs and we want the inner group to be parallelized using
vectors.

In other words, multiple gangs will execute the outer loop and the inner
loop will be processed using vectorization.

### General Rule

A general rule is to use gangs or workers for outer loops and vectors
for inner loops.

This is because vectors are well-suited for parallelizing operations
within a thread, namely fine grain parallelism. Whereas gangs and
vectors are better used to handle parallelism across threads.

**Optimal choices for vector size and number of gangs**

The optimal vector size and number of gangs is the maximal amount
possible within the constraints of the hardware.

Additionally for the vector size its a multiple of 32.

### Example - Loop with workers

``` code
#pragma acc parallel vector_length(32)
#pragma acc loop gang worker
for (int i = 0; i < n; ++i)
    #pragma acc loop vector
    for (int j = 0; j < m; ++j)
        // Code inside the loop
```

Here we set the vector length to `32` and additionally specified the
loop should be parallelized. We also stated that we want a gang of
workers to be utilized.

Using workers can help improve the occupancy which in turn generally
correlates with a higher efficiency because are are better utilizing the
SMs on the GPU.

You can also set the number of workers explicitly with `num_workers(2)`

------------------------------------------------------------------------

Sequential and Independent Loops
--------------------------------

💡

Sometimes loops need to be executed in a sequential manner, for example
if the result depends on a previous iteration. For this you can use the
`seq` clause. Alternatively to indicate sequential loops you can use the
`independent` clause.

### Example - seq. and independent loops

``` code
#pragma acc data copyin(a, b) copy(c)
#pragma acc kernels  
#pragma acc loop independent
for (int i = 0; i < n; ++i) {
    #pragma acc loop independent
    for (int j = 0; j < n; ++j) {
        #pragma acc loop seq
        for (int k = 0; k < n; ++k) {
            c[i][j] += a[i][k] * b[k][j];
        }
    }
}
    
```

------------------------------------------------------------------------

Collapse
--------

💡

The `collapse` clause in OpenACC is used to combine multiple loops into
a single larger loop. Collapsing outer loops creates more gangs allowing
for larger parallel execution of dataset. Collapsing inner loop enables
longer vector lengths improving efficiency of vectorized operations.

### General rule

You should always try to collapse loops where possible to enhance the
parallelism and vectorization.

### Example - loop collapse

**Using the collapse directive**

``` code
#pragma acc parallel loop collapse(2)
for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
        // Code inside the collapsed loop
```

**Equivalent “pre-collapsed” code**

``` code
#pragma acc parallel loop
for (int ij = 0; ij < n * m; ++ij)
    // Code inside the loop
```

------------------------------------------------------------------------

Reduce
------

💡

The `reduction` clause works the same way it does in all other parallel
frameworks, you reduce some variable across a parallelized loop, so you
just aggregate the results of the operations onto a single point.

### Usage and Benefit

You can use the reduction clause with other things like `parallel`,
`kernel`, `loop`

It supports a variety of reduction operations such as `sum`, `product`,
`min`, `max`, and more

**Benefit**

Using the `reduction` clause can help prevent race conditions because
the OpenACC compiler can optimize the reduction operation for parallel
execution.

### Example - loop reduction

``` code
// Example: Compute the sum of an array using reduction
#pragma acc parallel loop reduction(+:sum)
for (int i = 0; i < n; ++i) {
    sum += array[i];
}
```

------------------------------------------------------------------------

Atomic
------

💡

The `atomic` directive works the same way it does for OpenMP.

### Benifits

Avoid race conditions

Possible performance impact

### Example - atomic directive

``` code
#pragma acc data copyin(a[:n]) copyout(h[:nbins])
{
    #pragma acc parallel loop
    for (int i = 0; i < nbins; ++i)
        h[i] = 0;

    #pragma acc parallel loop
    for (int i = 0; i < n; ++i) {
        #pragma acc atomic update
        ++h[a[i]];
    }
}
```

------------------------------------------------------------------------

Functions in parallel regions
-----------------------------

💡

To call a function inside a parallel region you have to use the
**`routine`** directive and specify the level of parallelization, this
being `gang`**,** `worker`**,** `vector`**,** or `seq`

### Routine Directive

The routine directive specifies that the compiler should generate both a
GPU and CPU version of the function. To work with the routine directive
there are two key considerations

**Function source**

At the function source, so where you declare the function, you have to
use the `reduction` clause with the parallelism level, here the function
will then be built for the GPU.

``` code
// foo.h
#pragma acc routine seq
double foo(int i);
```

**Function call**

Here you call the function in the parallel block. The parallel block
will then, well, parallelize your work, but the routine you call in this
block will execute in whatever mode you specify.

``` code
// in main()
#pragma acc parallel loop
for(int i=0; i<n; ++i)
    array[i] = foo(i);
```

------------------------------------------------------------------------

Synchronization
---------------

💡

An implicit synchronization happens when leaving a kernels parallel
region. All threads will have completed before the region ends.

### Inside a parallel region

Inside a parallel region there is no synchronization. Which means a few
key notes

-   A second loop inside a parallel region could start before all
    threads doing the first loop finished.

 

-   The order of the loop execution is not preserved - indexes are
    processed in any order.

### Loop execution order in the kernel construct

Each loop in a separate kernel invocation on the device. Thus
maintaining the order of the different loops executing, though if the
loops are chosen to be parallelized then the same logic for the indexes
applies.

------------------------------------------------------------------------

Asynchronous Programming
------------------------

💡

Most OpenACC directives can be made asynchronous. Doing this means that
the host issues multiple parallel loops to the GPU. The host can preform
other calculations while the GPU is busy. Data transfers can happen
before the data is needed.

### Synchronous programming

This is the default approach when using OpenACC some important
properties of this are that

-   Work is scheduled in parallel sections ( or not )

 

-   When the sections finishes the host waits for the GPU

    This means that the host has to wait for all kernels to finish
    executing and all data transfers to complete.

### `async` and `wait`

`async(n)`

Clause which launches work asynchronously in queue `n`

`wait(n)`

Directive that waits for all the work in queue `n` to complete.

**Notes**

This can significantly reduce launch latency and enables pipelines and
concurrent operations.

If `n` is not specified then work will go to the default queue, and wait
will wait for all previously queued work.

### Example - `async` and `await`

``` code
#pragma acc parallel loop async(1)
…
#pragma acc parallel loop async(1)
for(i=0; i<N; ++i) …

// host can work independently here

#pragma acc wait(1)
for(i=0; i<N; ++i) … // Host deals with array
```

------------------------------------------------------------------------

Queues are CUDA streams
-----------------------

💡

In the context of NVIDIA GPUs and CUDA programming a “queue” is used
interchangeably with stream.

-   A stream does one thing at a time, in order (often what you want):
    -   Copy data to the GPU

     

    -   Execute a kernel

     

    -   Execute a second kernel …

     

    -   Copy results back to the CPU

 

-   With multiple queues/streams you can run multiple kernels at once
    -   Subject to resource constraints

     

    -   There are a fixed number of SMs

     

    -   You can have only one copy (possibly each direction) active at a
        time.

![](Introduction%20to%20OpenACC%200686e789539744f78902f0ee33207073/Untitled%203.png)
