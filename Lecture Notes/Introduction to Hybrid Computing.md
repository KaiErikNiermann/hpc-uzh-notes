
Hybrid Computing Model
----------------------

💡

The Hybrid computing model refers to the way computing is done in a
contemporary sense where you generally have the CPU interface with the
memory for computations.

### Moores Law

Moores law states that the number of transistors on microchips doubles
on average every 2 years.

![](Untitled.png)

### Dynamic vs Static RAM

Both of are random access memory types that allow the storage of working
data for the CPU. The differences are detailed as follows

**Dynamic RAM**

DRAM is the “normal” RAM type that you can find in servers, computers,
and the main memory of GPUs.

-   **Single capacitor** to hold **each bit**

 

-   **Capacitors** leak so they must be periodically **refreshed** to
    hold their state

 

-   DRAM has **worse performance** but also a **larger capacity**

**Static RAM**

SRAM is used in various type of electronics that require a higher
performance memory for working storage.

-   **Six transistors** to hold each bit

 

-   Does not use capacitors thus does not need any refreshing

 

-   SRAM has **better performance** but a **lower capacity**

### Modern computing models

In modern memory layout for the CPU you typically have a layer of **SRAM
Cache Memory** which interfaces with the **memory controller** which
itself interfaces with the slower **DRAM.**

![](Untitled%201.png)

**Multiple levels of cache**

The cache is then usually split into multiple levels with some being
closer to the CPU to access ( e.g. Level 1 cache ) but having a smaller
storage and some further away with a larger storage (e.g. Level 3
cache). L2 cache can be both embedded on the CPU along with L1 cache or
it can be separate.

*embedded L2 cache*

![](Untitled%202.png)

*separated L2 cache*

![](Untitled%203.png)

There is also **L3 cache** which is generally there to improve the
performance of L1 and L2 cache being slower than either but still
generally twice as fast as its DRAM counterpart.

**Multiple CPUs**

You can also introduce multiple CPUs into the layout where each CPU has
its own embedded cache and the two CPUs have some means of communicating
data amongst one another.

![](Untitled%204.png)

In this instance the CPUs both have an embedded L2 cache and a common
separated L3 cache which they can both access.

------------------------------------------------------------------------

Non-Uniform Memory Access
-------------------------

💡

NUMA is a **computer memory design** used in multiprocessing where
memory access time depend on the memory location relative to the
processor.

### Multi-core and Multi-CPU Systems

Modern High Performance Computing systems often have multiple CPUs each
having their own number of cores, one way to abstractly represent this
design is as follows

![](Untitled%205.png)

**Embedded memory controller**

A memory controller, which controls the flow of data coming to and from
the main memory can either be separate as in the diagram above, or
integrated onto the CPU. When a memory controller is integrated onto the
CPU its generally called an **integrated memory controller (IMC)**

![](Untitled%206.png)

### Example - CPU with IMC and embedded L3 cache

The Xeon processor lineup from intel which is generally used in high
performance computing has a few CPUs which follow the above design. An
example of this would be the **Intel Core i7-5960X**

**Processor Die Map**

We can see here that it has an integrated memory controller and an
embedded shared L3 cache between all of the cores.

![](Untitled%207.png)

------------------------------------------------------------------------

Vector processor and instructions
---------------------------------

💡

A **vector processor** is a CPU that implements **vector instructions**
which are designed to operate efficiently and effectively on large
one-dimentional arrays of data called **vectors**.

### Advanced Vector Extension (AVX)

AVX are **Single Instruction Multiple Data** extensions to the x86
instruction set architecture. Meaning they implement vector instructions
for compatible vector processors of the x86 architecture such as the
Xeon Phi x200 which first implement AVX-512bit. The 512bit number just
refers to the size of the vector on which the vector instruction can
preform some given operation.

### Array addition vector instructions

A basic example of a vector instruction is just the direct addition of
two arrays.

**Adding two 32bit int arrays**

![](Untitled%208.png)

**Adding two 64bit double arrays**

![](Untitled%209.png)

In both cases we can see how over time the size of the vectors you could
preform parallel operations on has increased.

### Performance improvements of AVX

We can demonstrate the speedup that comes with processors that can use
AVX as follows. First we define the performance of a processor by the
expression

$$\text{cores} + \text{speed} + \text{instructions~per~cycle} + \text{flops~per~instruction}$$cores+speed+instructions per cycle+flops per instruction

This allows us to compute the number of **flops** or **floating point
operations per second**

**Flop results**

![](Untitled%2010.png)

![](Untitled%2011.png)

There are two main observations that can be made here

-   Intuitively I think AVX outperforms scalar flops due to the fact
    that its just computing more within the same timespan in a
    concurrent manner.

 

-   With a higher core count you are limited by scaling, that is, the
    clock speed does not scale with the core count. In the case of
    perfect scaling we retain the same clock speed as for just a single
    active core but realistically this clock speed drops.

------------------------------------------------------------------------

Data Layout
-----------

💡

The layout of data in memory is something important to consider in High
Performance computing. Its generally faster to access **contiguous
memory** because they can be **loaded into the cache** more efficiently
than non-contiguous memory.

### Array Layout - Fortran

One very important basic thing to note for data layout is that when
looping over arrays you want to loop over them in a contiguous manner.
This means also being aware how arrays are layed out in memory which
differs for different languages.

**C/C++ - 2D arrays**

In C and C++ data arrays, especially 2D arrays have a row layout, so to
write better performance code you would want to loop over these arrays
in a row like fashion.

![](Untitled%2012.png)

**Fortran - 2D arrays**

In Fortran on the other hand arrays have a column layout, so here you
want to loop over the arrays by columns.

We can demonstrate the speedup difference using some basic Fortran code
that just loops over a 2D array by row (non-contiguous) vs by column
(contiguous)

![](Untitled%2013.png)

### Fortran row vs column iteration

**Row iteration**

`a1.f90`

``` code
program array
    real,dimension(10000,10000) :: a
    integer :: i,j
    do i=1,10000
        do j=1,10000
            a(i,j) = i*7 + j*3
        end do
    end do
end program array
```

compile and time command

``` code
gfortran -O2 -o a1 a1.f90 ; time ./a1
```

execution time

``` code
real 0m1.827s
user 0m1.639s
sys 0m0.162s
```

Here we are looping over the array using the traditional row like
approach and we have almost 2s of total runtime after our Fortran code
is compiled.

**Column iteration**

`a2.f90`

``` code
program array
    real,dimension(10000,10000) :: a
    integer :: i,j
    do j=1,10000
        do i=1,10000
            a(i,j) = i*7 + j*3
        end do
    end do
end program array
```

compile and time command

compile and time command

``` code
gfortran -O2 -o a1 a1.f90 ; time ./a1
```

execution time

``` code
real 0m1.827s
user 0m1.639s
sys 0m0.162s
```

Here all we did was swap the indices in which we are looping over the 2D
array and we saw a substantial speedup in our execution time. This being
due to the aforementioned fact that we are now looping over the array in
a contiguous fashion which makes it easier to load into the cache.

### Effect of different compilers

Compilers can sometimes understand when you are preform an equivalent
slower operation. Like the row iteration in Fortran and then optimize
this to use column iteration instead. Which means sometimes you don’t
even have to actively think about data alignment, the compiler just
replaces what are doing taking this into account.

**Using** `ifort` **instead of** `gfortran`

compilation commands

``` code
ifort -O2 -o a1 a1.f90 ; time ./a1
ifort -O2 -o a2 a2.f90 ; time ./a2
```

results from `a1.f90`

``` code
real 0m0.184s
user 0m0.028s
sys 0m0.157s
```

results from `a2.f90`

``` code
real 0m0.175s
user 0m0.037s
sys 0m0.139s
```

We can see that even though we are using different iteration approaches
where one should be slower, the compiler `ifort` in this case is able to
spot the specific pattern of iteration we are doing and replace it with
the equivalent optimized pattern which means in the end we have
basically the same runtime.

### Data alignment

How data is aligned is of particular relevance with SIMD instructions
because if its not aligned to the size of an SIMD register it can cause
certain issues with vector instructions. There are generally 3 different
situations of alignment with memory.

**No alignment**

Here memory is not aligned, that is, its not stored as some multiple of
a specific size, this is generally slower specifically with SIMD
instructions.

![](Untitled%2014.png)

**Full alignment**

Here the memory is perfectly aligned which is the optimal state for
performance.

![](Untitled%2015.png)

**Partial alignment**

One approach for getting at least some performance out of memory which
is not aligned is to **peel** the first few values of the array. Which
means we processes them using regular instructions and the rest of the
memory which is aligned can then still be processed concurrently with
the SIMD instruction.

![](Untitled%2016.png)

### Divergence

Divergence is another factor which is important in the runtime of HPC
code. In the context of HPC divergence refers to the situation where
different parallel threads or processes execute different branches of a
conditional statement which can lead to performance degradation,
synchronization problems or just a wrong result in the computation.

**Mitigating divergence**

So when writing high performance parallel code you generally want to
mitigate divergence. So its generally good to try and be aware of
potential places where diverge could occur and of mitigation methods
like using divergence optimizing compilers, and other means.

------------------------------------------------------------------------

Graphics Processing Unit (GPU)
------------------------------

💡

The GPU is of particular relevance in HPC because as opposed to the CPU
its **optimized for parallel tasks.** It has a lower main memory and
per-thread performance than a CPU though it compensates for this by
having a **higher thread-count.** It also has **more compute resources**
than a CPU.

### Bottleneck

One point of particular relevance with GPU computations it the potential
bottlenecks between transfer points such as the CPU ↔ GPU or GPU ↔
Memory. If there isn’t a balance maintained with the memory transfer
speeds and the computation speeds then one or the other point will lead
to a bottleneck. As an example we can compare the P100 and the A100 GPUs
from NVIDEA

### CPU ↔ GPU \| **Comparison P100 vs A100**

|                               |             |              |
|-------------------------------|-------------|--------------|
| PCIe                          | 32GB/s      | 64GB/s       |
| Gflops                        | 9300 Gflops | 19500 Gflops |
| floats transferred/s          | 8 billion   | 16 billion   |
| float op. / transferred float | 1100        | 1200         |
| flop / byte                   | \~300       | \~300        |

They key observation to make here is that we maintained the flop to byte
ratio, which means that the transfer speed increased at the same rate of
the speed of floating point operations, meaning in respect to the P100
the A100 does not introduce any bottlenecks.

### Memory↔ GPU \| **Comparison P100 vs A100**

|                               |             |              |
|-------------------------------|-------------|--------------|
| Memory                        | 732 GB/s    | 1555 GB/s    |
| Gflops                        | 9300 Gflops | 19500 Gflops |
| floats read /s                | 183 billion | 388 billion  |
| float op. / transferred float | 50          | 50           |

Similarly as above the key observation to make here is that with the
A100 we increase the memory throughput and the computation speed which
means we can still preform the same number of floating point operations
per transferred float but at a much higher rate without any bottleneck.

### Features of the P100

There are some core features that define the performance of the P100 GPU

**Streaming Multiprocessor (SM)**

These are the multiprocessors where the actual computation takes place,
some features of these are that

-   They consists of many individual **streaming processors (SP) /
    cores**

 

-   The P100 has many streaming multiprocessors

**Registers**

Registers are small storage areas which an SM can use for computations,
some notes on these in regards to the P100 are that

-   It has 32768 registers in total for 32 SMs

 

-   Per SM the P100 has 1024 registers

**Threads**

Each core can handle multiple threads, some points on this are

-   Each SM in the P100 can handle 32 threads

 

-   In total the P100 can handle 1024 threads

 

-   Each thread has access to 32 storage registers

**Precision**

This refers to the types of numbers the P100 can process, this being
specifically single precision (32bit) and double precision (64bit)
numbers.

**WARP**

WARP refers to a group of 32 threads being executed concurrently in an
SM, which is a characteristic of the SIMD instructions being used in the
GPUs.

**Fast memory**

This refers to the shared memory that is available to all cores in an
SM.

### Hiding Latency

As mentioned before a GPU can compensate for its lower memory and
per-thread performance by using parallelism to hide the resultant
latency of individual computations. It does this by overlapping
Independent calculations to hide the latency of any individual
calculations through the speedup gained from concurrency.

**Example**

![](Untitled%2017.png)

Here we have an example table which demonstrates the latency for two
types of instructions, along with their peak throughput instructions per
cycle and the necessary number of these instructions we would need to
concurrently execute to hide the latency of any individual instruction.

**Handling stalls**

Another way of hiding latency, specifically generated by stalls, which
is when a thread is waiting for something else to happen is simply
switch to another thread to continue any other computations and keep the
GPU busy. This process of utilizing all available threads is called
**thread-level parallelism**

------------------------------------------------------------------------

GPU Programming
---------------

💡

GPU programming is just the act of running a program which can make use
of big parallelism on a GPU. Two popular ways of doing this is either
with the use of **OpenACC** or **CUDA**.

### Example - OpenACC Program

OpenACC is similar to OpenMPI in that you enable parallelism through
compiler directives that identify areas which should be parallelized.
But it differentiates itself by the fact that you can target both the
CPU or GPU to execute this program.

**The actual program code**

For some example code lets use the algorithm which finds a numerical
solution for pi.

`cpi_openacc.c`

``` code
#include <stdio.h>

double getTime(void);

static long steps = 1000000000;

int main(int argc, const char *argv[]) {
    int i;
    double x;
    double pi;
    char *p;
    double step = 1.0 / (double)steps;
    double sum = 0.0;
    double start = getTime();

    #pragma acc parallel
    #pragma acc loop reduction(+:sum) private(x)
    for (i = 0; i < steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    double delta = getTime() - start;

    printf("PI = %.16g computed in %.4g seconds\n", pi, delta);

    return 0;
}
```

So here as you can see its quite similar in that we just annotate that
the loop should be parallelized and that we want to preform a reduction
operation on the sum variable along with keeping `x` private for all
spawned threads.

**Running the program**

To run the program we want to first load the necessary modules on our
HPC then compile it by linking with OpenACC and then we can run the
final executable.
