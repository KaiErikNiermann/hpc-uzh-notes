Moore’s law and hardware constraints
------------------------------------

Moores law states that the **number of transistors** that can be placed
on an **integrated circuit** at a reasonable cost **doubles** ever **two
years**.

### Energy consumption and cost

**Power consumption**

Dissipated electric power scales to cube of clock frequency. So as clock
speeds increase, the power consumption increases significantly.

**Cooling**

Dissipated power per square centimeter is limited by cooling
capabilities. So as electronic components become smaller and more
densely packed, managing heat becomes more challenging.

### Constraints

**Processor frequency plateau**

Processor frequency reached a plateau in the early 2000s. Higher clock
speeds have becoming increasingly hard to achieve without generating too
much heat.

**Modern Moore’s law**

Moore’s law has remained true, though increase in processing power has
increased to other methods.

**Alternative approaches**

Increasing the number of cores has been one approach instead of focusing
just on increasing clock-speed.

------------------------------------------------------------------------

The memory wall
---------------
The memory wall describes the **implications** of the **processor/memory
performance** gap that has grown steadily over the last several decades.

### Prerequisite - Von Neumann Architecture
Architecture which describes the basic abstract structure of most modern
computers.

**Core aspects**
Programming instructions and data are stored in the **same** memory unit
and **share memory**.

During execution a programs **instruction** and **data** are loaded
**into Central Processing Unit**

Result of computation stored back in **memory unit**

### Problem *-* Von Neumann Bottleneck

These bottlenecks refer to a set of limitations arising from the
specific structure of the Von Neumann architecture.

**Memory bandwidth**
The rate at which data can be moved to and from the CPU is not
increasing as quickly as computing power of CPU. In other words, the CPU
can **processes data faster** than it can be **supplied** from memory.

**Memory latency**
The time it takes the CPU to access data from memory has been decreasing
**slowly**.

**Increasing cores**
The increasing number of cores per memory unit increasing exacerbates
the memory bottle neck as we have more cores competing for memory
access.

### Consequences - CPU Wait Cycles
One major consequence is that the CPU having to wait for incoming data
leads to wasted processing cycles where the CPU **could have been doing
something**. Reducing overall efficiency.

### Solutions to the Memory Wall
The main goal of solutions to the memory wall is to increase memory
throughput and thus make better use of otherwise wasted CPU cycles where
its done processing and just waiting for new data from memory.

**Cache memory**
This refers to small amounts of memory typically stored closer to the
CPU containing frequent information. This reduces the amount of times
the CPU needs to wait for data from the main memory thus leader to
better efficiency.

**Parallel Memory access**
This refers to the processor making use of things like **vector
architectures** or **advanced vector extensions** to allow for parallel
memory access which in turn also increases throughput.

------------------------------------------------------------------------

Amdahls Law
-----------
A law or expression which denotes the **theoretical maximum speedup**
obtained by parallelizing some code for a given problem with a fixed
size.

### Equation

$$\text{Speedup}(N) = \frac{T_{S}}{T_{p}(N)} = \frac{T_{S}}{\alpha T_{S} + (1 - \alpha)\frac{T_{S}}{N}} = \frac{1}{\alpha + \frac{1 - \alpha}{N}}$$

**Variables**

$T_{S}$TS​ - Execution time of the serial code

$T_{p}$Tp​ - Execution time of the parallel code

$\alpha$α - Fraction of the code which is not parallel

$N$N - Number of processors

### Observation

As the number of processors

$N$N approaches infinity the value of the speedup for the piece of code
approaches the value

$\frac{1}{\alpha}$α1​. This can formally be expressed as

$$\underset{N\rightarrow\infty}{\lim}\left\lbrack \text{Speedup}(N) \right\rbrack = \frac{1}{\alpha}$$N→∞lim​\[Speedup(N)\]=α1​

**Reasoning**

The

$\frac{1 - \alpha}{N}$N1−α​ term approaches 0 due to the denominator

$N$N becoming much larger than the numerator

$1 - \alpha$ 

### Application

**Example - Serial and Parallel part**

> *Problem statement* : We have a program where **30% of the execution
> time may be subject to speedup**. Furthermore the improvement makes
> the affected part of the program **twice as fast**.

30% of the execution time being subject to speedup is equivalent as
saying 30% of the code is parallelizable. This in turn means that 70%
denotes the fraction of code we *cannot* parallelize. Thus

$\alpha = 0.7$α=0.7.

Our improvement making the effected part twice as fast simply means that
we are using 2 processors, or that

$N = 2$N=2.

Therefore to figure out the speedup under these conditions we have that.

$$\text{S(N)} = \frac{1}{0.7 + \frac{0.3}{2}} \approx 1.18$$S(N)=0.7+20.3​1​≈1.18

**Example - Many Parallel parts**

> *Problem statement* : We have a serial task split into **4 consecutive
> parts** each part has the following execution times, p1 = 0.11, p2 =
> 0.18, p3 = 0.23, and p4 = 0.48. Since all portions are subject to
> speedup we must divide all by the respective speedup
>
> $N$N for each part, which is s1 = 1, s2 = 5, s3 = 20, s5 = 1.6.

This gives us the the following expression for speedup

$$\text{S}(N) = \frac{1}{\frac{0.11}{1} + \frac{0.18}{5} + \frac{0.23}{20} + \frac{0.48}{1.6}} \approx 2.19$$S(N)=10.11​+50.18​+200.23​+1.60.48​1​≈2.19

**Example - Serial program with multiple parts**

> *Problem statement :* We have a serial program with two parts
>
> $A$A and
>
> $B$B for which
>
> $T_{A} = 3\text{s},T_{B} = 1\text{s}$TA​=3s,TB​=1s  
> What are the speedups if part  
>
> $B$B is made to run 5 times faster and
>
> $A$A 2 times faster ?  
> If we can only speedup one part which is the better choice ?  

Firstly to calculate the two different speedups for

$B$B with have

$N = 5$N=5 and for

$A$A we have

$N = 2$N=2, and since

$B$B represents 25% of the runtime and

$A$A 75% we have that.

$$S_{B}(5) = \frac{1}{0.25 + \frac{0.75}{5}} = 1.25$$SB​(5)=0.25+50.75​1​=1.25

$$S_{A}(2) = \frac{1}{0.75 + \frac{0.25}{2}} = 1.60$$SA​(2)=0.75+20.25​1​=1.60

To compare the **percentage improvement** we can apply the formula

$$\text{\%~improvement} = 100\left( 1\frac{1}{S(N)} \right)$$% improvement=100(1S(N)1​)

Which means if we choose to speedup part

$A$A by two the % improvement is 37.5% whereas speeding up

$B$B by 5 gives us a % improvement of 20%.

From this we can conclude that it would clearly be better to speed up

$A$A twice.

------------------------------------------------------------------------

Gustafson Law
-------------

💡

The **theoretical maximum speedup** obtained by parallelizing some code
for a problem of a constant **size per core**.

Alternatively this law also presents the slowdown if you were to run the
program on just a single core. It differs to Ahmdal’s law in that it
focuses on how we can utilize scale (e.g. more processors) to
parallelize the workload.

### Equation

$$\text{Speedup}(N) = \frac{T_{S}}{T_{P}(N)} = \alpha + N(1 - \alpha)$$Speedup(N)=TP​(N)TS​​=α+N(1−α)

### Observation

$$\underset{N\rightarrow\infty}{\lim}\lbrack\text{S}(N)\rbrack = N(1 - \alpha)$$N→∞lim​\[S(N)\]=N(1−α)

The key thing to note here is that as opposed to Ahmdals law which
suggests that the speedup is limited by the non-parallel portion

$\alpha$α with Gustafson’s law we don’t converge instead we just scale
more slowly.

That is,

$\alpha$α is only a scale factor for a given

$N$N as opposed to being a converging value.

![](Introduction%20to%20Parallel%20Programming%202e17062638cc48b18f60a94cbf6239f5/Untitled.png)

------------------------------------------------------------------------

Hardware Evolution
------------------

💡

There are certain key trends in hardware evolution that have occurred
over the past few decades.

### Technical trends

**Computing power doubling Annually**

This is to an extent a reflection of Moore’s law which described the
exponential growth on transistors on IC’s and the subsequent impact on
computing power.

**Massively Parallel and Many-Cores Architecture**

As discussed previously due to the fact that the clock speed of
processors has hit a plateau for the most part due to heat constraints
there has been a shift to focus on multi-core and parallel architectures
as a means of achieving speedup.

**GPUs as key players in HPC**

GPUs have become increasingly popular in HPC as they can accelerate a
wide range of computational tasks due to being highly parallel
processors.

**Increasing hardware complexity**

Over time hardware has become more complex, with things like hybrid
systems which combined different types of processors and multiple cache
layers.

**Memory per core stagnating or decreasing**

This trend suggests that despite the increasing core count due to the
clock speed plateau the amount of memory we can supply to said cores is
slowing down or even decreasing due to the aforementioned memory wall
which limits the memory throughput.

**Performance per core stagnating**

The performance per core is slowly down, which indicates the importance
of parallelization to fully utilize available resources.

**Disk Input/Output Bandwidth increasing slowly**

The speed at which data can be read and written from/to disk is not
increasing rapidly. This being a prominent issue with the memory wall.

------------------------------------------------------------------------

Evolution of Supercomputers
---------------------------

💡

Over time the architecture of supercomputers has evolved to utilize
different kinds of hardware.

### Evolution of architectures

![](Introduction%20to%20Parallel%20Programming%202e17062638cc48b18f60a94cbf6239f5/Untitled%201.png)

### Different supercomputer eras

![](Introduction%20to%20Parallel%20Programming%202e17062638cc48b18f60a94cbf6239f5/Untitled%202.png)

------------------------------------------------------------------------

Communication Networks
----------------------

💡

Within the context of HPC you often need to **connect a vast number of
processors** for the parallel computation of large-scale tasks. This
requires **high-performance networking infrastructure** to facilitate
efficient communication and data exchange among processors.

### Basic requirements

**Switches**

To connect nodes in an HPC cluster you need **network switches** which
act as a central hub for data exchange.

**Fiber cables**

Fiber cables are used to connect each node to the switch, ensuring fast
and reliable data transfer.

### Layout

**Fat-Tree network topology**

The networks also need to follow some sort of layout, this is where
fat-tree network topology comes in. This topology is designed to
minimize bottlenecks and efficiently handle communication needs of
large-scale computing clusters.

### Protocols

After having the basic requirements and layout there also needs to be
some kind of communication protocol that is followed, an example of this
is

**Infiniband**

High-speed, low-latency data transfer technology and communication
protocol. With the specifications :

-   latency = 0.5 → 1 microsecond

 

-   bandwidth = 5 → 40 GB/sec

These specifications make Infiniband a popular choice due to the
exceptional data transfer rate and low latency

### Configurations and Parameters

The network also generally has certain configuration to maximize the key
parameters **latency** and **bandwidth**

**Blocking configuration**

In a non-blocking network every node can communicate with every other
node simultaneously, whereas in a blocking configuration there might be
limitations on concurrent communication paths.

**Network parameters**

The two key parameters that matter in a network are latency and
bandwidth. Low latency means fast data transfer and high bandwidth means
high volume of data transfer.

------------------------------------------------------------------------

Current state of HPC for Users and Developers
---------------------------------------------

💡

The current state of HPC which is a result of things like **Moore’s
law**, **Memory wall**, and **Clock speed plateau** has a multitude of
effects on users and developers of HPC systems.

### Users

### Developers

**Clock speed plateau**

Its necessary to exploit many (relatively slow) cores

Performance of individual cores not really increasing anymore

**Memory wall**

The memory per core is constant or even decreasing ; CPUs are memory
limited

More complex architectures to overcome current limitations

**Parallelism**

High level of parallelism is needed but there are I/O bottlenecks

Multi-disciplinary approach is required leading to concept of co-design

------------------------------------------------------------------------

Current state of HPC programming
--------------------------------

💡

The different aspects of HPC evolution have also led to certain
programming languages and paradigms to be the current focus nowadays.

### MPI

The Message Passing Interface programming model which is widely used for
parallel programming in a cluster is still the dominant programming
technique nowadays despite its age.

### OpenMP and MPI approaches

A hybrid approach of using the thread based OpenMP and process based MPI
model is the most effective way to program on modern HPC systems.

### GPU message passing

This technique of message passing within the GPU as opposed to the CPU
is a technique that is gaining popularity.

### New parallel programming languages

There are various novel parallel programming languages being
developed/worked on. There are also new runtime systems being developed
to handle task-based parallelism such as Charm++ and HPX.

-   Co-arrary
-   Fortran
-   PGAS
-   X10
-   Chapel
