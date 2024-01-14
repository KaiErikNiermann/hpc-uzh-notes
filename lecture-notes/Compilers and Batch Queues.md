
------------------------------------------------------------------------

Modules
-------

Environment modules are encapsulated **settings** and **software**
needed for a certain application.

### common actions

`module list` - Listing all currently loaded modules

`module switch x y` - Switching a module x with another y

`module avail` - Listing all available modules

``` code
module avail gcc // lists available gcc versions 
```

`module load x` - Loading a module `x`

``` code
module load PrgEnv-gnu/8.3.3 // 
```

------------------------------------------------------------------------

Batch Queue System
------------------

### Important data-structures
**Queue** - First In, First Out
**Stack** - Last In, First Out
**Priority Queue** - Queue in which elements are ordered by priority ( e.g. size )

### Supercomputer terminology
**Node** - Individual computer that consists of one or more CPUs
together with memory
**Core** - Individual part of CPU, can have multiple threads
**CPU** - Compute unit, usually has several cores

### Job Script

Two main parts

1.  Job metadata
    ``` code
    #SBATCH --account=uzh8        // Relevant slurm account that is running the job
    #SBATCH --job-name=hpc_test   // Name for the job that you can see when you run it 
    ```
    
2.  Job parameters
    ``` code
    #SBATCH --time=01:00:00       // Max alloted time to run the program
    #SBATCH --nodes=1             // Number of nodes to run process
    #SBATCH --ntasks-per-core=1   // Number of tasks per core
    #SBATCH --ntasks-per-node=36  // Number of processes 
    #SBATCH --cpus-per-task=1     // Number of threads 
    #SBATCH --partition=normal    // Parition to run the program on 
    #SBATCH --constraint=mc       // Constraint assigned my administrator

    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    srun ./program
    ```

### Slurm commands

`sbatch` - Submit a job
`squeue` - See all currently running jobs
-   `-t` pending jobs
-   `-r` running jobs
-   `-u` your jobs
`sinfo` - See partition information
-   `-p` normal partition
-   `-d` debug partition
`srun` - Run a job
`scancel` - Cancel a job
`accounting` - view usage

------------------------------------------------------------------------

MPI and OpenMP
--------------
**M**essage **P**assing **I**nterface is a standard designed for
parallel computing of multiple systems with distributed memory where
processes pass messages between one another.  **O**pen  **M**essage **P**assing is designed for shared memory (single) systems with multiple cores based on the idea of thread/core sharing

### Key differences
**MPI**
MPI works primarily in terms of processes, which run on *different*
memory spaces

**OpenMP**
OpenMP works primarily with threads, which run on *shared* memory spaces
### The basic MPI program

``` code
#include <mpi.h> 

int main() {
    MPI_Init(&argc, &argv); 
    
    // Run parallel code 

    MPI_Finalize(): 

    return 0;
}
```

Example of basic MPI program structure in C

### Common MPI functions

`MPI_Init(&argc, &argv)`

-   **Initializes** MPI environment
-   Must **always** be called and be **first**
-   Can be used to **pass command line arguments**

`MPI_Finalize()`

-   **Terminates** MPI environment
-   **Last** MPI function call

**Communicators**

Type of MPI object which define the processes that can talk.
All communicators have a size property.

**Rank**

Defines the process and has the bounds 0≤ Rank < Size , in other
words, its the ID of the process

`MPI_COMM_WORLD`
-   Communicator object predefined as **all** of the MPI processes
`MPI_Comm_rank(comm, &rank)`
-   Communicator function that returns the rank of the calling process
    as the `rank` variable within the communicator `comm`
`MPI_Comm_size(comm, &size)`
-   Returns the total number of processes within the communicator `comm`
    as the variable `size`

**Example application**
``` code
int my_rank, size; 
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
```
Programm that reads the rank and size of the world comm. object.  
Size would in this case return the number of processes and  
`my_rank` the process id.

### MPI Hello World example
``` code
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);                // initialize MPI library
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // get number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // get my process id
    
    // do something
        
        // Code runs independently on each process    
    printf("Hello World from rank %d\n", rank);
    if (rank == 0) printf("MPI World size = %d processes\n", size);

    MPI_Finalize();  // MPI cleanup
    return 0;
}
```

**Output**

![](Compilers%20and%20Batch%20Queues%20e98f74423bfe46a38d4340195a1b6872/Untitled%201.png)

***Observations***
-   The code ran on each process independently
-   MPI processes have **private variables**
-   Processes *can* be on **different machines**

------------------------------------------------------------------------

Example MPI and OpenMP program
------------------------------
We can demonstrate the basic concept of **speedup** using an example
program that uses numerical integrated to approximate pi.

### MPI Version
[MPI Version](Compilers%20and%20Batch%20Queues%20e98f74423bfe46a38d4340195a1b6872/MPI%20Version%203b2263f9f1c6479c8dd317dbf19cbe94.md)

``` code
cc -O3 -o cpi_mpi cpi_mpi.c
```

### OpenMP Version
[OpenMP Version](Compilers%20and%20Batch%20Queues%20e98f74423bfe46a38d4340195a1b6872/OpenMP%20Version%20d4a53f1180e84e3ab67244cbac3c86fe.md)

``` code
cc -O0 -o cpi_omp -fopenmp cpi_omp.c
```

### Job Script

``` code
#!/bin/bash -l
#SBATCH --job-name="cpi_imp"
#SBATCH --account="uzh8"
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=mc

export OMP_NUM_THREADS
    =$SLURM_CPUS_PER_TASK

srun ./cpi_mpi
```

**Notes**

We are using 36 **processes**

``` code
#!/bin/bash -l
#SBATCH --job-name="cpi_openmp"
#SBATCH --account="uzh8"
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --partition=debug
#SBATCH --constraint=mc

export OMP_NUM_THREADS
    =$SLURM_CPUS_PER_TASK

srun ./cpi_omp 
```

**Notes**

We are using 36 **threads**

### Timing and Speedup plots

💡

We can use **GNU plot** to plot the runtime and speedup using a
different number of threads to run the program

**Speedup formula**

$${\text{speedup~}\%} = \frac{\text{time~with~1~thread}}{\text{time~with~}n\text{~threads}}$$speedup %=time with n threadstime with 1 thread​

**Plotting timings**

``` code
set title 'OpenMP Timing'
set xlabel 'Threads’
det ylabel 'Time (seconds)'
set key top right
plot "cpi_openmp.dat" u 1:2 w lp
lw 2 t 'OpenMP'
```

![](Compilers%20and%20Batch%20Queues%20e98f74423bfe46a38d4340195a1b6872/Untitled%202.png)

**Plotting speedup**

``` code
set title 'OpenMP Speedup'
set xlabel 'Threads'
set ylabel 'Speedup'
set key top left
plot x lc 2 lw 2 t 'Ideal Speedup’,\
"cpi_openmp.dat" u 1:(4.55/$2)\
w lp lc 1 lw 2 t 'OpenMP'
```

![](Compilers%20and%20Batch%20Queues%20e98f74423bfe46a38d4340195a1b6872/Untitled%203.png)

------------------------------------------------------------------------

Compiler basics
---------------
A compiler is a piece of software which converts a piece of source code
(e.g. a C file) into object code.

### Compilation basics

![](Compilers%20and%20Batch%20Queues%20e98f74423bfe46a38d4340195a1b6872/Untitled%204.png)

1.  The *source code* is compiled into *object code (machine language)*
2.  A *Linker* is used to “link” additional libraries which contain
    object code with your existing program.
3.  The result of this process becomes a runtime version of your program
    which contains *executable code.*

### Compiler optimizations
The compiler can employ various means of optimizing your code, it does
this automatically but you can also specify the extent to which your
code will be optimized using flags.

**Table of optimization levels**

|       |                                                                                                                                                                                            |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-O0` | **No optimization** occurs, same as not using the command at all.                                                                                                                          |
| `-O1` | Somewhat increased compilation time and performance. Increase memory usage, that is, larger final code.                                                                                    |
| `-O2` | No *loop unrolling* or *function inlining.* Increases compilation time but also performance of code.                                                                                       |
| `-O3` | Turns on all optimizations specified by -O2 and also turns on the `-finline-functions`, `-funswitch-loops`, `-fpredictive-commoning`, `-fgcse-after-reload` and `-ftree-vectorize` options |

### Compile commands

**Basic compile, no optimization**

`gcc -o cpi cpi.c ; ./cpi`

No optimization, simply creating and running the executable `cpi`

**Basic optimization**

`gcc -O1 -o cpi cpi.c ; ./cpi`

This is using the *optimize* flag, `1` is the lowest level and `3` is
the highest level.

**Advanced optimization**

`gcc -O3 -ffast-math -maxv2 cpi cpi.c ; ./cpi`

You can also apply additional optimization flags such as `-ffast-math`
and `-mavx2` , which both use various techniques to increase the runtime
of your code.

❗**Critical Note**

Compiler optimization flags like `-ffast-math` while they can increase
the performance by reordering expressions this process in turn can also
lead to a different result on occasions.

### Compiling multiple files
If we have 3 files in the same folder: `cpi.c` , `gettime.c` and the
header file `gettime.h`

`cpi.c`
``` code
#include <stdio.h>
#include "gettime.h"

...
```

`gettime.h`
``` code
double getTime(void);
```

`gettime.c`
``` code
#include <stdio.h>
#include <sys/time.h>


double getTime(void) {
    // function body
}
```

**Compiling together**

`gcc -o cpi cpi.c gettime.c`

**Compiling and linking separately**

Object files for `cpi.c`

`gcc -c -o cpi.o cpi.c`

This produces the file `cpi.o`

Object files for `getTime.c`

`gcc -c -o gettime.o gettime.c`

This produces the file `gettime.o`

*Linking together*

`gcc –o cpi cpi.o gettime.o`

This then yields us the executable `cpi` which we can execute

### **Equivalent Makefile for separate compiling and linking**

``` code
cpi : cpi.o gettime.o
    gcc -o cpi cpi.o gettime.o

cpi.o : cpi.c gettime.h
    gcc -O3 -ffast-math -mavx2 -c -o cpi.o cpi.c

gettime.o : gettime.c gettime.h
    gcc -O3 -ffast-math -mavx2 -c -o gettime.o gettime.c

clean:
    rm -f cpi cpi.o gettime.o
```

**Improvement 1 - Default rules**
We can use default rules which Make employs to reduce the amount of
repeated code

``` code
cpi : cpi.o gettime.o

cpi.o : cpi.c gettime.h

gettime.o: gettime.c gettime.h

clean:
    rm -f cpi cpi.o gettime.o
```

*Commands executed*

`cc -c -o cpi.o cpi.c`
`cc -c -o gettime.o gettime.c`
`cc cpi.o gettime.o -o cpi`

**Improvement 2 - Customizing default rules**
We can specify custom properties of the default rules to bring back user
defined flags for things like optimization

``` code
CFLAGS=-O3 -ffast-math -mavx2
CC=gcc

cpi : cpi.o gettime.o

cpi.o : cpi.c gettime.h

gettime.o: gettime.c gettime.h

clean:
    rm -f cpi cpi.o gettime.o
```

------------------------------------------------------------------------

Message Passing
---------------

### Reasons for using MPI

**Memory limitations**

Good to use when you encounter **memory limitations** of a single
computing node, as here we can work in a *distributed* fashion.

**Calculation time**

As MPI exists to parallelize a workload through the use of processes it
can **improve the runtime** of your program through the distribution of
work.

**Been around for a long time**

MPI has **been around** for a considerable amount of time and is still
often in use thus making it important to learn and use.

### Breaking down CPI program using MPI

**Initialization**

Initializes the program and gets the total number of processes and their
respective ranks

``` code
MPI_Init(&argc, &argv); /* Connect processes to each other */
MPI_Comm_size(MPI_COMM_WORLD, &numprocs); /* Get total number of processes */
MPI_Comm_rank(MPI_COMM_WORLD, &myid);     /* Rank of this process */
MPI_Get_processor_name(name, &resultlen);
```

![](Compilers%20and%20Batch%20Queues%20e98f74423bfe46a38d4340195a1b6872/Untitled%205.png)

**Barrier**

Makes sure that all processes reach a certain point before continuing
execution

``` code
MPI_Barrier(MPI_COMM_WORLD);
```

**Broadcast**

Conditional operation occurring only in process 0

``` code
if (myid == 0) {
    printf("This program uses %d processes\n", numprocs);
    n = 1000000000;
}
```

Broadcasts the value `n` of type `MPI_INT` (integer) from process `0` to
all other processes denoted by `MPI_COMM_WORLD`

``` code
MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

**Reduce**

Here we do the main approximation of pi and then combine it

``` code
sum = 0.0;
h = 1.0 / n;
for (i = myid + 0.5; i < n; i += numprocs) {
    sum += dx_arctan(i * h);
}
mypi = 4.0 * h * sum;
```

We then add single (`1`) partial sums `mypi` from all procs
`MPI_COMM_WORLD` to each other into `pi` using the sum ( `MPI_SUM` )
operation.

``` code
MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
```

**Finalize**

``` code
MPI_Finalize();
```

This finalizes the program.

------------------------------------------------------------------------

Applying OpenMP in CPI
----------------------
We can apply OpenMP for a similar effect as in MPI to parallelize the
workload. The equivalent C program is broken down as follows.

Instead of having the **Initialization, Barrier** and **Broadcast** step
we can just create a parallel region for the for loop, the same thing we
parallelized via message passing using MPI.

### **Reduce**

Here we use two key features of OpenMP, first of all

**Loop parallelism -** `#pragma omp parallel for`

This is a preprocessor directive that informs the compiler that we wish
to parallelize the subsequent loop.

**Reduction clause -** `reduction(+ : sum)`

Here instead of `MPI_SUM` we specify the reduction operation using the
operator `+` and the variable on which the reduction occurs in parallel
is `sum`

``` code
#pragma omp parallel for reduction(+ : sum)
    for (i=0; i < steps; i++) {
        double x = (i+0.5)*step;
        sum += 4.0 / (1.0+x*x);
    } 
```

### Using the correct Makefile

``` code
# Added the linker flag `-fopenmp` 
# to say we are using OpenMP now
CFLAGS=-Wall -O3 -ffast-math -mavx2 -fopenmp
LDFLAGS=-fopenmp
CC=gcc

cpi : cpi.o gettime.o

cpi.o : cpi.c gettime.h

gettime.o : gettime.c gettime.h

clean:
    rm -f cpi cpi.o gettime.o
```

Unlike with MPI which needs no alterations in the compilation process,
for **OpenMP** we do need to change how we compile our program.  
  
Using the same custom  
*default rules* as before we simply need to add the correct linker flag
so that we tell the linker to include the appropriate libraries so that
our code executes properly.

------------------------------------------------------------------------

Parallel Pitfalls
-----------------

### Problem - Race condition

A race condition is a condition of a program where its behavior depends
on relative timing or interleaving of multiple threads or processes.

**Why is this bad ?**

In the case of threads for example they share access to the same memory
by default, thus the way in which they modify a shared variable will
depend on the thread scheduling algorithm. Which in turn means that the
end result is not necessarily consistent across multiple runs of your
program.

**Solution - Locking**

You can lock the data the moment it is accessed to create a consistent
result in how the threads modify the data.

### Problem - Deadlocks

An issue that arises in which a certain piece of data is permanently
locked due to an issue in how a locking algorithm functions.

**Cause of a deadlock**

1.  Process A currently holds some resource `K` and then tries to get
    some resource `M` currently held by Process B but has to wait until
    B finishes

 

1.  Process B, currently holding some resource `M` wants to get access
    to resource `K` so it waits until `A` is finished

*Circular Dependency* : Since both processes are holding on to a
resource, and they both want each others resources they get locked in a
permanent wait cycle, a “deadlock”.

**Requirements for a deadlock**

Obviously a deadlock can only occur if threads behave under certain
conditions

*Mutual exclusion*

Threads cant share resources : A and B cant share the resource, they are
both holding onto it as its their own

*Hold and Wait*

Threads most hold a resource while waiting for another : A and B cant
give resource away because they both want each-others resource

*No Pre-emption*

Held resources cannot be given away : A and B cannot just give resource
away  
  

*Circular Wait*

The wait dependency must be circular : A wants B resources and B wants A
resource

**Solution - No Mutual exclusion**

In some cases we can have multiple threads share the same data (e.g.
read only file systems)

**Solution - Pre-emption**

Automatically release all currently held resource if current one not
available

**Solution - No Hold and Wait**

We can ensure a process only requests another resources if its not
holding on to any existing resource.

**Solution - No Circular wait**

We can ensure this is the case by imposing a total ordering of all
resources such that a resource can never request something lower than
itself.

------------------------------------------------------------------------
