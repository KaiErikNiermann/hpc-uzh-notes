
Compiler suites
---------------

Compiler suites are a **collection of tools**, most prominently for
compiling your program, but also things more general development and
testing, including benchmarking.

### Examples

|                                 |                                                                                                                            |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Cray Compiler (`PrgEnv-cray`)   | High performance compiler suite mainly for C/C++/Fortran                                                                   |
| GNU Compiler (`PrgEnv-gnu`)     | General purpose compiler that supports various languages/architectures and OS’s                                            |
| Intel Compiler (`PrgEnv-intel`) | [Intel oneAPI DPC++/C++ compiler](https://en.wikipedia.org/wiki/Intel_C%2B%2B_Compiler) for Intel processor based systems. |
| Portland (`PrgEnv-pgi`)         | High performance compiler suite made for C and C++ , now part of Nvidia HPC SDK stack                                      |
| Clang (Apple and Cray)          | Compiler meant for C, C++, Objective-C, and Objective-C++                                                                  |

### Compiler wrappers
Compiler wrappers aim to **streamline compilation process** especially
for High Performance Applications which would otherwise require a
specific combination of flags and steps to get the desired executable.

These wrappers mainly add the necessary flags and automatically link the
required libraries (`mpi.h` for example).

### Example - MPI Compilation
In the case of MPI you use the `mpi.h` header file to be able to access
the necessary functions. In the standard case you would need to tell the
compiler where to find the library to which the header file is
referring. Which would look something like this.

``` code
gcc -o my_mpi_program my_mpi_program.c -I/path/to/mpi/include -L/path/to/mpi/lib -lmpi
```

**that’s annoying**
So instead you have compiler wrappers, in this case we have `mpicc` for
C code or `mpic++` for C++ code. This mainly just tells the compiler
where everything is so you can simplify the compilation process down to
just.

``` code
mpicc -o my_mpi_program my_mpi_program.c
```

------------------------------------------------------------------------

Application : N-Body physics
----------------------------
N-Body simulation is a simulation of a dynamical system (a system that
changes over time) of interactions between particles, usually under the
influence of some force, in our case gravity.

### Relevance

This serves as a nice demonstration for how we might use benchmarking
techniques to improve the speed of our simulation.

### Some background

Each particle obviously must have some position in 3D space (x, y, z),
lets represent this as a vector

$\mathbf{x}_{i}$xi​ for a particle

$i$i.

We want an equation to represent the particles acceleration, this being
the change in velocity over time. We can express this numerically as the
sum of all gravitational forces acting on the particle. Since we are
working with vector we take the vector form for newtons law of gravity.

$$\mathbf{F}_{21} = - G\frac{m_{1}m_{2}}{\mid\mathbf{x}_{21}\mid^{2}}{\hat{\mathbf{x}}}_{21}$$F21​=−G∣x21​∣2m1​m2​​x^21​

We assume unit mass for particles (

$m_{i} = 1$mi​=1) furthermore we assume the gravitational force is
constant, so it has no relative effects. Also since we are working in 3
dimensions we cube the denominator.

$$\mathbf{F}_{21} = \frac{\mathbf{x}_{2} - \mathbf{x}_{1}}{\mid\mid\mathbf{x}_{2} - \mathbf{x}_{1}\mid\mid^{3}}$$F21​=∣∣x2​−x1​∣∣3x2​−x1​​

And since this only represents the gravitational force between object 1
and 2 we have to take the sum of forces acting on the body, namely

$$\mathbf{x}_{i}^{\prime\prime} = \sum\limits_{j = 1,i \neq j}^{n}\frac{\mathbf{x}_{j} - \mathbf{x}_{i}}{\mid\mid\mathbf{x}_{j} - \mathbf{x}_{i}\mid\mid^{3}}$$

Now we have an equation to model the change in the particles velocity
over time, from this we can start building our little simulation.

### The code

**Initial requirements**

```cpp
#include <stdio.h>
#include <vector>
#include <random>

typedef struct {
    float x, y, z;      // position
    float vx, vy, vz;   // velocity
    float ax, ay, az;   // acceleration
} particle;

typedef std::vector<particle> particle_list;

void init_cond(particle_list &plist, int n) {
    std::random_device rd;  // Seed for random number generator
    std::mt19937 gen(rd()); // Mersenne Twister generator
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    plist.clear();
    plist.reserve(n);
    for (auto i = 0; i < n; ++i) {
        particle p {
            dis(gen), dis(gen), dis(gen),
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        };
        plist.push_back(p);
    }
}

int main(int argc, char** argv) {
    int n = 20000;
    particle_list plist;
    init_cond(plist, n);
    forces(plist);
    return 0;
}
```

Should be mostly intuitive what’s going on here, we have a vector of
particle structs that we are just initializing to random starting
positions.

**The force calculations**

``` code
void forces(particle_list &plist) {
    for (auto &p1 : plist) {
        p1.ax = p1.ay = p1.az = 0.0;  // starting with 0 acceleration
        for (auto &p2 : plist) {
            if (&p1 == &p2) continue; // skip self
            auto dx = p2.x - p1.x;
            auto dy = p2.y - p1.y;
            auto dz = p2.z - p1.z;
            auto dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            auto inv_dist3 = 1.0 / (dist * dist * dist);
            
            // Update acceleration
            p1.ax += dx * inv_dist3;
            p1.ay += dy * inv_dist3;
            p1.az += dz * inv_dist3;
        }
    }
}
```

Should again be mostly straightforward here, we are looping over all
particles and updating their position relative to that of the other
particles in the processes skipping self.

### Compilation and basic benchmarking

So now for the reason you might be reading this, for basic benchmarking
lets just see what happens with linux’s builtin `time` utilitly.

`Commands`

``` code
g++ -o nbody nbody.cpp
time ./nbody
```

`output`

``` code
real    0m9.327s
user    0m9.316s
sys     0m0.011s
```

Quite simple method of benchmarking, lets see if we can improve the
speed with basic use of flags.

`Commands`

``` code
g++ -o -O3 -ffast-math -o nbody nbody.cpp
time ./nbody
```

`output`

``` code
real    0m8.466s
user    0m8.465s
sys     0m0.001s
```

Ok, not much better, but its somewhat of an improvement.

**Debugging**

Something of note is that certain compiler flags might lead to a loss in
information when debugging your program, of use here is the `-g` flag
which adds information that can be used when debugging your program. So
sometimes its wise to add the flag in there, especially at higher
optimization levels.

**Minimizing file size**

If you would want to decrease the file size by removing certain parts of
the compile executable not necessary to the execution you can use the
`strip` utility which removes symbols (function names / global variables
names / static variables names)

Likewise this also removes any debug information added by the `-g` flag,
so if you want to remove this information without having to recompile
its a handy thing to be aware of.

------------------------------------------------------------------------

Benchmarking with Craypat
-------------------------

💡

Craypat is a **performance analysis tool** for HPC systems, to use it
you have to execute your program as a job using the queue system and
load the `perfotools-lite` module

### Using Craypat

**perftools-lite module**

This is essentially just an easy to use version of Craypat, so lets load
it using `module load` , additionally you might also have to load `cray`
as this might not be loaded yet so run the following command

``` code
module load cray perftools-lite
```

Now a key thing, you want to use the clang compiler `CC` **AFTER** you
have loaded perf-tools to generate the executable, because once CrayPAT
is loaded any subsequent compilation sprinkles in all the nice
performance analysis files that attach to your code. So now do

``` code
CC -g -ffast-math -o nbody nbody.cpp
```

This should give you an output something like

`output`

``` code
WARNING: PerfTools is saving object files from a temporary directory into directory
'/users/<uname>/.craypat/nbody/18328'

INFO: creating the PerfTools-instrumented executable 'nbody' (lite-samples) ...OK
```

⚠️ **Warning** the optimization flags strip away certain symbols which
can sometimes limit the extent to which Craypat can analyze the runtime
of specific functions. So if you compile with these flags it might lead
to less information about the runtime of your program. So always
optimize with `-g` and just be aware of this fact.

**the job script**

Nothing to special about this, since for now its just a serial program
that executes quickly we can select some basic partition with a low
time-limit.

`nbodyscript`

``` code
#!/bin/bash -l
#SBATCH --account=my_account
#SBATCH --job-name=nbody
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=mc

srun nbody
```

You then run the script using the usual `sbatch` and if you wanna check
on it remember `sinfo -p <partition>`

``` code
sbatch nbodyscript
```

### The output

After the jobscripts complete if you `ls` you should see a new directory
popup called something like `nbody+130946-8639969s` which contains a
collection of files detailing profiling information about your file. The
easiest way to view everything is to use the `pat_report` command

``` code
pat_report nbody+130946-8639969s
```

This dumps all the information into your terminal, a summary of the key
tables

| Table Number | Table Name                                    | Description                                                                                                                |
|--------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Table 1      | Profile by Function                           | Shows functions and line numbers with significant exclusive sample hits, averaged across ranks.                            |
| Table 2      | Profile by Group, Function, and Line          | Displays sample information for functions and lines within the 'USER' group, including source file and line details.       |
| Table 3      | Program HW Performance Counter Data           | Provides hardware performance counter data for the entire program, averaged across ranks or threads.                       |
| Table 4      | Program Energy and Power Usage (from Cray PM) | Shows energy and power usage for nodes, including maximum, mean, and minimum values, as well as the sum over all nodes.    |
| Table 5      | Memory High Water Mark by Numa Node           | Presents the total size of all pages mapped into physical memory for each Numa Node, captured near the end of the program. |
| Table 6      | Memory Bandwidth by Numanode                  | Displays memory traffic and bandwidth information for each Numa Node, highlighting the top and bottom 3 nodes.             |
| Table 7      | Wall Clock Time, Memory High Water Mark       | Shows total wall clock time for ranks, maximum memory usage, and average usage, providing an overview of program time.     |

### Sampling

Sampling is one of the main methods in which a programs performance is
analyzed. During sampling experiments/runs the call stack of your
program is sampled at a specified interval which allows you to see for
example which function was most frequently called.

A common thing you might want to do is see the actual code segments
which are taking up alot of the runtime. Since in table 2 you get a
handy breakdown of the most performance impacting lines you can take
note of them, for example

``` code
...
4|||   3.6% |  30.0 |   -- |    -- | line.21
4|||  37.7% | 313.0 |   -- |    -- | line.22
4|||  22.4% | 186.0 |   -- |    -- | line.25
4|||   4.9% |  41.0 |   -- |    -- | line.26
...
```

we can see the lines from 21 to roughly 26 where the most performance
impacting

**The** `sed` **command**

You can use the sed command to isolate the lines of code which are
specified to be the most performance impacting. So lets do this, lets
isolate the lines 21 → 26 in our `nbody.cpp`

`command`

``` code
cat -n nbody.cpp | sed -n 21,26p
```

This command pipes ( `|` ) the output to the `sed` command where we then
isolate 21 → 26, which gives us the following

`output`

``` code
21              auto dist = std::sqrt(dx*dx + dy*dy + dz*dz);
22              auto inv_dist3 = 1.0 / (dist * dist * dist);
23
24              // Update acceleration
25              p1.ax += dx * inv_dist3;
26              p1.ay += dy * inv_dist3;
```

**Observations**

From this we can clearly see that most of the runtime comes from
computing certain parts of the velocity equation of the particles.

### app2

This uses the same information generated by craypat but instead of
dumping it at you it interprets it and gives you a nice visual overview
of everything.

**Usage**

``` code
app2 nbody+130946-8639969s
```

------------------------------------------------------------------------

Self timing
-----------

💡

One of the most basic ways of timing specific segments of code. The idea
being to measure the start and end time then look at the difference.

### General concept

We need some kind of function to get the current time and then call it
before and after our code segment.

``` code
before = time()

// your code 

after = time()

elapsed = after - before 
```

**Time functions**

Here's a markdown table summarizing the information about different
timing functions:

| Function          | Description                                                          | Header                  |
|-------------------|----------------------------------------------------------------------|-------------------------|
| `time()`          | Get time in seconds since the Epoch (1970-01-01 00:00:00 +0000 UTC). | `#include <time.h>`     |
| `gettimeofday()`  | Get time and timezone information.                                   | `#include <sys/time.h>` |
| `settimeofday()`  | Set time and timezone information.                                   | `#include <sys/time.h>` |
| `clock_gettime()` | Retrieve time of the specified clock.                                | `#include <time.h>`     |
| `clock_getres()`  | Find the resolution (precision) of the specified clock.              | `#include <time.h>`     |

**Usage**

`test.c`

``` code
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

int main(int argc, char** argv) {
    struct timeval tv2;
    struct timeval tv3;

    printf("time() = %ld\n", time(NULL));
    
    gettimeofday(&tv2, NULL);
    printf("gettimeofday() = %ld\n", 
            tv2.tv_sec);

    // settimeofday 
    tv3.tv_sec = 0;
    tv3.tv_usec = 0;
    settimeofday(&tv3, NULL);
    printf("settimeofday() = %ld\n", 
            tv3.tv_sec);

    // clock_gettime
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    printf("clock_gettime() = %ld\n", 
            ts.tv_sec);

    // clock_getres
    struct timespec ts2;
    clock_getres(CLOCK_REALTIME, &ts2);
    printf("clock_getres() = %ld\n", 
            ts2.tv_sec);
    return 0;
}
```

**Compilation**

``` code
gcc -o out test.c -lrt ; ./out
```

Of note here is that sometimes stuff is undefined with time related
things so you might have to link `-lrt` for everything to compile.

`output`

``` code
time() = 1700326897
gettimeofday() = 1700326897
settimeofday() = 0
clock_gettime() = 1700326897
clock_getres() = 0
```

The output is mostly obvious, the 3 time functions get the time,
settimeofday as you see you can set the time, this might be helpful when
you want to control certain aspects about time.

**Resolution ?**

Then `clock_getres()` function measures the resolution. This being the
smallest measurable time unit for your given clock.

### High resolution (precise) C++ clock - Measuring fast stuff

If you need to measure something either very precisely or very fast then
you want to be using a clock which can measure very small timesteps or
**ticks.**

**Basic example**

``` code
#include <chrono>
#include <iostream>

int main() {
        // Defining a namespace and alias to simplify code
    using namespace std::chrono;
    using clock = high_resolution_clock;

    double tick = 1.0 * clock::period::num / clock::period::den;
    auto t1 = clock::now();
    
    std::cout << "timer res = " << tick << "s\n"; 
    
    auto t2 = clock::now();
    auto dt = duration_cast<duration<double>>(t2 - t1).count(); 
    std::cout << "dt = " << dt << "s\n"; // elapsed time as double
    return 0;
}
```

`output`

``` code
timer res = 1e-09s
dt = 0.0009293s
```

------------------------------------------------------------------------

Advanced - cycle.h
------------------

💡

There is an implementation of self timing that measures the exact
difference in the number of ticks that passed. The header is defined
here

[](https://github.com/NetApp/SS-CDC/blob/master/cycle.h)

https://github.com/NetApp/SS-CDC/blob/master/cycle.h

### Usage

You include the `cycle.h` header file and then use the `getticks()`
function

`test.c`

``` code
#include <stdio.h>
#include "./cycle.h"

int main(int argc, char** argv) {
    ticks t1 = getticks();
    ticks t2 = getticks();

    printf("elapsed : %f\n", elapsed(t2, t1));
    return 0;
}
```

`output`

``` code
elapsed : 22.000000
```

So the difference between t2 and t1 is 22 ticks

------------------------------------------------------------------------
