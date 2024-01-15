
Introduction and History
------------------------
In the mid 90s many vendors where using their own methods of
multithreading but they then adopted OpenMP.

### History

|             |                                                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------------------------------|
| mid-90s     | Many vendors are using their own multi-threading directives such as CRAY, NEC and IBM during the era of vector processing |
| October ‘97 | Many vendors met and adopted Open Multi Processing standard                                                               |
| late-90s    | OpenMP specified by Architecture Review Board                                                                             |
| 2000 - now  | OpenMP received numerous updates and new technologies                                                                     |

------------------------------------------------------------------------

General concepts of OpenMP
--------------------------
Before we get into using some OpenMP there are two key concepts,
**multi-threading** and **parallel regions** that we should talk about.

### Multi-threading

OpenMP applies the concept of multithreading in a specific manner, for a
given program using OpenMP we have that

**creation of threads**
Program is executed by **one process**, called the **master thread.**

Master thread **activates light-weight processes,** called
**workers/slave threads** at the entry of a **parallel region.**

**variables and tasks**
Each thread executes a **task**, corresponding to a **block of
instructions.**

During the execution of a task, variables can be **read from** or
**updated** in **memory.**

**private variable**
A variable which is defined in the **local memory** of a thread

**shared variable**
A variable which is defined in the main **shared memory (RAM)**
### Regions

**parallel region**
Region of program executed by **several threads** in parallel (i.e.
master and slave threads).

**sequential region**
Region of program executed by only **master thread** 0.

------------------------------------------------------------------------

Work sharing
------------
OpenMP has 3 main ways of sharing/distributing work that it employs as a
part of its programming model, this being: **dividing loop iterations,
dividing code sections** and, **executing multiple instances of the same
code**

### Dividing loop iterations
As the name implies this involves dividing the loop iterations amongst
some number of threads. Ensuring that each thread has an equal amount of
work to do.

### Dividing sections of code
Again as the name implies this is simply the process of divvying up
sections of code amongst the threads, ensuring that each thread executes
a different segment of code.

### Executing multiple instances of the same procedure, one per thread
This means that each thread has its own copy of the procedure and will
execute it independently of other threads.

------------------------------------------------------------------------

Threads and Cores in OpenMP
---------------------------
OpenMP has a specific way of dealing with threads. The OMP scheduler
**maps threads onto cores** depending on a number of factors like their
**load, the amount of work,** and **hints provided.**

### Best case mapping
The OMP schedule distributes threads evenly among cores to in turn have
an even distribution of the workload thus efficiently parallelizing the
task.

### Worse case mapping
The OMP scheduler maps all threads to a single core, this leads to a
poor distribution of the workload and thus inefficient parallelism.

### Mapping in practice

**Thread migration**
Threads can be **migrated across cores** by OS . Which can result in a
certain overhead.

**Thread rank pinning**
The runtime can also pin threads to cores according to their rank which
can **improve performance** by reducing **overhead of thread
migration**.

------------------------------------------------------------------------

OpenMP program structure and compilation
----------------------------------------
OpenMP programs generally follow a certain structure and also have to be
compiled in a specific manner.

### Structure / Main attributes

**Compilation directives and clauses**
> ❗Seen as *comment lines* unless program is compiled properly

Allow for things like the
-   Creation of threads
-   Defining of work and data sharing strategies
-   Synchronizing shared variables

**Functions and routines**
OpenMP has certain library functions for various purposes that must be
linked at link time using `-openmp` or `-fopenmp` in compile command.

**Environment variables**
OpenMP has several environment variables that can be set at execution
time that change the parallel computing behavior of OMP. An example of
this is setting the number of threads it uses

``` code
export OMP_NUM_THREADS=4
```

### Compilation
As mentioned before a compiler sees directives and clauses as just
comments unless we link to OpenMP when we compile, this can be done by
either adding `-openmp` (icc compiler) or `-fopenmp` (GNU compilers e.g.
gcc, g++) flag to our compilation command.

`test.c`
``` cpp
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
    #pragma omp parallel
    {
        printf("Hello World!\n");
    }
    return 0;
} 
```

**Compilation command and setting threads**

``` code
export OMP_NUM_THREADS=4
gcc -fopenmp test.c -o test
```

`output`
``` code
Hello World!
Hello World!
Hello World!
Hello World!
```

------------------------------------------------------------------------

OpenMP vs MPI
-------------
While both of these models enable parallel programming they each have
some key differences.

### OpenMP

### MPI

**Multithreaded Model**
Multithreaded model that operates with **single process** (master thread
and worker threads). Where programmers can define **regions of code**
that are **executed across multiple threads**

**Multi-Process Model**
Involves the execution of **multiple independently running processes**
which can run on **separate nodes.**

**Implicit communication**
Designed for **shared memory architecture,** where multiple threads can
access the **same memory space**. Programmer does **not** need to manage
communication, hence its **implicit**.

**Explicit communication**
As processes are running independently and generally do not run on a
shared memory space communication has to be **explicitly** defined by
the programmer. Which makes MPI useful for distributed memory
architectures like clusters with InfiniBand networks.

**Usage**
Used mainly in **shared-memory, multi-core** architectures. Well suited
for tasks that can be parallelized within a **single node**.

**Usage**
MPI is commonly used in HPC environments where data needs to be
exchanged **between nodes in a cluster**. Ideal for applications that
have **high inter-process communications**.

**Benefits**
Easy to parallelize code via OMP directives.

**Benefits**
provide **fine-grained control over communication between processes**.
Suitable for **complex** and **data-intensive** applications that rely
on **message passing**.

------------------------------------------------------------------------

OpenMP Directives
-----------------
OpenMP directives are a key part of how OpenMP is used to implement
parallelism within your program.

### Core concepts

**Threads during runtime**
On entry the master thread activates a team of **children threads**. On
exit these threads disappear or hibernate.

**How directives work ( fork-join model )**
When the executing code hits a directive the operating system creates
**parallel regions** and spawns **multiple threads**.

**Spawning threads is expensive**
Spawning threads comes with overhead, so their application needs to be
carefully considered.

### Directive Syntax
``` code
#pragma omp directive-name clause ... clause new-line
```

`#pragma omp`
Each directive starts with `#pragma omp`

`directive-name`
You then specify the specific name of the directive, in the case of
wanting to parallelize a region of code you would just use the
`parallel` directive.

`clause, ..., clause`
You then specify an arbitrary number for utilizing specific parts of
OpenMP, like `for` to parallelize loops, or `default(none)` to tell the
compiler to treat all variables in a parallel region as having undefined
data-sharing attributes unless explicitly specified in another clause.

Clauses can also obviously be chained together.

------------------------------------------------------------------------

Parallel Region - OpenMP application
------------------------------------
**Reminder** - Creating a parallel region means telling OpenMP that you
want to parallelize a specific region of code.

### Creating a basic parallel region

**Notes**

Region is in the statement after `parallel` in `{}`

All threads execute the **same code** in parallel

Variables are **shared by default**

Implicit **barrier** after region

``` code
int main(int argc, char** argv) {
    // serial region
        #pragma omp parallel
    {
                // parallel region
        printf("Hello World!\n");
    }
        // serial region 
    return 0;
}
```

*We specified OMP should use 4 threads, hence the region executes 4
times giving us 4 outputs.*

**Compilation command and setting threads**

``` code
export OMP_NUM_THREADS=4
gcc -fopenmp test.c -o test
```

`output`

``` code
Hello World!
Hello World!
Hello World!
Hello World!
```

### Advanced parallel region

A valid question to ask would be what specifically is counted as a
parallel region. The scope of the parallel region is bounded by two
types of extents.

**Static extent**

This includes things like the line directly below the directive or the
segment of code you enclose with braces.

``` code
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    float a = 10.10;
    #pragma omp parallel 
    {
        a = a + 20;
        printf("Hello World %f\n", a);
    }
    return 0;
}
```

**Dynamic extent**

This includes any code generated by for example calling a method.  
  

``` code
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

void add(float a) {
    a = a + 20;
    printf("Hello World %f\n", a);
}

int main(int argc, char** argv) {
    float a = 10.10;
    #pragma omp parallel
    add(a);
    
    return 0;
}
```

**Note - Passing variables to subroutines**

Variables in subroutines are private if passed by value but shared if
passed by reference, that is, a variable acts like a private variable if
you pass a copy of it to a function, but it acts like a shared variable
if you only pass the memory location. As in the latters case the
variable does not leave the thread local scope.

### Different thread choices

**Notes**

We can make some observations if choose different threads and then look
at the result of the `omp_in_parallel();` `omp.h` library function.

`test.c`

``` code
// _OPENMP only true ifOP_NUM_THREADS > 1
#ifdef _OPENMP
    #include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    float a = 10.10;
    bool p = false; 
    #pragma omp parallel 
    {
        #ifdef _OPENMP
                        // if _OPENMP true then p true
            p = omp_in_parallel();
        #endif
        printf("Hello World! %d\n", p);
    }
    return 0;
}
```

**Compilation command used**

``` code
export OMP_NUM_THREADS=1
gcc -fopenmp test.c -o test 
```

`output`

``` code
Hello World! 0
```

**Observation**

Since we only chose 1 thread clearly there is no parallelism occurring.
This in turn means we are not going to include `omp.h` which then means
the directives will be seen as comments and `p` will be the default
value of `false` .

### Dynamic Forking - controllig threads

If you want a more fine grained control over the number of threads in a
specific parallel region this can be specified by using the
`num_threads(n)` clause where `n` is the number of threads you want for
the corresponding parallel region.

`test.c`

``` code
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    float a = 10.10;
    #pragma omp parallel num_threads(3)
    {
        a = a + 20 ;
        printf("Hello World %f\n", a);
    }
    return 0;
}
```

`output`

``` code
Hello World 30.100000
Hello World 30.100000
Hello World 50.099998
```

**Result**

Since specified we want 3 threads to execute the parallel region we end
up getting 3 total outputs.

If you want to know the number of threads in a parallel region you can
use `omp_get_num_threads()`

------------------------------------------------------------------------

Controlling variables
---------------------

💡

As we said before variables are by **default shared** though you might
not want this to always be the case, in which case you can manually set
this.

For the following examples I use the same compilation command

``` code
export OMP_NUM_THREADS=4
gcc -fopenmp test.c -o test
```

### The default behavior

`test.c`

``` code
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    float a = 10.10;
    #pragma omp parallel 
    {
        a = a + 20;
        printf("Hello World %f\n", a);
    }
    return 0;
}
```

`output`

``` code
Hello World 70.099998
Hello World 50.099998
Hello World 50.099998
Hello World 30.100000
```

**Result**
As `a` is by default shared between all threads it initially has a value
of `10.10` then; in an order decided by the thread scheduler; the
variable `a` is incremented by the threads.

**Key observation**
We can see here that within two threads the value of `a` is the same,
this is due to the fact that one thread incremented as it should, but
another thread also tried incrementing but due to OpenMPs deadlock
prevention mechanism it evidently was not given access to

### Manual specification - thread local private

`test.c`
``` cpp
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    float a = 10.10;
    #pragma omp parallel \
        default(none) private(a)
    {
        a = a + 20;
        printf("Hello World %d\n", a);
    }
    return 0;
}
```

`output`

``` cpp
Hello World 20.000000
Hello World 20.000000
Hello World 20.000000
Hello World 20.000000
```

**Result**

As we are declaring `a` private here it means that each thread has its
own instance in its **thread local memory** of the `a` variable. The
variable might be uninitialized but in this instance is 0 thus adding
`20` gives 20 in each thread.

**Key observations**

The moment we add the `default(none)` clause it strips the default
(`shared`) classification of all variables, thus for any variable
modified within the parallelized region ( the code coming after
`#pragma omp parallel` ) we **have to** specify if its either shared (
`shared(a)` ) or private ( `private(a)` ) as an added clause.

Unless we manually initialized the variable stored in the thread local
memory it will be by default uninitialized memory which could lead to
undefined behavior.

### Manual specification - inheriting global state into thread local

An alternative to having to initialize private thread-local variables is
to initialize them in the master thread before we enter a parallel
region. We can do this by using the `firstprivate(a)` which

1.  Initializes the value of a variable used in a parallel region to the
    its value in the global region
2.  Makes the variable private, same as the `private(a)` clause

`test.c`

``` cpp
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    float a = 10.10;
    #pragma omp parallel \
        default(none) firstprivate(a)
    {
        a = a + 20;
        printf("Hello World %f\n", a);
    }
    return 0;
}
```

`output`

``` code
Hello World 30.100000
Hello World 30.100000
Hello World 30.100000
Hello World 30.100000
```

**Result**

Unsurprisingly the result here is `30.1` we initialize all the thread
local variables to `10.10`. Then as all variables are private, that is,
they are all in their own thread local memory, adding 20 leads to the
same value on all threads.

------------------------------------------------------------------------

Work Distribution using OpenMP
------------------------------
The obvious main benefit of OpenMP is to use multithreading as a means
of parallelizing / distributing work, which can be done primarily
through something like loops, which OpenMP has directives for
parallelizing.

### Properties of loop parallelism

**What is a parallel loop ?**

A parallel loop is just a `for` loop in which the work of each iteration
is parallelized amongst some number of threads. So we are distributing
the work occurring across the threads.

**OpenMP parallel loops**

They are specified using the `for` directive, that is

``` code
#pragma omp parallel for 
...
```

OpenMP only supports loops with **loop indices**, while loops and for
loops without indices (i.e. for-each loops) are not supported.

The distribution strategy is set by a **schedule** clause. Its important
to have proper scheduling for optimizing the **load balancing** of work.

**Loop indices** are always **private integer variables** in parallel
loops.

You can **nest multiple** `for` **directives** inside a parallel region
to parallelized multiple loops concurrently.

**Synchronization**

The default behavior of OpenMP is that it includes a **global
synchronization at the end** of a parallelize loop. The `nowait` clause
removes this global synchronization but you cannot use this in the case
of a parallel loop.

### Basic parallel loop

**Expanded version**
This is the basic anatomy of a parallel loop, we first declare a
parallel region, since we obviously want the loop to occur in a parallel
context, then we use the `for` directive to tell OpenMP that we want to
parallelize the loop iterations.

``` cpp
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    #pragma omp parallel 
    {
        #pragma omp for 
        for (int i = 0; i < 8; i++) {
            printf("Hello from thread %d\n", 
                            omp_get_thread_num());
        }
    }
    return 0;
}
```

**Condensed version**
This is the exact same as the version above but just using a more
condensed syntax.

``` cpp
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    #pragma omp parallel for 
    for (int i = 0; i < 8; i++) {
        printf("T_id %d, index %d\n", 
                    omp_get_thread_num(), i);
    }
    return 0;
}
```

❗**Observation**
Starting a parallel region is **expensive** so its often good to have
one parallel region containing multiple parallelized **for** directives.

Just something to remember when combining directives, you are still
doing two things, starting a parallel region and then parallelizing a
loop.

`Output`

``` code
T_id 1, index 2
T_id 1, index 3
T_id 3, index 6
T_id 3, index 7
T_id 2, index 4
T_id 2, index 5
T_id 0, index 0
T_id 0, index 1
```

Here we can see the loop iterations; 8 in total; have the different
iterations handled by the 4 different threads, where the work of the
iterations is evenly distributed with each thread handling 2 iterations.

We can also note that the indexes are obviously not handled in a
sequential manner, just in whichever manner the thread scheduler
schedules the threads.

------------------------------------------------------------------------

Work scheduling
---------------
This is a way of manually deciding how OpenMP divides up the work of the
different loop iterations, this is done using the `schedule()` directive
by the programmer, which takes various arguments denoting different
approaches of scheduling work.

Note, I’m using the same demo for loop as above, just for the sake of
brevity thought it would be simpler to just show the directive as
opposed to copy pasting boilerplate.

### Using `schedule(static, n)`
Using this directive we are telling OpenMP that we want to divide the
loop into exactly `n` separate chunks. Where the chunks are assigned to
the threads in a cyclic manner (round-robin algorithm).

`test.c`
``` cpp
#pragma omp parallel for schedule(static, 4)
```

`output`
``` code
Thread 1: 4
Thread 1: 5
Thread 1: 6
Thread 1: 7
Thread 0: 0
Thread 0: 1
Thread 0: 2
Thread 0: 3
```

**Results**

Somewhat expected here we can see that we are dividing `8` iterations
into chunks of `4` , this will generate 2 chunks of work, which in turn
will be processed by two threads assuming we have allocated at least 2.

If we allocate more than 2 threads only 2 will be utilized in this
example.

### Using `schedule(runtime)`

Using this means we want to decide how to schedule / distributed the
work of the iterations among the threads **at runtime**. And how we do
this distribution at runtime depends on the **problem size** or on the
**simulation parameters**.

The **optimal strategy** is a balance between the size of each
task-larger tasks mean overhead has less of an influence-, and the
granularity of the tasks-the smaller the better because of improved load
balancing-.

`test.c`

``` code
#pragma omp parallel for schedule(runtime)
```

`output`

``` code
Thread 0: 0
Thread 0: 4
Thread 0: 5
Thread 0: 6
Thread 0: 7
Thread 3: 3
Thread 2: 2
Thread 1: 1
```

**Results**

Here a single thread handles the bulk of the iterations, this being due
to the fact that it probably leads to less overhead since the problem
size is somewhat equivalent I think ?

### Using `schedule(dynamic, n)`

We want to divide the loop into `n` chunks of work but we assign these
chunks to threads when they are available. That is, each thread is
assigned a chunk of 4 iterations and when it finishes it requests
another chunk from a shared queue.

`test.c`

``` code
#pragma omp parallel for schedule(dynamic, 4)
```

`output`

``` code
Thread 0: 0
Thread 0: 1
Thread 0: 2
Thread 0: 3
Thread 2: 4
Thread 2: 5
Thread 2: 6
Thread 2: 7
```

**Results**

So we can see that in this case it acts pretty similar to just using the
static scheduling

### Using `schedule(guided, n)`

Similar to `dynamic` but instead of the chunks all being of equal size
they start of large but then decrease as the size of the iterations are
distributed among threads. Useful when the amount of work per iteration
is **not uniform**.

`test.c`

``` code
#pragma omp parallel for schedule(guided, 4)
```

`output`

``` code
Thread 0: 0
Thread 0: 1
Thread 0: 2
Thread 0: 3
Thread 2: 4
Thread 2: 5
Thread 2: 6
Thread 2: 7
```

**Results**

In our instance the work is uniform so its not the best example but for
the sake of clarity I just kept it like this. Just imagine if the work
was non-uniform.

------------------------------------------------------------------------

Reduction operation
-------------------
I think rather intuitively breaking up a problem often times literally
manifests into creating partial results to a problem and them combining
these results, this is what the `reduce()` directive is meant for.

### **Basic syntax**

``` code
#pragma omp (parallel) for reduction(operator : a_1, ..., a_n)
```

`operator`

This can refer to any operator including some functions so

-   Arithmetic : `+`, `*`, `-`

 

-   Logical/Boolean : `&`, `|`, `^`, `&&`, `||`

 

-   Predefined : `max`, `min`

 

-   User defined

`a_1, ..., a_n`

The different variables you wish to separately reduce applying the
chosen operator.

### Basic example

`test.c`
``` cpp
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum) 
    for (int i = 0; i < 8; i++) {
        sum += 1;
        printf("thread %d , sum %d\n", 
                    omp_get_thread_num(), sum);
    }
    printf("sum: %d\n", sum);
    return 0;
}
```

`output`

``` code
thread 3 , sum 1
thread 3 , sum 2
thread 1 , sum 1
thread 1 , sum 2
thread 0 , sum 1
thread 0 , sum 2
thread 2 , sum 1
thread 2 , sum 2
sum: 8
```

**Results**

We can observe that each thread after its executed both of its
responsible iterations it has a partial sum, that collectively sum to 8

------------------------------------------------------------------------

Controlling variables in parallel loops
---------------------------------------
A key thing to understand when dividing up work using parallel loops is
if you want the loops to work together on something (i.e. shared
variables ) or if you want them to work in an isolated setting on data (
i.e. private variables ).

### `firstprivate(a)`

Same as in the case of the normal parallel region. The variable adopts
any value it was initially assigned prior to the parallel loop and then
becomes private, so any further modifications **within** the loop do not
effect the variable in any other loop iteration.

### `lastprivate(a)`

Here the variable is declared as private for each chunk of work (loop
iterations) assigned to a thread. But, the value that `a` adopts in the
**last iteration** is the one that `a` will have after the parallel
portion of the code. To demonstrate what this means we can use the
following example

**Example**

Here we can see that at the last iteration `i = 7` the value of sum was
`6` , therefor this value is passed on to the outer shared variable.

The reason why sum is 6 is because remember we start with `sum` being
undefined, which often times makes it default to `9` , then for the work
allocated to thread `3` that being 2 iterations, we increment it both
times by the rank of this just being 3, hence is 6.

`test.c`

``` cpp
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    int sum = 0;
    #pragma omp parallel for lastprivate(sum)
    for (int i = 0; i < 8; i++) {
        sum += omp_get_thread_num();
        printf("thread %d , sum %d, i %d\n", 
            omp_get_thread_num(), sum, i);
    }
    printf("sum: %d\n", sum);
    return 0;
}
```

`output`

``` code
thread 0 , sum 0, i 0
thread 0 , sum 0, i 1
thread 1 , sum 1, i 2
thread 1 , sum 2, i 3
thread 3 , sum 3, i 6
thread 3 , sum 6, i 7
thread 2 , sum 2, i 4
thread 2 , sum 4, i 5
sum: 6
```

**Note -** Remember that the value at the last iteration just comes down
to how loop iterations are assigned to threads.

### `nowait` directive

This **cannot** be used at the end of a parallel for section because you
are leaving a parallel section.

------------------------------------------------------------------------

Exclusive execution
-------------------
Somewhat similar to how `parallel for` leads to chunks of loop
iterations being distributed amongst threads you can manually designate
chunks of code to likewise be distributed amongst threads using
different directives, leading to the concept of **exclusive execution**.

### Syntax

**Forcing parallel execution**

`sections` directive

Denotes the following region of code contains **section** blocks which
themselves are smaller regions of code to be distributed amongst the
available threads.

`section` directive

Denotes some amount of work ( lines of code ) to be executed on some
thread. And belongs to a larger group of sections.

**Forcing serial execution**

`single` directive

Expresses that you want to execute a code block only on a **single
thread**.

`master` directive

Expresses that you want to execute a code on the **master thread**.

### Broadcasting
Sometimes you might be doing some modifications to a variable in a
single thread but after this you want all threads to have this updated
value. To achieve this you can use the `copyprivate(a)` clause to
broadcast `a` to all threads.

**Using** `copyprivate(a)`

`test.c`
``` cpp
#include <omp.h>
#include <stdio.h>

int main() {
    int a = 0, b = 0;

    #pragma omp parallel \
        private(a) private(b)
    {
        b = a;
        #pragma omp single copyprivate(a)
        {
            a = 1;
        }
        printf("Thread %d: a=%d, b=%d\n", 
            omp_get_thread_num(), a, b);
    }

    return 0;
}
```

`output`
``` code
Thread 1: a=1, b=0
Thread 3: a=1, b=0
Thread 0: a=1, b=0
Thread 2: a=1, b=0
```

**Not using** `copyprivate(a)`

`test.c`
``` code
#include <omp.h>
#include <stdio.h>

int main() {
    int a = 0, b = 0;

    #pragma omp parallel \
        private(a) private(b)
    {
        b = a;
        #pragma omp single 
        {
            a = 1;
        }
        printf("Thread %d: a=%d, b=%d\n", 
            omp_get_thread_num(), a, b);
    }

    return 0;
}
```

`output`
``` code
Thread 2: a=0, b=0
Thread 0: a=0, b=0
Thread 1: a=1, b=0
Thread 3: a=0, b=0
```

We can see here pretty clearly that when we use copy private the we end
up getting the same value as `a` in all threads, as opposed to the
default behavior where `a` is only modified in thread 1, so after the
single thread execution clearly its only changed in thread 1 due to
being private.

### Basic example
Similar to creating a parallel region in which we handle loops, we can
combine the `parallel` directive with the `sections` directive, to
denote that the parallel region will contain `section` ( be aware of the
singular here ) directives.

If we choose 2 threads (0 and 1) we get the following results.

`test.c`

``` cpp
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    int sum = 0;
    #pragma omp parallel sections 
    {
        #pragma omp section 
        {
            sum += 1;
            int t = omp_get_thread_num();
            printf("Thread %d: sum = %d\n", t, sum);
        }

        #pragma omp section
        {
            sum += 2;
            int t = omp_get_thread_num();
            printf("Thread %d: sum = %d\n", t, sum);
        }

        #pragma omp section
        {
            sum += 3;
            int t = omp_get_thread_num();
            printf("Thread %d: sum = %d\n", t, sum);
        }

        #pragma omp section
        {
            sum += 4;
            int t = omp_get_thread_num();
            printf("Thread %d: sum = %d\n", t, sum);
        }
    }
    return 0;
}
```

`output`

``` code
Thread 1: sum = 3
Thread 1: sum = 6
Thread 1: sum = 10
Thread 0: sum = 1
```

Here we have 4 sections

1.  Adding 1
2.  Adding 2
3.  Adding 3
4.  Adding 4

Thread 0 was assigned to execute (1)

Thread 1 was assigned to execute (2), (3) and (4)

------------------------------------------------------------------------

Synchronization
---------------
I’ve mentioned `nowait` a bit before but wanted to actually now
demonstrate what it means to not wait.

### General theory
Fundamentally synchronization is necessary to ensure all threads have
reached a given line in a program. In other words to ensure they have
all executed their denoted segment of work.

This is especially critical when we have multiple threads effecting the
same shared memory. Because intuitively we don’t want some thread to
proceed when the work is only half done.

Due to the generally critical nature that all threads should finish work
many OpenMP constructs have an implicit barrier. So that all threads in
a parallel region finish execution before proceeding.

**Removing synchronization**
This implicit barrier can often be removed by appending the `nowait`
clause.

**Forcing synchronization**
To force all threads within a parallel region to be synchronized we can
use the `barrier` directive

**Avoiding race conditions**
To avoid race conditions we want to use the `atomic` or `critical`
directive to force a serial variable ( all threads properly ) update

### **Example - Basic synchronization**
Here we have a simple counter program where we distributed loop
iterations of incrementing the counter amongst the threads. Furthermore
we add the `atomic` directive to avoid race conditions. In other words
to ensure that the counter is properly updated.

I think intuitively what we would want to have happen is that we want
all threads to update the counter, and only then do we want to print the
final value.

`test.c`
``` cpp
#include <omp.h>
#include <stdio.h>

int main() {
    int shared_counter = 0;

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < 5; i++) {
            #pragma omp atomic
            shared_counter++;
            printf("t = %d, i = %d, c = %d\n", 
                omp_get_thread_num(), i, shared_counter);
        }

        #pragma omp single
        {
            printf("Counter is %d\n", 
                shared_counter);
        }
    }

    return 0;
}
```

`output`

``` code
t = 1, i = 2, c = 1
Counter is 4
t = 3, i = 4, c = 4
t = 2, i = 3, c = 2
t = 0, i = 0, c = 3
t = 0, i = 1, c = 5
```

**Result**

Though as we can see here we seem to be printing the counter value
early.

The thing is that its still a for loop, so once we reach the last
iteration, if we don’t wait then whichever thread has finished the loop,
will just continue, and here this happened to be thread 3 and which c
was 4.

The two key reasons as to why not waiting here leads to us not getting
the right value for `c` . Is due to two fundamental properties of loop
parallelism. This being that

-   **Order** - iterations are not executed in order, they are treated
    as independent chunks, meaning we might execute things in order from
    i = 0 → i = 1 as we see on thread `0` but we can also observe that
    at this stage `c` is 5, so clearly later stages were executed
    scheduled earlier.

-   **Loop behavior** - in the context of a single thread things and
    using `nowait` we just have normal behavior, if the loop condition
    no longer holds execution continues, so if we then encounter a
    single block, clearly we just print whatever the value of `c` is on
    that thread.

### Example - Using critical directive
In principle this is similar to atomic but its more general usually
referring to a segment of code to be executed using locking as opposed
to atomic operations on single pieces of data like variables.

`test.c`
``` cpp
#include <omp.h>
#include <stdio.h>

int main() {
    int counter = 92290;
    #pragma omp parallel
    {
        #pragma omp critical
        {
            ++counter;
        }
    }
    printf("Final counter is %d\n", 
            counter);
    return 0;
} 
```

`output`
``` code
Final counter is 92294
```

**Reasoning**
Should be pretty clear at this point, we have 4 threads, each which
either wait for their turn to access the region then update the
variable, or just update the variable.

If we remove this we can see that we sometimes get false outputs like

``` code
Final counter is 92292
```

**False outputs**

Not using something like the critical region means that if a thread
wants to access a variable but this is currently being accessed by
another thread as opposed to waiting it often times just proceeds
execution to avoid something like a deadlock. But to force it to wait we
using things like `atomic` or `critical`

------------------------------------------------------------------------

Memory Issues
-------------
How memory is accessed is an aforementioned important part in parallel
program, so its critical to understand the importance of certain memory
access patterns.

### Memory Access Hierarchy
To avoid the memory bottle neck OpenMP programs utilize the memory
hierarchy. That is, they make use of the fact that certain memory is
kept close to the CPU to avoid costly accesses to data from the main
memory.

**Hierarchy : Fastest → Slowest**

1.  CPU registers, are the fastest form of memory and are directly on
    the CPU
2.  L1 cache, this is the second fastest, usually smaller than the lower
    cache levels
3.  L2/L3 cache, these are the third fastest, slower than L1 but still
    significantly faster than RAM
4.  RAM, this is the slowest form of memory access, aside from reading
    from the hard drive directly

### **Cache conflicts, misses and benefits**

**Conflicts**
Conflicts between threads can lead to poor cache memory management, this
in turn can result in Cache misses.

**Misses**
Cache misses refers to the situation when data needed by a thread is not
present, then the CPU must fetch it from a slower level of the memory
hierarchy.

Thus **minimizing cache misses** is essential for achieving good
performance in OpenMP programs, particularly for L1 and L2 cache memory
management.

Efficiently **mapping data in memory** is key to faster runtime in your
OpenMP program. This is especially key in Non-Uniform Memory Access
machines where main memory is distributed across different sockets and
local access is faster than remote.

### Performance analysis
There are various functions to benchmark OpenMP code, to simply measure
the elapsed time of a segment of code you can use the `omp_get_wtime()`
function.

------------------------------------------------------------------------

Summary
-------

OpenMP requires **multi-processor** machines with **shared memory.**

Parallelization is relatively easy to implement, starting from a
sequential program via **incremental parallelization.**

With parallel regions, **work can be shared** within **loops** or
**sections**.

Parallel regions are extending to **all subroutines** called within,
scope is **static** and **dynamic**

**Synchronization** is sometimes necessary

**Data sharing** is key to good and robust parallelism

Performance is often related to **memory layout** and **cache memory
management**
