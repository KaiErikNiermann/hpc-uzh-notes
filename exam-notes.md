# Compilers and Queues 
## version control 
- [[git-cheatsheet.pdf]]
- [[Compilers and Batch Queues]]
- [[hpc lecture 2 - Compilers and Batch Queues.pdf]]

## phases of compiler 
- [[Compilers and Batch Queues]]
- [[hpc lecture 2 - Compilers and Batch Queues.pdf]]
#### Lexical Analysis
The first phase of scanner works as a text scanner. This phase scans the source code as a stream of characters and converts it into meaningful lexemes. Lexical analyzer represents these lexemes in the form of tokens as:

`<token-name, attribute-value>`
### compilation phases
#### Syntax Analysis
The next phase is called the syntax analysis or **parsing**. It takes the token produced by lexical analysis as input and generates a parse tree (or syntax tree). In this phase, token arrangements are checked against the source code grammar, i.e. the parser checks if the expression made by the tokens is syntactically correct.
#### Semantic Analysis
Semantic analysis checks whether the parse tree constructed follows the rules of language. For example, assignment of values is between compatible data types, and adding string to an integer. Also, the semantic analyzer keeps track of identifiers, their types and expressions; whether identifiers are declared before use or not etc. The semantic analyzer produces an annotated syntax tree as an output.
#### Intermediate Code Generation
After semantic analysis the compiler generates an intermediate code of the source code for the target machine. It represents a program for some abstract machine. It is in between the high-level language and the machine language. This intermediate code should be generated in such a way that it makes it easier to be translated into the target machine code.
#### Code Optimization
The next phase does code optimization of the intermediate code. Optimization can be assumed as something that removes unnecessary code lines, and arranges the sequence of statements in order to speed up the program execution without wasting resources (CPU, memory).
#### Code Generation
In this phase, the code generator takes the optimized representation of the intermediate code and maps it to the target machine language. The code generator translates the intermediate code into a sequence of (generally) re-locatable machine code. Sequence of instructions of machine code performs the task as the intermediate code would do.
#### Symbol Table
It is a data-structure maintained throughout all the phases of a compiler. All the identifier's names along with their types are stored here. The symbol table makes it easier for the compiler to quickly search the identifier record and retrieve it. The symbol table is also used for scope management.
### Linking 
The linker is what produces the final compilation output from the object files the compiler produced. This output can be either a shared (or dynamic) library (and while the name is similar, they haven't got much in common with static libraries mentioned earlier) or an executable.

It links all the object files by replacing the references to undefined symbols with the correct addresses. Each of these symbols can be defined in other object files or in libraries. If they are defined in libraries other than the standard library, you need to tell the linker about them.

At this stage the most common errors are missing definitions or duplicate definitions. The former means that either the definitions don't exist (i.e. they are not written), or that the object files or libraries where they reside were not given to the linker. The latter is obvious: the same symbol was defined in two different object files or libraries.

## Importance of Optimization 
- [[Compilers and Batch Queues]]
- [[hpc lecture 2 - Compilers and Batch Queues.pdf]]
### summary
Compiler optimization is generally implemented using a sequence of _optimizing transformations_, algorithms which take a program and transform it to produce a semantically equivalent output program that uses fewer resources or executes faster. It has been shown that some code optimization problems are NP-complete, or even undecidable. In practice, factors such as the programmer's willingness to wait for the compiler to complete its task place upper limits on the optimizations that a compiler might provide. Optimization is generally a very CPU- and memory-intensive process. In the past, computer memory limitations were also a major factor in limiting which optimizations could be performed.

## Basics of Make 
### summary 
In software development, Make is a build automation tool that builds executable programs and libraries from source code by reading files called makefiles which specify how to derive the target program. Though integrated development environments and language-specific compiler features can also be used to manage a build process, Make remains widely used, especially in Unix and Unix-like operating systems.
### basic example application 
**Overview:**
Makefiles are used to organize code compilation efficiently. They automate the build process, ensuring that only the necessary parts of the code are recompiled when changes are made.

**Example Files:**
- `hellomake.c`: Main program.
- `hellofunc.c`: Functional code.
- `hellomake.h`: Include file.

**Manual Compilation:**
Compile using the command:
```bash
gcc -o hellomake hellomake.c hellofunc.c -I.
```
- `-I.` specifies the current directory for include files.

**Simple Makefile (Makefile 1):**
```make
hellomake: hellomake.c hellofunc.c
    gcc -o hellomake hellomake.c hellofunc.c -I.
```
- Run with `make`.
- Efficient for avoiding retyping compile commands.

**Improved Makefile (Makefile 2):**
```make
CC=gcc
CFLAGS=-I.

hellomake: hellomake.o hellofunc.o
    $(CC) -o hellomake hellomake.o hellofunc.o
```
- Use constants for compiler and flags.
- Compiles individual files before linking.

**Dependency on Include Files (Makefile 3):**
```make
CC=gcc
CFLAGS=-I.
DEPS = hellomake.h

%.o: %.c $(DEPS)
    $(CC) -c -o $@ $< $(CFLAGS)

hellomake: hellomake.o hellofunc.o
    $(CC) -o hellomake hellomake.o hellofunc.o
```
- Introduces dependency on include files.
- Re-compiles when header files change.

**General Compilation Rule (Makefile 4):**
```make
CC=gcc
CFLAGS=-I.
DEPS = hellomake.h
OBJ = hellomake.o hellofunc.o

%.o: %.c $(DEPS)
    $(CC) -c -o $@ $< $(CFLAGS)

hellomake: $(OBJ)
    $(CC) -o $@ $^ $(CFLAGS)
```
- Uses special macros `$@` and `$^` for general compilation rule.

**Organized Project Structure (Makefile 5):**
```make
IDIR =../include
CC=gcc
CFLAGS=-I$(IDIR)

ODIR=obj
LDIR =../lib

LIBS=-lm

_DEPS = hellomake.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = hellomake.o hellofunc.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.c $(DEPS)
    $(CC) -c -o $@ $< $(CFLAGS)

hellomake: $(OBJ)
    $(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
    rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~
```
- Defines paths for include, object, and library directories.
- Cleans up source and object directories with `make clean`.

**Conclusion:**
The provided Makefile can be modified for small to medium-sized software projects, offering efficiency and automation in the compilation process. Additional rules can be added, and the Makefile structure can be adapted as needed. For more details, refer to the GNU Make Manual.
### Batch queue systems 
- [[Compilers and Batch Queues]]

---
# Performance 
## Moore's Law and Top500 list
- [[Introduction to Parallel Programming]]
- [[hpc lecture 5 - Introduction to Parallel Programming.pdf]]
## strong vs weak scaling 
- [[Introduction to Parallel Programming]]
- [[hpc lecture 5 - Introduction to Parallel Programming.pdf]]
### Strong scaling 
Concerns the speedup for a fixed problem size with respect to the number of processors, and is governed by Amdahl’s law.
![[Pasted image 20240114040408.png]]
![[Pasted image 20240114040416.png]]
### Weak scaling 
Concerns the speedup for a scaled problem size with respect to the number of processors, and is governed by Gustafson’s law.
![[Pasted image 20240114040436.png]]![[Pasted image 20240114040444.png]]


## Latency and Bandwidth 
- 
### summary 
- [[Introduction to Parallel Programming]]
- [[hpc lecture 5 - Introduction to Parallel Programming.pdf]]
Latency and throughput are two metrics that measure the performance of a computer network. Latency is the delay in network communication. It shows the time that data takes to transfer across the network. Networks with a longer delay or lag have high latency, while those with fast response times have lower latency. In contrast, throughput refers to the average volume of data that can actually pass through the network over a specific time. It indicates the number of data packets that arrive at their destinations successfully and the data packet loss.
## Memory, cache and bus 
- [[Introduction to Hybrid Computing]]
- [[hpc lecture 11 - Hybrid Computing Introduction.pdf]]
## Benchmarking 
- [[Benchmarking]]
- [[hpc lecture 8 - Benchmarking.pdf]]
### different ways of benchmarking
1. **Compilation and Basic Benchmarking:**
    - Basic benchmarking is performed using Linux's built-in `time` utility.
    - Compilation commands with optimization flags are used to observe changes in runtime.
2. **Debugging and Minimizing File Size:**
    - Debugging flags (`-g`) retain information for debugging.
    - The `strip` utility removes unnecessary symbols from the compiled executable, reducing file size.
3. **Benchmarking with Craypat:**
    - Craypat is a performance analysis tool for HPC systems.
    - It requires loading the `perftools-lite` module and using the Clang compiler after loading Craypat.
    - A job script is used to submit the program for analysis, and `pat_report` provides detailed profiling information.
4. **Sampling and App2:**
    - Sampling is a method to analyze program performance by sampling the call stack at intervals.
    - Craypat provides detailed profiling information, and `app2` interprets and visualizes the data.
5. **Self Timing:**
    - Basic timing concepts involve measuring the start and end time to calculate elapsed time.
    - Functions like `time()`, `gettimeofday()`, and `clock_gettime()` are used to obtain time information.
6. **High-Resolution C++ Clock:**
    - `std::chrono` provides a high-resolution clock for precise timing.
    - Example code demonstrates measuring elapsed time using the high-resolution clock.
7. **Advanced - cycle.h:**
    - The `cycle.h` header provides an advanced self-timing implementation based on the number of ticks.
    - The `getticks()` function is used to obtain tick counts, and `elapsed()` calculates the difference.
---
# OpenMP 
## directive based 

## memory model - threads vs cores

## serial and parallel region 

## parallel for loop 

## synchronization and performance 

## shared vs private variables

## OpenMP thread scheduling

---
# MPI
## model

## load balancing / domain decomposition 

## compiling MPI programs 

## common functions 

## message passing and common problems


---
# Cloud & Containers 
## difference between VM and container 

## snapshots 

## ephemeral computing 

## root in vms 

## dockerhub 

## accessing data / mounting

---
# MapReduce 

## compute sent to data 

## data model 

## map and reduce phase

## key and value

---
# Hybrid Computing 

## CPU sockets/cores and GPU SMs/cores

## SIMD 

## CPU vs GPU 

## divergence 

## data alignment 

## latency hiding and occupancy 

## instruction latency on GPU vs CPU 


---
# OpenACC
## Directive Based 
As mentioned before the fundamental way most programmers are going to
use OpenACC is through directives, the same principle as with OpenMP,
but in this case its generally meant for parallel programming using the
GPU.
## Basic GPU Operations 
### kernel invocation 
Similar to the latency caused by starting up a parallel region using MPI
the cost of invoking the GPU kernel to execute some routine is an
expensive processes whos trade-off should be well considered.
### allocation
1. `pcreate(x[0:N])`: Indicates the creation of a device pointer for array `x` on the GPU, managing it on the device during subsequent parallel loops.
2. `pcopyout(y[:N])`: Specifies that the data at pointer `y` (representing the computation results) should be copied back from the GPU to the host after the completion of parallel loops.
## data management 

## difference kernel and parallel 

## difference grid, worker and vector 

## shared vs private variables 

## synchronization 

## async ops 

---
# CUDA 
## compiling 
## streaming multiprocessor 

## grid/block/warp/thread 

## indexing