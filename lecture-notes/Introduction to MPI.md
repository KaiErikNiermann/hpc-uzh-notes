Introduction to MPI
===================

------------------------------------------------------------------------

Introduction and History
------------------------

💡

MPI is a library which allows process coordination and scheduling using
a **message-passing** paradigm.

### Sequential programming model

In the sequential programming model the program is executed by **only
one process**.

All variables and constants of program are **allocated in process
memory**.

Process is executed on a **physical CPU** of the node. ( in an HPC
setting )

### Message passing programming model

All variables are by default **private** and reside in **local memory**
of each process.

Each process can execute a different **part of the program.**

Variables can be **exchanged between processes** via a call to the
**message passing subroutine**.

### Message passing concepts

**Message attributes**

Message is sent from source process (w/ sender address) to target
process (w/ receiver address)

Message contains a header with the following properties and some data :

-   **Sender Id** - identifier of the sending process
-   **Data type** - Type of the message data
-   **Data length** - The length of the message
-   **Receiver Id** - The identification of the receiver

**Environment**

The messages are managed and interpreted by a runtime system which
handles message exchange.

Messages are sent to specific addresses. Receiving processes must be
able to classify and interpret incoming message.

And MPI application is a group of autonomous processes on different
nodes (computers) communicating to the other processes via calls to
routines in MPI library.

------------------------------------------------------------------------

Distributed vs Shared memory
----------------------------

A key distinction is the basic memory model which OpenMP and MPI used to
implement parallel computations. MPI uses a **distributed memory
paradigm,** whereas OpenMPI uses a **shared memory paradigm.**
### Shared memory

Data are shared implicitly ( by default ) within the node through RAM.
This is the main method that underlies OpenMP’s implementation of
parallelism.
### Distributed memory

Data are transferred explicitly ( by the developer ) between nodes
through the network.

Usually via some interconnect network between the nodes.

This is the main method that underlies MPIs implementation of
parallelism.

![](Introduction%20to%20MPI%20be91a41bb4424987ae74d6a74ac0e8ca/Untitled.png)

### Data distribution

In MPI data are **distributed between nodes** using a **domain
decomposition strategy**. This being a technique for solving problems by
dividing the spatial domain of the problem into smaller subdomains.

------------------------------------------------------------------------
MPI Environment
---------------
The MPI environment generally consists of a few core features that make
up an MPI application, this includes things like **environment
variables, launching the MPI environment, terminating it** and then any
**MPI routine calls** happening during the parallel section of the code.

To get the version for this information we used in the introduction see

------------------------------------------------------------------------

Communicators
-------------

All inter-process communications occur via communicator objects, which
define processes that can talk.
### Point-to-point communications

Point-to-point communication involves the **exchange of data** between
two processes in a **parallel or distributed computing environment**.

**Identification**

One process is the receiver identified by their rank and another is the
sender likewise identified by their rank. The collection of all
processes that can talk to each other is called the **communicator**.

### Message header

As indicated in the introduction messages sent between processes are
typically also accompanied by a header. To elaborate on the exact
contents of this header

**Sender rank** which identifies the sender, so the processes that
initiated the communication

**Receiver rank** which identifies, unsurprisingly, the receiver (wowie)

**Message tag** which is some piece of user-defined label/tag that helps
the processing of the message on the receivers end, in other words it
tells it how to handle the message

**Communicator identifier** which specifies the communicator to which
the sender and receiver belong, which ensures that the communications
occur within the designated group

**Type of exchange data** which indicates well, the type of data being
exchanged, which can be primitives (e.g. int, float) or more complex
data structures

### Transfer modes

For each communication you can select a transfer mode corresponding to a
specific **protocol.**

**Protocols**

These decide how the data is transferred between processes and can vary
in things like performance, reliability and other characteristics.

Some examples include

-   Synchronous - So blocking communication

 

-   Asynchronous - Non blocking communication

 

-   Buffered

------------------------------------------------------------------------

Basic synchronous communication
-------------------------------

💡

This can be achieved with the `MPI_Send` and `MPI_Recv` function

### `MPI_Send`

This routine sends a message of `count` elements starting at address
`buf` and of type `datatype` . The message is tagged by `tag` and is
sent to the process of rank `dest` within the communicator `comm`.

**Synopsis**

``` code
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
```

**Input Parameters**

-   **buf**: Initial address of the send buffer (choice)

 

-   **count**: Number of elements in the send buffer (nonnegative
    integer)

 

-   **datatype**: Datatype of each send buffer element (handle)

 

-   **dest**: Rank of the destination (integer)

 

-   **tag**: Message tag (integer)

 

-   **comm**: Communicator (handle)

### `MPI_Recv`

This routine receives message of `count` elements starting at address
`buf` and of type `datatype`. Message bust be tagged with `tag` and is
coming form the process of rank `source` within the communicator `comm`

`status` receives information about the communication (i.e. source, tag,
error codes). `MPI_Recv` only works with a corresponding `MPI_Send` if
they have the same header (same `source`, `dest`, `tag`, `comm`)

**Synopsis**

``` code
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
```

**Output Parameters**

-   **buf**: Initial address of the receive buffer (choice)

 

-   **status**: Status object (Status)

**Input Parameters**

-   **count**: Maximum number of elements in the receive buffer
    (integer)

 

-   **datatype**: Datatype of each receive buffer element (handle)

 

-   **source**: Rank of the source (integer)

 

-   **tag**: Message tag (integer)

 

-   **comm**: Communicator (handle)

### Blocking

As mentioned before both of these calls work in a synchronous or
**blocking** fashion, this has different implications to keep in mind

**Blocking send**

Execution remains blocked until the message can be re-written without
the rusk of overwriting the value being sent.

**Block receive**

The execution remains blocked until the message is fully received in
`buf`

### Example - Synchronous communication

**installation requirements**

``` code
// for mpicc command 
sudo apt install openmpi-bin 
// for mpi.h header       
sudo apt install libopenmpi-dev    
```

**compilation**

``` code
mpicc -o test test.c
mpirun -np 2 ./test
```

`-np2` denotes you want to use 2 processes, namely `{0, 1}`

`test.c`

``` code
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, value;
    int tag = 127;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        value = 17;
        MPI_Send(&value, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(&value, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        printf("proc 1 recv %d from proc 0\n", value);
        printf("status source %d, tag %d, error %d\n", 
            status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR);
    }

    MPI_Finalize();

    return 0;
}
```

`output`

``` code
proc 1 recv 17 from proc 0
status source 0, tag 127, error 0
```

$$\underset{\texttt{MPI\_COMM\_WORLD}\text{~communicator}}{\underbrace{{\texttt{proc}\ \texttt{0}}\overset{\texttt{\{MPI\_INT}\ \texttt{val=17,}\ \texttt{tag=127,}\ \texttt{source=0\}}}{\rightarrow}{\texttt{proc}\ \texttt{1}}}}$$MPI\_COMM\_WORLD communicator

proc 0{MPI\_INT val=17, tag=127, source=0}

​proc 1​​

I think for the most part everything should be pretty self explanatory
here

1.  We start off by initializing the parallel region.

 

1.  For the 0th process which will only execute its code, we send the
    value of 17 to the process with rank 1

 

1.  Then in turn for the process of rank 1 it blocks and waits for
    anything from the rank 0th process

 

1.  Once we received everything execution continues in rank 1 and we
    print out the received data along with metadata about the message
    from the status struct.

### Communication Options

We can decide various options for the send and receive commands that
alter certain behaviors.

|                                  |                                                                                                                                                                                                              |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MPI_ANY_SOURCE` / `MPI_ANY_TAG` | These are wildcard options, in case specifying a unique sender or tag would not matter.                                                                                                                      |
| `MPI_PROC_NULL`                  | Any communication with this as the `rank` will have no effect, its in essence a null communication                                                                                                           |
| `MPI_STATUS_IGNORE`              | As the name implies its just if you don’t care about the status                                                                                                                                              |
| Variants                         | There are variants of the send and receive commands that behave similar to the standard functions but with slightly different behavior for certain purposes like `MPI_SENDRECV()` or `MPI_SENDRCV_REPLACE()` |

**Note**

Its possible to replace the predefined datatypes with user defined data
types.

### Predefined datatypes

Here is just a table of all the predefined datatypes you can use

| MPI Datatype                  | Description                                          |
|-------------------------------|------------------------------------------------------|
| MPI\_CHAR                     | Character                                            |
| MPI\_INT                      | Integer                                              |
| MPI\_LONG                     | Long integer                                         |
| MPI\_FLOAT                    | Single-precision floating-point                      |
| MPI\_DOUBLE                   | Double-precision floating-point                      |
| MPI\_SHORT                    | Short integer                                        |
| MPI\_LONG\_LONG               | Long long integer                                    |
| MPI\_UNSIGNED\_CHAR           | Unsigned character                                   |
| MPI\_UNSIGNED                 | Unsigned integer                                     |
| MPI\_UNSIGNED\_LONG           | Unsigned long integer                                |
| MPI\_UNSIGNED\_SHORT          | Unsigned short integer                               |
| MPI\_UNSIGNED\_LONG\_LONG     | Unsigned long long integer                           |
| MPI\_BYTE                     | Byte                                                 |
| MPI\_WCHAR                    | Wide character (Unicode)                             |
| MPI\_LONG\_DOUBLE             | Extended precision floating-point                    |
| MPI\_C\_FLOAT\_COMPLEX        | Complex number with single-precision real and imag   |
| MPI\_C\_DOUBLE\_COMPLEX       | Complex number with double-precision real and imag   |
| MPI\_C\_LONG\_DOUBLE\_COMPLEX | Complex number with extended precision real and imag |
| MPI\_UNSIGNED\_CHAR           | Unsigned character                                   |
| MPI\_PACKED                   | Packed data                                          |

------------------------------------------------------------------------

Communicators
-------------

💡

I was bored so I wanted to do a small exploration into what exactly
communicators are. Abstractly we know they are just collections of
processes with certain ranks. But what does this actually mean. Lets
take a look with the following example.

### Example - Splitting the world communicator

We start off by initializing the parallel region and required variables

``` code
// Required vairables
int rank, size;
MPI_Comm even_comm, odd_comm;
MPI_Comm_set_name(even_comm, "Even_Comm");
MPI_Comm_set_name(odd_comm, "Odd_Comm");

// Initialize MPI
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
```

Because the rank changes once you place a certain process into the
region of a new communicator lets save the rank of the process from when
it was in the world communicator.

After this lets put all processes with an even rank into the `even_comm`
and odd in `odd_comm`

``` code
int var = rank;

// Split the world into two communicators based on rank parity
if (rank % 2 == 0) {
    int group_color = 0;
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &even_comm);
    MPI_Comm_set_name(even_comm, "Even_Comm");
} else {
    int group_color = 1;
    MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &odd_comm);
    MPI_Comm_set_name(odd_comm, "Odd_Comm");
}
```

Here the `MPI_Comm_split` function first of all takes the communicator
we want to split, it then takes a color, which just denotes the group we
want to place the new process in.

We are creating 2 subgroups, so the colors are 0 and 1. We then pass
along the rank to the new communicator

We can then print out some information about the communicators for the
different ranks depending on if they are even or not.

``` code
void dump_comm(MPI_Comm comm, int var) {
    int rank, size;
    int comm_name_size = MPI_MAX_OBJECT_NAME;
    char comm_name[MPI_MAX_OBJECT_NAME];
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Comm_get_name(comm, comm_name, &comm_name_size);
    
    printf("Comm: %s, Rank: %d, Size: %d, og rank: %d\n", 
        comm_name, rank, size, var);
}
```

The `output` we get is

``` code
Comm: Odd_Comm, Rank: 0, Size: 2, og rank: 1
Comm: Even_Comm, Rank: 1, Size: 2, og rank: 2
Comm: Odd_Comm, Rank: 1, Size: 2, og rank: 3
Comm: Even_Comm, Rank: 0, Size: 2, og rank: 0
```

Which I think quite intuitively shows the result of what’s going on.
Ranks (`rank`) which were originally even are in the `Even_comm` and vv.
for odd ranks. Since each communicator now contains 2 processes the
ranks are numbered `0` and `1` for each comm.

[Full code for communicator
split](Introduction%20to%20MPI%20be91a41bb4424987ae74d6a74ac0e8ca/Full%20code%20for%20communicator%20split%206a43c1bd60b94b659d6f5936477f6d1f.md)

------------------------------------------------------------------------

Deadlock
--------

💡

As mentioned before a deadlock happens when two processes or threads
want a specific resource which both are currently holding onto leading
to a circular wait loop. Due to the fact that `MPI_Send` and `MPI_Recv`
are blocking we can get a deadlock under certain circumstances.

### Deadlock with `MPI_Send` and `MPI_Recv`

A classical mistake which can cause a deadlock with the use of these two
functions is to first to initiate a blocking send from some Process A
and then then have a receive from Process B, and likewise in Process B
have a send to Process A and then a receive from process A

We can demonstrate this in code

``` code
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        data = 42; // 
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        data = 13;
        MPI_Send(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Finalize();

    return 0;
}
```

**Note** something rather annoying when trying to demonstrate the
concept of a deadlock in MPI is that it **doesn’t necessary occur**.
Reason for this being that **in theory** a deadlock happens only if the
implementation of the `MPI_Send()` function **does not complete /
blocks** until the corresponding Receive function has received all the
data.

### **Blocking behavior in practice**

But **in practice** `MPI_Send()` doesn’t necessarily block, some
implementations might return control to the caller when the buffer has
been sent to a lower communication layer.

Other implementations wait until there is a matching `MPI_Recv()` call
on the other end, i.e. the receiver is ready to receive.

**Avoid Deadlock code, even if it works**

In most implementations though the larger the piece of data you are
sending the more likely it is that `MPI_Send()` decides that it wants to
wait for `MPI_Recv()` in turn meaning you are more likely to encounter a
deadlock.

So just in general avoid code that theoretically causes a deadlock
because even if it might work right now it does not guarantee that it
works in the future.

**Why does this cause a deadlock ? ( in theory )**

Assuming `MPI_Send()` is implemented in a way where it does block until
the receiver is finished. We encounter the following scenario

1.  Process 0 sends 42 → Process 1, it now blocks at the `MPI_Send()`
    call until Process 1’s Receive call has completed, that is, until
    all data has been transmitted

 

1.  Process 1 sends 13 → Process 0, it now too blocks at the
    `MPI_Send()` call for the same reason as stated above

 

1.  Deadlock ⇒ We can now see that Process 0 is waiting for the Receive
    call to complete from Process 1, but clearly process 1 is still on
    its send call waiting for the receive call of Process 0 to complete,
    which is still on its send call … you get the picture.

------------------------------------------------------------------------

Collective Communication
------------------------

💡

So we’ve now mentioned how point to point communication works, that is,
communication from A → B. But as you might be able to guess this alone
isn’t very useful if we want to do large scale parallel computations. So
we introduce the concept of **collective communication** which refers to
the process of using certain functions to **abstract complex series of
point-to-point communications.**

### Combining Point-to-Point calls

Point-to-point communication obviously still underlies the concept of
message passing, but to make large scale communication possible there
are methods which abstract very common complex sequences of
point-to-point communications for various purposes. Some properties of
these abstracted communication patterns are that

**Involve all processes**

So collective communication generally refers to communication happening
between all processes in the domain of a specific communicator.

**Blocking nature**

Collective communications are generally blocking, that is, they are only
completed when all the inner point to point communications are finished

**No need for barriers**

As an implicit part of these communications there is built in
synchronization due to their inherently blocking nature

**No need to specify tags**

You don’t need to specify tags along with the communications, the
collective operation explains the type of communication that occurs and
all involved processes are inherently a part of this.

### Collective operations

Some examples of collective operations, so operations occurring on all
processes in a communicator

`MPI_Barrier()`

This is a global synchronization collective operation. It ensures that
all processes in an MPI program reach a certain point before continuing
execution.

**Collective data transfer**

`MPI_Bcast` , `MPI_Scatter` , `MPI_Gather`

These are all functions used for **collective data transfer**. That is,
the transfer of data between all processes in a communicator.

**Collective transfer + Operations**

`MPI_Reduce()`

This is a type of collective operation in which we have both data being
transferred between all processes but also an operation (e.g. summation,
maximization, minimum, etc. ) applied to the data as it’s gathered from
different processes.

------------------------------------------------------------------------

Reduce operation
----------------

💡

An operation / method in which we take some set of distributed elements
(e.g. certain variable values) across a set of processes in a
communicator and then apply some operation on them to obtain a single
value.

### `MPI_Reduce()`

This routine perform a reduction over `count` elements starting at
address `sendbuf` and of type `datatype`. The reduction is based on
operation `op` for each process within the communicator `comm`. This
routine returns the result at address `recvbuf` only in process `root`.

**Function signature**

``` code
int MPI_Reduce(
                            const void *sendbuf, 
                                         void *recvbuf, 
                                               int count, 
                         MPI_Datatype datatype,
                                   MPI_Op op, 
                                                int root, 
                                            MPI_Comm comm
                            )
```

**Input Parameters**

-   `sendbuf`: Address of the send buffer (choice).

 

-   `count`: Number of elements in the send buffer (integer).

 

-   `datatype`: Data type of elements of the send buffer (handle).

 

-   `op`: Reduce operation (handle).

 

-   `root`: Rank of the root process (integer).

 

-   `comm`: Communicator (handle).

**Output Parameters**

-   `recvbuf`: Address of the receive buffer (choice, significant only
    at the root).

**Intuition example**

Not to hard to understand, if we use `+` as an example operation and a
set of values `{1, 2, 3}`, each one belonging to 1 of 3 processes and we
say that we want the result to be collected on Process `0`. Then we just
apply the operation, namely

$$1 + 2 + 3 = 6$$1+2+3=6

And then we take the result `6` and write it to some variable in the
memory of process 0.

**Code example**

`test.c`

``` code
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int res = 0;
    int s = rank + 1;
    
    MPI_Reduce(&s, &res, 1, MPI_INT, 
        MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("res = %d\n", res);
    }

    MPI_Finalize();

    return 0;
}
```

`output`

``` code
res = 6
```

**Result**

Unsurprisingly the result is 6, in each process we take the rank and add
1, then sum the resulting `s` from each process, this being

Process 0 → s = 1

Process 1 → s = 2

Process 2 → s = 3

Therefore res = 1 + 2 + 3 = 6

And since we sent it to process 0 in the function call that’s where
`sum` will = 6. If you want you can test for yourself, in all other
processes `res` will remain 0.

### `MPI_Allreduce()`

Routine preforms a reduction over `count` elements starting at address
`sendbuf` and of type `datatype`. The reduction is based on operation
`op` for each process with the communicator `comm`

The routine returns the result at address `recvbuf` in **all
processes**. So its the same as reduce with the exception that it
returns the result of the reduction operations to all processes instead
of a specified single process.

**Function signature**

``` code
int MPI_Allreduce(
                                    const void *sendbuf, 
                                      void *recvbuf, 
                                                    int count,
                MPI_Datatype datatype, 
                                                        MPI_Op op, 
                                                MPI_Comm comm
                                    )
```

**Input Parameters**

-   `sendbuf`: Starting address of the send buffer (choice).

 

-   `count`: Number of elements in the send buffer (integer).

 

-   `datatype`: Data type of elements of the send buffer (handle).

 

-   `op`: Operation (handle).

 

-   `comm`: Communicator (handle).

**Output Parameters**

-   `recvbuf`: Starting address of the receive buffer (choice).

**Code example**

`test.c`

``` code
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int res = 0;
    int s = rank + 1;
    
    MPI_Allreduce(&s, &res, 1, 
        MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("rank %d: res = %d\n", rank, res);

    MPI_Finalize();

    return 0;
}
```

`output`

``` code
rank 1: res = 6
rank 2: res = 6
rank 0: res = 6
```

**Result**

Again, the result is 6 for the same reason as above, its just that in
all processes `{0, 1, 2}`, we have the result saved, not just in a
single process.

### Reduction operations

| Operation     | Description                                                     |
|---------------|-----------------------------------------------------------------|
| `MPI_MAX`     | Computes the maximum element.                                   |
| `MPI_MIN`     | Computes the minimum element.                                   |
| `MPI_SUM`     | Computes the sum of all elements.                               |
| `MPI_PROD`    | Computes the product of all elements.                           |
| `MPI_LAND`    | Computes the logical AND of all elements.                       |
| `MPI_BAND`    | Computes the bitwise AND of all elements.                       |
| `MPI_LOR`     | Computes the logical OR of all elements.                        |
| `MPI_BOR`     | Computes the bitwise OR of all elements.                        |
| `MPI_LXOR`    | Computes the logical XOR of all elements.                       |
| `MPI_BXOR`    | Computes the bitwise XOR of all elements.                       |
| `MPI_REPLACE` | Replace the receive buffer with the send buffer (no reduction). |

This table provides a brief description of each reduction operation
available in MPI. You can use these operations with functions like
`MPI_Reduce` or `MPI_Allreduce` to perform collective reductions in
parallel programs.

------------------------------------------------------------------------

Collective data sending operations
----------------------------------

💡

Collective data sending refers to a collection of routines which preform
different types of data transmission across multiple processes.

### General theory

There are two main forms of collective sending, there is broadcasting
and scattering.

**Broadcasting**

Here we send the same piece of data from one process to all other
processes.

**Scaterring**

Here we have multiple pieces of data within one process, and we are
scattering, that is, evenly distributing this data amongst all other
processes.

### `MPI_Bcast()`

**Intuition**

Here we very simply send a single message from one process to all other
processes. You could think of this like yelling some information into a
room for everyone to hear. Everyone has the same information and one
person broadcasts it.

**Syntax information**

Send `count` elements starting at address `buffer` and of type
`datatype` . The broadcast is performed by the process `root` towards
all processes within communicator `comm`. The routine returns the
results at address in all processes including root.

![](Introduction%20to%20MPI%20be91a41bb4424987ae74d6a74ac0e8ca/Untitled%201.png)

**Function Signature**

``` code
int MPI_Bcast(
         void *buffer,
            int count,           MPI_Datatype datatype,
             int root,
            MPI_Comm comm
             )
```

**Input/Output Parameters**

-   `buffer`: Starting address of the buffer (choice).

**Input Parameters**

-   `count`: Number of entries in the buffer (integer).

 

-   `datatype`: Data type of the buffer (handle).

 

-   `root`: Rank of the broadcast root (integer).

 

-   `comm`: Communicator (handle).

**Code example**

`test.c`

``` code
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int message = 0;
    if (rank == 0) message = 100;

    MPI_Bcast(&message, 1, 
        MPI_INT, 0, MPI_COMM_WORLD);
    
    printf("Rank %d, message %d\n", 
                rank, message);

    MPI_Finalize();

    return 0;
}
```

`output`

``` code
Rank 0, message 100
Rank 1, message 100
Rank 2, message 100
```

**Result**

As you might have already expected here we broadcast the value 100 from
proc 0 to all other processes, which then all have the value 100 in
their memory.

If you wann see for yourself you can add a print statement before the
broadcast and confirm that only proc 0 has message = 100.

### `MPI_Scatter()`

**Intuition**

Halloween candy seems like a good analogy. You have 8 people wanting
candy, and you as evenly as you can distribute this candy amongst the
people.

**Function Signature**

``` code
int MPI_Scatter(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm
    )
```

![](Introduction%20to%20MPI%20be91a41bb4424987ae74d6a74ac0e8ca/Untitled%202.png)

**Input Parameters**

-   `sendbuf`: Address of the send buffer (choice, significant only at
    the root).

 

-   `sendcount`: Number of elements sent to each process (integer,
    significant only at the root).

 

-   `sendtype`: Data type of send buffer elements (significant only at
    the root) (handle).

 

-   `recvcount`: Number of elements in receive buffer (integer).

 

-   `recvtype`: Data type of receive buffer elements (handle).

 

-   `root`: Rank of the sending process (integer).

 

-   `comm`: Communicator (handle).

**Output Parameters**

-   `recvbuf`: Address of the receive buffer (choice).

**Syntax information**

Send `sendcount` elements starting at `sendbuf` and of type `sendtype` .
Scatter is performed by and in process `root` towards all process in
`comm`. Routine returns type `recvtype` at address `recvbuf` and size
`recvcount` in all processes in.

There are some key things to remember here

-   The combination of `sendcount` and `sendtype` must represent the
    same amount of data than the combination `recvcount` and `recvtype`

 

-   Data are scattered in chunks of equal sizes

 

-   Chunk with index `i` is always sent to rank `i`

**Code example**

`test.c`

``` code
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define n 8

int main(int argc, char* argv[]) {
    int rank, size; 
    int data[n] = {
        10, 20, 30, 40, 50, 60, 70, 80
    };

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int block_size = n / size;
    int *recv = malloc(block_size * sizeof(int));

    MPI_Scatter(
        data,           // send buffer 
        block_size,     // send count
        MPI_INT,        // send type
        recv,           // recv buffer
        block_size,     // recv count
        MPI_INT,        // recv type
        0,              // root
        MPI_COMM_WORLD
    );

    // print received data
    printf("Rank %d received: \n", rank);
    for (int i = 0; i < block_size; i++) 
        printf("%d ", recv[i]);
    printf("\n");

    free(recv);
    MPI_Finalize();
    return 0;
}
```

`output`

``` code
Rank 0 received: 
10 20 
Rank 1 received: 
30 40 
Rank 2 received: 
50 60 
Rank 3 received: 
70 80
```

**Run command**

``` code
mpirun -np 4 ./test
```

**Result**

Since we have 4 processes and 8 pieces of data an even distribution
would mean 4 chunks with 2 pieces each.

This being exactly what happens. Furthermore as we said chunk with index
i is sent to rank 0 we can see that quite nicely reflected in the
result. 10, 20 are the first two digits making up chunk 0 and they are
sent to proc with rank 0.

### Uneven data distribution

I think a fair question to ask is what if we don’t have a situation
where data splits as nice as this. So lets choose 3 processes instead of
4.

**Run command**

``` code
mpirun -np 3 ./test
```

`output`

``` code
Rank 2 received: 
50 60 
Rank 0 received: 
10 20 
Rank 1 received: 
30 40
```

**Observations**

I think if you spend some time to ponder on why this is the output it
should become clear. Remember main rule here

**Even distribution** To get an even distribution we divide the number
of elements we want to distribute by the number of places we want to
distribute these elements to. For 3 processes that yields 8 / 3 = 2.667,
clearly we can’t round up since if we were to send 3 elements to 3
processes each we would need 9 total elements, thus we must round down.

Hence we only end up sending two elements to each process. And as we saw
with the first example the 0th chunk goes to the proc with the 0th rank.

------------------------------------------------------------------------

Collective data gathering operations
------------------------------------

💡

The same way we would reasonable want to send out a bunch of information
it makes sense that we would also like some operations for collecting
data back.

### `MPI_Gather()`

**Intuition**

I mean not too complex of a thing, its just gathering a bunch of values
from other processes.

**Syntax information**

We send `sendcount` elements starting at address `sendbuf` and of
`sendtype`. The scatter is performed by and in all processes within the
communicator `comm` towards the process `root`.

![](Introduction%20to%20MPI%20be91a41bb4424987ae74d6a74ac0e8ca/Untitled%203.png)

**Function Signature**

``` code
int MPI_Gather(
    const void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
)
```

Key things to remember

`sendcount` and `sendtype` must represent same amount of data as
`recvcount` and `recvtype`

Data is collected in the process `root` in the same order as the order
processors ranks.

**Input Parameters:**

-   `sendbuf`: Address of the send buffer (choice, significant only at
    the root).

 

-   `sendcount`: Number of elements sent to each process (integer).

 

-   `sendtype`: Data type of send buffer elements (significant only at
    the root) (handle).

 

-   `recvcount`: Number of elements in receive buffer (integer).

 

-   `recvtype`: Data type of receive buffer elements (handle).

 

-   `root`: Rank of the sending process (integer).

 

-   `comm`: Communicator (handle).

**Output Parameters:**

-   `recvbuf`: Address of the receive buffer (choice).

This routine collects a result of type `recvtype` at address `recvbuf`
and size `recvcount` only in the process `root`

**Code example**

`test.c`

``` code
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define n 8

int main(int argc, char* argv[]) {
    int rank, size; 
    int block_length, *proc_data;
    int recv_data[n] = {0};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    block_length = n / size;
    proc_data = (int*)malloc(block_length * sizeof(int));
    for (int i = 0; i < block_length; i++) 
        proc_data[i] = (rank + 1) * 10;

    MPI_Gather(
        proc_data,     // send buffer 
        block_length,  // send count
        MPI_INT,       // send type
        recv_data,     // recv buffer
        block_length,  // recv count
        MPI_INT,       // recv type
        0,             // root
        MPI_COMM_WORLD // communicator
    );
    
    if (rank == 0) {
        for (int i = 0; i < n; i++) 
            printf("%d ", recv_data[i]);
        printf("\n");
    }
    
    MPI_Finalize();
    return 0;
}
```

**Run command**

``` code
mpirun -np 3 ./test
```

`output`

``` code
10 10 20 20 30 30 0 0
```

**Run command**

``` code
mpirun -np 4 ./test
```

`output`

``` code
10 10 20 20 30 30 40 40 
10 10 20 20 30 30 40 40 
10 10 20 20 30 30 40 40 
10 10 20 20 30 30 40 40
```

### **Outputs explained**

In some ways its kind of like the reverse of scatter. We start with each
process having some chunk of data it would like to send to a single
process for collection.

**4 processes**

Since the data process rank corresponds to the index of the data it
sends we can see that process 0 with the data `{10, 10}` gets placed in
the corresponding first spot. And so on for rank 1, 2 and 3.

**3 processes**

Here we can see that it behaves the exact same way as for 4 processes,
it just leaves the last little bit untouched because there are no more
processes to add their data.

There are various ways you could I guess understand this as, easiest in
my opinion is just seeing it as a bunch of process in the the order of
their ranks appending data to a shared array.

### `MPI_Allgather()`

Same concept as gather but we just all processes collect collect the
values as opposed to just one.

**Syntax Information**

In `MPI_Allgather`, each process provides a send buffer containing its
own data. The data is then gathered from all processes in the
communicator `comm` and distributed to all processes, including the
sender, in the receive buffer `recvbuf`. The number of elements sent by
each process is defined by `sendcount`, and the number of elements
received by each process is defined by `recvcount`. The data types of
the send and receive buffers are specified using `sendtype` and
`recvtype` parameters, respectively.

**Intuition**

`MPI_Allgather` is similar to `MPI_Gather`, but instead of gathering the
data into a single process, it gathers the data from all processes and
distributes it to all processes.

**Function Signature**

``` code
int MPI_Allgather(
    const void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm
)
```

**Input Parameters**

-   `sendbuf`: Address of the send buffer (choice).

 

-   `sendcount`: Number of elements sent to each process (integer).

 

-   `sendtype`: Data type of send buffer elements (handle).

 

-   `recvcount`: Number of elements received from each process
    (integer).

 

-   `recvtype`: Data type of receive buffer elements (handle).

 

-   `comm`: Communicator (handle).

**Output Parameters**

-   `recvbuf`: Address of the receive buffer (choice).

**Code Example**

`test.c`

``` code
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define n 8

int main(int argc, char* argv[]) {
    int rank, size; 
    int block_length, *proc_data;
    int recv_data[n] = {0};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    block_length = n / size;
    proc_data = (int*)malloc(block_length * sizeof(int));
    for (int i = 0; i < block_length; i++) 
        proc_data[i] = (rank + 1) * 10;

    MPI_Allgather(
        proc_data,     // send buffer 
        block_length,  // send count
        MPI_INT,       // send type
        recv_data,     // recv buffer
        block_length,  // recv count
        MPI_INT,       // recv type
        MPI_COMM_WORLD // communicator
    );
    
    for (int i = 0; i < n; i++) 
        printf("%d ", recv_data[i]);
    printf("\n");
    
    MPI_Finalize();
    return 0;
}
```

**Run command**

``` code
mpirun -np 4 ./test
```

`output`

``` code
10 10 20 20 30 30 40 40 
10 10 20 20 30 30 40 40 
10 10 20 20 30 30 40 40 
10 10 20 20 30 30 40 40
```

**Reason**

I think pretty self explanatory hopefully, same exact thing as with the
previous one, just over all threads.

### `MPI_Alltoall()`

**Intuition**

We have a bunch of people who each have a collection of numbered items;
say from 0 to 3 marked with their name. Assuming all people also have a
number associated with their name from 0 to 3. `Alltoall` would mean
that everyone takes the items from every other person with their
corresponding number.

For example Person 0 with name A keeps the item with his number A0, but
takes B0, C0 and D0 since the numbers all correspond to his number. And
same goes for all other instances.

![](Introduction%20to%20MPI%20be91a41bb4424987ae74d6a74ac0e8ca/Untitled%204.png)

**Syntax rules**

Send data from all process to all other processes.

**Function signature**

``` code
int MPI_Alltoall(
    const void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm
)
```

**Input Parameters:**

-   `sendbuf`: Address of the send buffer (choice).

 

-   `sendcount`: Number of elements sent to each process (integer).

 

-   `sendtype`: Data type of send buffer elements (handle).

 

-   `recvcount`: Number of elements received from each process
    (integer).

 

-   `recvtype`: Data type of receive buffer elements (handle).

 

-   `comm`: Communicator (handle).

**Output Parameters:**

-   `recvbuf`: address of the receive buffer

**Code example**

`test.c`

``` code
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define n 8
#define elem(rank) (rank + 1) * 10

int main(int argc, char* argv[]) {
    int rank, size; 
    int proc_data[n]; 
    int recv_data[n];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int i = 0; i < 4; i++) 
        proc_data[i] = elem(rank) + i;

    printf("0 rank %d: ", rank);
    for (int i = 0; i < 4; i++) 
        printf("%d ", proc_data[i]);
    printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Alltoall(
            proc_data,      // send_data
            1,              // send_count
            MPI_INT,        // send_type
            recv_data,      // recv_data
            1,              // recv_count
            MPI_INT,        // recv_type
            MPI_COMM_WORLD  // comm
    );

    printf("1 rank %d: ", rank);
    for (int i = 0; i < 4; i++) 
        printf("%d ", recv_data[i]);
    printf("\n");

    MPI_Finalize();
    return 0;
}
```

`output`

``` code
0 rank 0: 10 11 12 13 
0 rank 1: 20 21 22 23 
0 rank 2: 30 31 32 33 
0 rank 3: 40 41 42 43 
1 rank 0: 10 20 30 40 
1 rank 1: 11 21 31 41 
1 rank 2: 12 22 32 42 
1 rank 3: 13 23 33 43
```

**Reason**

Basically exactly what was explained above. Again using the same example
we made here the process with rank 0, after the `Alltoall` call has the
value at the 0th index of all the different processes.

------------------------------------------------------------------------

Variable size collectives
-------------------------

💡

Previously for both sending and receiving data everything occurred with
a fixed size. That is for sending each chunk was of an equal size and
likewise for receiving each chunk was of the same size. But there are
variants for all of the collective operation functions which allow
variable sizes.

### `MPI_Gatherv()`

**Intuition**

Abstractly is the same as just `MPI_Gather()` just with the added
benefit that the size of the data each process is sending can now be,
surprise, surprise, variable !

**Function signature**

``` code
int MPI_Gatherv(
    const void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void* recvbuf,
    const int* recvcounts,
    const int* displs,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
)
```

**Syntax information**

Process i sends array `sendbuf` to process `root` , which stores it in
an array `recvbuf` but in chunk `i` of size `recvcounts(i)` with an
offset `displs(i)`

**Input Parameters:**

-   `sendbuf`: Address of the send buffer (choice, significant only at
    the root).

 

-   `sendcount`: Number of elements sent to each process (integer,
    significant only at the root).

 

-   `sendtype`: Data type of send buffer elements (handle, significant
    only at the root).

 

-   `recvcounts`: Integer array (of length group size) containing the
    number of elements that are received from each process (significant
    only at the root).

 

-   `displs`: Integer array (of length group size). Entry `i` specifies
    the displacement relative to `recvbuf` at which to place the
    incoming data from process `i` (significant only at the root).

 

-   `recvtype`: Data type of receive buffer elements (handle).

 

-   `root`: Rank of receiving process (integer).

 

-   `comm`: Communicator (handle).

**Output Parameters:**

-   `recvbuf`: Address of receive buffer (choice, significant only at
    the root).

**Code example**

`test.c`

``` code
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define n 6
#define elem(rank) (rank + 1) * 10

int main(int argc, char* argv[]) {
    int rank, size; 
    int *proc_data; 
    int recv_data[n];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
        // Give process some data
        proc_data = (int*) malloc((rank + 1) * sizeof(int));
    for (int i = 0; i < rank + 1; i++) 
        proc_data[i] = elem(rank) + i;
    
        // Output the data for the process
    printf("Process %d has data: ", rank);
    for (int i = 0; i < rank + 1; i++) 
        printf("%d ", proc_data[i]);
    printf("\n");

        // Initialize recvcounts and displs
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int))
    for (int i = 0; i < size; i++) {
        recvcounts[i] = i + 1;
        displs[i] = i * (i + 1) / 2;
    }

        // Synchronize so all processes print data before Gathering 
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(
        proc_data, 
        rank + 1, 
        MPI_INT, 
        recv_data, 
        recvcounts, 
        displs, 
        MPI_INT, 
        0, 
        MPI_COMM_WORLD
    );

        // Print data in process 1
    if (rank == 0) {
        printf("Process %d has data: ", rank);
        for (int i = 0; i < n; i++) 
            printf("%d ", recv_data[i]);
        printf("\n");
    }

    free(recvcounts);
    free(displs);
    free(proc_data);

    MPI_Finalize();
    return 0;
}
```

So this is quite a big segment of code, most of it you likely understand
because its similar to just the normal gather process, but a key segment
of code is important here.

``` code
for (int i = 0; i < size; i++) {
    recvcounts[i] = i + 1;
    displs[i] = i * (i + 1) / 2;
}
```

Lets try and understand just for full clarity what `recvcounts` and
`displs` actually are and how we create them.

`recvcounts`

From the explanation this pretty clear, you just slap how much stuff a
process is sending to `root` at the index corresponding to the processes
rank. So if a process with rank 0 is sending 1 item then you would say
`recvcounts[0] = 1`

`displs`

Again mostly clear, this is more or less saying where to I slot in the
chunk of data of coming from a process. So lets say we dealt with the
data from process 0 and inserted one element at index 0, now we know to
properly append the data we want to start at 1, hence the first two
entries in `displs` must be `dspls = [0, 1]` because the first chunk
should start in the 0th index and the second in the 1st index.

### **How do I generally make these arrays ?**

Well that’s where you can probably get creative. A nice situation is of
course when you can describe the entries for both of these arrays as
functions of some easily accessible variable.

**Our example**

In our case for `recvcounts` its clear that the number of items each
process sends is just its rank + 1, and since ranks go from
`[0, ..., comm_size]` we can just utilize a basic for loop to initialize
the values.

For `displs` it becomes a bit more interesting. The number of elements
for each rank follows the sequence : `[1, 2, 3, ..., size]` and size of
the array in root as we add these elements increases as the sequence
`[1, 3, 7, ...]` . Since it increases as a sum we end up with the very
nice case where we can express the index as the sum of all previously
added elements. Or mathematically speaking

$$\texttt{displs[i]} = \sum\limits_{i = 0}^{\text{size}}\text{i} = \frac{i(i + 1)}{2}$$displs\[i\]=i=0∑size​i=2i(i+1)​

If you are still confused just play around with it a little.

**In general**

Though my assumption is that in practice this probably doesn’t happen to
often, but I would assume you can find other ways of populating the
arrays correctly, if you have just random sizes you could probably make
use of some of the previously discussed functions for everything to
share the sizes of everything else so that every process can have a
correct version of the displs and recvcount arrays.

------------------------------------------------------------------------

Non-blocking routines
---------------------

💡

Non-blocking routines as the name implies do not block the flow of
execution. This means that deadlocks cannot occur but it comes with its
own set of considerations such as **memory leaks**, **algorithmic
complexity,** successful data transfer etc.

### General theory

**Exiting before completion** Non-blocking communication allows the
calling routine to proceed even before a message has been sent or
received. Which means that you can initiate a communication operation
and continue with other tasks without waiting to finish.

**Additional buffers** are what enable non-blocking communication. These
buffers store temporary data which allows the calling routine to proceed
while the communication happens in the background.

**Memory leaks** can occur in non-blocking communication. Since
transmission is asynchronous, the user must ensure that the additional
buffers are eventually released to avoid memory leaks.

**Additional routines to test completion** should be used to avoid the
aforementioned memory leaks and allow for testing and synchronization of
your program when is necessary.

**Allow hiding communications with computations** since you can overlap
communication and computation. Meaning you can initiate a communication
task and continue performing other computations which can lead to
improved performance.

### Non-blocking routines

**Sending and Receiving \|** `MPI_ISend()` **,** `MPI_IRecv()`

These functions can send and receive in a non-blocking manner, as
opposed to `MPI_Send` and `MPI_Recv` which are blocking / synchronous.

![](https://www.mpich.org/favicon.ico)

![](https://www.mpich.org/favicon.ico)

**Waiting for completion \|** `MPI_Wait()`

This routine will block until the specified non-blocking communication
operation has finished.

**Testing Completion \|** `MPI_Test()`

This allows you to check if a non-blocking communication has finished
without blocking the calling routine. This is one of the tradeoffs where
you end up getting a higher algorithmic complexity.

### `MPI_Wait()` and `MPI_Test()`

`MPI_Test()`

We can construct a basic example where we do a non-blocking send and
receive between to processes then utilize the `MPI_Test()` function to
loop until the message comes through.

`test.c`

``` code
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, flag = 0;
    MPI_Request request;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int send_val = rank, recv_val = 0;
    if (rank == 0) flag = 1; // skip waiting for rank 0

    if (rank == 0) 
        MPI_Isend(&send_val, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    else 
        MPI_Irecv(&recv_val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);

    while (!flag) {
        printf("Rank %d is waiting\n", rank);
        MPI_Test(&request, &flag, &status);
    }

    if (rank == 1) 
        printf("Rank %d received value %d\n", rank, recv_val);

    MPI_Finalize();
    return 0;
}
```

`output`

``` code
Rank 1 is waiting
Rank 1 received value 0
```

**Result**

The output then naturally ends up being some combination of waiting and
then the final value once everything is sent through.

`MPI_Wait()`

Equivalently to the code above we can just replace the while loop
mechanism with `MPI_Wait()` to block until everything goes through and
then continue execution.

`test.c`

``` code
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, flag = 0;
    MPI_Request request;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int send_val = rank, recv_val = 0;
    if (rank == 0) flag = 1; // skip waiting for rank 0

    if (rank == 0) 
        MPI_Isend(&send_val, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    else 
        MPI_Irecv(&recv_val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);

    MPI_Wait(&request, &status);
    
    if (rank == 1) 
        printf("Rank %d received value %d\n", rank, recv_val);

    MPI_Finalize();
    return 0;
}
```

`output`

I don’t think I really have to show it, since we are blocking everything
works normally and rank 1 receives the message.

------------------------------------------------------------------------

Derived datatypes
-----------------

💡

Till now we’ve been dealing with predominantly primitive types as data
for messages, but we can also send derive datatypes. So just more
complex data structures.

### **General information**

The two main derived datatypes are structs and classes, both being
essentially just a collection of variables and or functions encapsulated
into one unit.

### `MPI_Type_create_struct()`

This method allows you to create these structs by mapping C/Fortran
structs. That is to say that we are creating structs in `C` or
`Fortran`. For example

``` code
typedef struct {
    int age;
    char name[20];
} Person;
```

**Mapping**

… and we then turn this struct into something which MPI can actually
transmit, since MPI cannot transmit just raw C structs, it needs stuff
in its own format.

**Function signature**

``` code
int MPI_Type_create_struct(
  int count,
  int *array_of_blocklengths,
  MPI_Aint *array_of_displacements,
  MPI_Datatype *array_of_types,
  MPI_Datatype *newtype
)
```

**Input parameters**

-   `count`: number of blocks in the struct (integer).

 

-   `array_of_blocklengths`: number of elements in each block (array of
    integers).

 

-   `array_of_displacements`: displacement of each block in bytes (array
    of MPI\_Aint).

 

-   `array_of_types`: datatype of each block (array of MPI\_Datatype).

**Output parameters**

-   `newtype`: new datatype created (handle).

`MPI_Type_commit()`

The `MPI_Type_commit()` function commits a derived datatype for use in
communication operations.

**Function Signature:**

``` code
int MPI_Type_commit(
    MPI_Datatype *datatype
)
```

Once a derived datatype is committed, it can be used in communication
operations, such as sending and receiving data.

**Input parameters**

-   `datatype`: Pointer to the derived datatype object (handle).

**Return Value:**

-   `MPI_SUCCESS` on success

 

-   `MPI_ERR_ARG` if the datatype argument is invalid

 

-   Other specific error codes for other failures

`MPI_Type_free()`

`MPI_Type_free()` is a function in MPI that frees a derived datatype
object that was created with `MPI_Type_create_struct()` or other
datatype creation functions.

**Function Signature:**

``` code
int MPI_Type_free(MPI_Datatype *datatype)
```

This function should be called to deallocate the memory associated with
the derived datatype object once it is no longer needed.

**Input Parameters:**

-   `datatype`: Pointer to the derived datatype object (handle).

**Return Value:**

-   `MPI_SUCCESS` on success

 

-   `MPI_ERR_ARG` if the datatype argument is invalid

 

-   Other specific error codes for other failures

### Basic example

Combining this all together you should just remember that the flow with
derived data types is

1.  Create - Map the C/Fortran type → MPI type

 

1.  Commit - Make the type available for use within the context of MPI

 

1.  Free - Free the datatype

`test.c`

``` code
#include <stdio.h>
#include <mpi.h>
#define count 2 

typedef struct {
    float value; 
    char rank;
} Person;

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
          
        // Initialize the MPI datatype requirements
        int          blocklengths[count] = {1, 1};
        MPI_Datatype types[count] = { MPI_FLOAT, MPI_CHAR };
        MPI_Datatype mpi_person_type;
        MPI_Aint     displacements[count] = {
            offsetof(Person, value),
            offsetof(Person, rank)
        };
        
        // Create the new MPI datatype
        MPI_Type_create_struct(count, blocklengths, displacements, types, &mpi_person_type);
        MPI_Type_commit(&mpi_person_type);
            
        Person person;
        if (rank == 0) {
            person.value = 3.14159;
            person.rank = 'A';
        }
        
        MPI_Bcast(&person, 1, mpi_person_type, 0, MPI_COMM_WORLD);
        printf("Rank %d: person.value = %f, person.rank = %c\n", 
            rank, person.value, person.rank);

    MPI_Finalize();
    return 0;
}
```

The key section to observe in the above code sample is specifically this

``` code
 
// Initialize the MPI datatype requirements
int          blocklengths[count] = {1, 1};
MPI_Datatype types[count] = {MPI_FLOAT, MPI_CHAR};
MPI_Datatype mpi_person_type;
MPI_Aint     displacements[count] = {
    offsetof(Person, value),
    offsetof(Person, rank)
};

// Create the new MPI datatype
MPI_Type_create_struct(count, blocklengths, displacements, types, &mpi_person_type);
MPI_Type_commit(&mpi_person_type);
```

Our struct has 1 value for two primitive types, hence the block lengths
are `{1, 1}` this would not be the case if we for example had an array
of chars like `char rank[20]` then it would be `{1, 20}`

Next we create the array of types, which I think understandably is a
float and a char.

Then we create a new MPI struct to denote the person type but in a
format understandable to MPI.

We then calculate the displacements. What this means is that a struct
has a certain memory layout in where it places is elements, to build our
MPI struct we need to know this layout, for this we use the `offsetof()`
function to get the position in memory of where `value` and `rank` are.

When then pass our MPI type and all the information along to MPI to
create our MPI formatted derived type based on the C struct.

Once we have this type we can commit it which allows use to use this
derived type now the same way you would use any other Primitive type as
we can see here.

``` code
    
Person person;
if (rank == 0) {
    person.value = 3.14159;
    person.rank = 'A';
}

MPI_Bcast(&person, 1, mpi_person_type, 0, MPI_COMM_WORLD);
printf("Rank %d: person.value = %f, person.rank = %c\n", 
    rank, person.value, person.rank);
```

And then I think quite expectedly we get the following output

``` code
Rank 0: person.value = 3.141590, person.rank = A
Rank 1: person.value = 3.141590, person.rank = A
```

**Key note**

You want to use `offsetof()` to get the position in memory of the
different values because C structs can have padding in certain places
which makes using just basic pointer arithmetic not necessarily the
safest bet to get the accurate position.

### More complex example

To solidify knowledge lets create a more complex example, previously we
used a struct which only had primitive elements, now lets introduce some
arrays aswel.

`test.c`

``` code
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#define n 4

typedef struct {
    char category[5];
    int mass;
    float coords[3];
    bool valid;
} particle;

int main(int argc, char** argv) {
    int rank;
    int tag = 99;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize particle type
    particle* particles = malloc(n * sizeof(particle));
    int blocklengths[4] = {5, 1, 3, 1};
    MPI_Datatype datatypes[4] = {
        MPI_CHAR, MPI_INT, MPI_FLOAT, MPI_C_BOOL
    };
    
    MPI_Datatype particle_type;
    MPI_Aint offsets[4] = {
        offsetof(particle, category),
        offsetof(particle, mass),
        offsetof(particle, coords),
        offsetof(particle, valid)};

    // Create particle type
    MPI_Type_create_struct(4, blocklengths, offsets, datatypes, &particle_type);
    MPI_Type_commit(&particle_type);

    // Send and receive particles
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            particles[i].mass = i;
            particles[i].valid = true;
        }

        MPI_Send(particles, n, particle_type, 1, tag, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(particles, n, particle_type, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Print received particles
        printf("Rank %d received:\n", rank);
        for (int i = 0; i < n; i++)
            printf("Particle %d has mass %d and validity %d\n", i, 
                particles[i].mass, particles[i].valid);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
```

This once again follows a mostly similar pattern that we saw before. We
start of by defining the information about the C struct. A key thing to
just not is that for the block lengths since we have arrays we have a
different number there. Aside from that its mostly the same thing.

We create the struct same as we did before, and we can then send and
receive it the same way we can any other datatype.

------------------------------------------------------------------------

Ghost communication and ghost zones
-----------------------------------

💡

Broadly speaking “ghost zones” just refer to specific processes in a
parallel computation which hold some value needed by more
central/critical processes. And one way of interacting with these ghost
zones to preform your calculation is to exchange data with them.

### Abstract concept

The general idea is quite simple. There are a lot of computations based
on the idea of updating values iteratively in a grid **based on
neighboring values**. You can imagine it as updating the entries of a
matrix for example.

Now in the simple case if we take some center entry of our grid all
neighbors are readily available and there is no real issue updating our
central value here.

The interesting case appears when we start to encounter values on the
borders of our grids. For example lets say we have a grid point on the
right border, meaning we have an upper, left and below neighbor, but
clearly no right neighbor.

**So what to do we do ?**

This is where ghost zones come into play, ghost zones are **regions at
the boundary of computations** which assist in calculating the value of
a grid point.

### An abstract example

Say you have some computation as above where you iteratively update
values in a grid, naturally you might think about splitting this grid
into chunks so you can parallelize your workload.

But in this process of splitting these chunks you might see that at the
boundaries certain grid points just lost their neighbors.

**So what do we do ?**

Well, we simply take the former neighbors, and slap them back around the
border, that way we can once again get the grid point values since we
once again have all neighbors. It should be noted that **we only compute
the grid point values for the interior points.**

By that I simply mean that we **do not** update the value of the ghost
cells that we added on. Since if you think about it, if we add the
appropriate ghost cells to all chunks, and for all chunks calculate the
interior points using the neighbors, then recombine the chunks we will
have also calculated the grid points for the ghost cells. Just in the
other chunk.

**Ok wait, what about chunks which didn’t have neighbors to begin with
?**

This is where something called boundary conditions come into effect,
these are ghost cells that we essentially just make up, so that each
grid point has the full neighbors.
