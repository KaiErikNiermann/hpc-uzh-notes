
------------------------------------------------------------------------

Overview Hadoop and MapReduce
-----------------------------

💡

Hadoop is a an collection of open-source **software utilities** used for
**distributed storage** and **processing large datasets**. MapReduce is
the primary **programming model** that **Hadoop implements** for
processing parallelizable problems across large datasets. The other
major component of Hadoop is its distributed file system HDFS which is
what implements the distributed storage aspect.

### History

|      |                                                |
|------|------------------------------------------------|
| 2004 | Google releases white paper on MapReduce       |
| 2006 | Hadoop development starts                      |
| 2011 | Version 1 of Hadoop is released                |
| 2012 | Version 2 of Hadoop is released alongside YARN |
| 2017 | Version 3 of Hadoop is released                |

So essentially MapReduce was described by google as a programming model
and this was then later implemented by Hadoop.

### Hadoop Components

There are a few central aspects to Hadoop

**Client**

The user interacts with Hadoop through the **Hadoop client** which
submits jobs to the Hadoop cluster.

**HDFS**

Stands for Hadoop Distributed File system which is as the name implies
the implementation for a distributed storage system. This being a file
system simultaneously mounted on multiple servers.

**NameNode**

This is the central node (machine) that **manage file system namespace**
and **regulates file access** by clients.

**YARN**

Stands for Yet Another Resource Negotiator is a **resource management
layer** in Hadoop that schedules and allocates resources to applications
running on the Hadoop cluster. It splits up resources managing into two
tasks

1.  Global Resource Manager - Which manages the global resources

 

1.  Application Resource Manager - Which manages individual applications
    which usually means currently running jobs

**Data Node**

Is a node which is responsible to serve **read and write requests** from
the client and its where the HDFS actually stores its data.

**Node Manager**

Is responsible for **managing resources** (CPU, memory, disk, network)
of a single node in a Hadoop cluster.

**MapReduce**

Is the programming model which Hadoop implements as a means for
processing large data sets with parallel distributed algorithms on a
cluster. Its composed of two main components

1.  Map procedure - filtering and sorting of the dataset

 

1.  Reduce - summary operation on the dataset

------------------------------------------------------------------------

Standard File System
--------------------

💡

The standard file system most Linux distributions use is called **ext4**
and for windows its **NTFS. File metadata** is the data which describes
the properties of files such as the creation date, name, date, size,
author etc.

### ext4 - Linux

Provides support for large files and volumes.

Has faster disk accesses and is more resilient to power outages and
system crashes compared to its predecessor ext3.

**File metadata**

The meta for ext4 is stored on the drive itself. It also supports
several drives that can be combined to look like a single drive.

### **NTFS - Windows**

Similarly provides support for large files and volumes.

Has better security features than its predecessor, FAT32.

**File metadata**

Same as for ext4, it stores metadata on the drive and supports **spanned
volumes** this just being two volumes (drives) which were combined.

### Distributed Network Storage

There are various different protocols that can be used to share files
over a network.

**Network File System (NFS)**

Commonly used in Unix-based systems. It allows users on a client
computer to access files over a network as if those files were on the
local computer.

**Samba (SMB/CIFS)**

Commonly used to share printers over a network this protocol allows file
sharing between Unix and Windows based systems.

**Apple Filing Protocol**

Commonly used on Mac OS X computers. It allows for network file sharing
between apple computers.

------------------------------------------------------------------------

RAID and RAID Levels
--------------------

💡

RAID (redundant array of independent disks) is a way of storing the
**same data** in **different places** on **multiple drives**. RAID is
classified into different levels which focus on different benefits of
this multiple storage approach.

Each RAID level can be broken down into 4 main aspects, what happens to
the data, and as a result what is the efficiency, performance and
reliability.

### Striping

Striping is the process of segmenting logically sequential data into
blocks and spreading the blocks across multiple storage devices.

### RAID Level 0 - Striping (high performance)

RAID 0 has a focus on higher performance at the cost of reliability

**Data distribution**

In RAID Level 0 we stripe the data, without any mirroring so we get no
fault tolerance or redundancy benefit.

**Storage efficiency**

Since we are basically just treating both drives as contiguous storage
its 100% , there is no overhead from mirror.

![](MapReduce%207105dd071cf8442b9635aa21404686a1/Untitled.png)

**Performance**

This is the main benefit of RAID 0, because our data is split amongst

$n$n drives we are able to read and write files

$n$n times faster than if it where just on a single drive.

**Reliability**

Since data is split amongst drives the failure of a single drive could
be catastrophic for the entire array of RAID 0 drives. So Level 0 has
overall low reliability.

### RAID Level 1 - Mirroring (high reliability)

RAID 1 has a focus on higher reliability but at the cost of performance

**Data distribution**

Here the data are copied identically on two or more disks, this provides
redundancy and fault tolerance.

**Storage efficiency**

50% since half of the drives storage is used for data and the other half
for mirroring the data.

![](MapReduce%207105dd071cf8442b9635aa21404686a1/Untitled%201.png)

**Performance**

**High read** speeds as the same data is accessible from multiple
different drives. **Low write** speeds as we obviously need to write the
data to all different drives to maintain the mirror state.

**Reliability**

High, because we can recover the data from the surviving disk if one or
more disks fail. This is the main purpose of mirroring.

### RAID Level 5 - Distributed parity (medium)

RAID 5 attempts to compromise between reliability and performance

**Data distribution**

Similar to RAID 0 we stripe the data amongst multiple disks, but in
addition we include parity information which provides fault tolerance
and error correction. But only to a limited extent.

**Storage efficiency**

Around 66% with the other 33% going to storing the parity information.

![](MapReduce%207105dd071cf8442b9635aa21404686a1/Untitled%202.png)

**Performance**

Moderate read and write speeds as you get the benefit of multiple drives
but you have to calculate the parity to ensure data integrity which
slows things down.

**Reliability**

Moderate reliability as data can be reconstructed if a limited number of
drives fail but a catastrophic failure is still unrecoverable.

------------------------------------------------------------------------

File servers
------------

💡

A file server is a computer **attached to a network** that provides a
location for **shared disk access.**

### Key aspects of File servers

**CPU and Memory**

Fundamentally its still a computer so it has regular computer hardware
to process requests from client and manage the file system.

**Multiple Drives**

A file server obviously also has multiple drives which can be combined
as a disk array with something like RAID.

**RAID**

Most file servers have their drives in a raid configuration either to
provide a performance improvement or data redundancy in case of failure.

**JBOD (Just a Bunch Of Disks)**

JBOD is the same technology you can use on windows to combine drives.
Instead of spreading the data amongst multiple drives each drive still
contains its own files its just when one gets file subsequent files go
on the next drive.

**File I/O**

File servers are optimizes specifically for fast file I/O operations
that is, high read/write speeds.

**Network Performance**

This is another key factor, a file server must have a good network
performance to handle multiple requests and provide fast data access to
clients.

------------------------------------------------------------------------

Lustre Distributed File System
------------------------------

💡

Lustre is a type of **distributed file system** that can handle
large-scale **parallel cluster computing**.

### File provider

In the most basic sense Lustre is designed like regular Unix-like file
system in the fact that it provides a standard interface for accessing
and manipulating files.

### Lustre architecture

Lustre has 2 main types of servers

**Metadata servers (MDS)**

These store the **namespace metadata**, which includes things like
filenames, directories, access permissions and file layout

**Object storage server (OSS)**

These are nodes that store file data on one or more **object storage
target** (OST) devices. These OSTs contain one or more **OST objects**
which contain the files data. OST objects are fixed size chunks of data
striped across multiple OST’s to which data is written to in fixed-size
chunks and striped across objects. So we got that double striping
action.

![](MapReduce%207105dd071cf8442b9635aa21404686a1/Untitled%203.png)

### File assignments

Lustre uses a hashing algorithm to ensure a balanced distribution of
files across the server. The balanced distribution ensures high
throughput with concurrent File I/O because people can concurrently
access different parts of a file from separate drives.

### Single file access

The fact the Lustre employs striping means that, same as for RAID, high
throughput due to being able to utilize multiple drives for R/W
operations as opposed to R/W just from a single drive.

### Aggregate performance

Due to this aspect of employing both striping and a balanced
distribution of file data means that Lustre can scale up to hundreds of
petabytes of storage or hundreds of GB of I/O throughput.

------------------------------------------------------------------------

Object storage
--------------

💡

In object storage data are managed as **objects** these objects usually
consist of : the **data,** some amount of **metadata**, and a **globally
unique identifier**. An example of an implementation of object storage
is the software-defined storage platform **ceph**.

### Ceph

Ceph is a software-defined distributed object storage system. Meaning it
enables to interaction with an object-based file system while also being
specifically designed to be applied in a distributed/cluster
environment.

**Objects are duplicated**

One of the primary techniques ceph uses to ensure data durability is
**replication**. Which means that the data - objects in this case - are
duplicated and exist on multiple servers for fault tolerance.

This fact also enables the aforementioned **distributed operation** of
data without any single point of failure.

The **replication** and **distributed operation** are handled by the
Ceph Object Storage Daemon ( ceph-ods ).

**Number of replicas**

In ceph you can manually configure the number of object replicas. This
value is usually set to 3. Which means that each object is stored three
times across the cluster to ensure data safety. Lower values than 3 mean
that failure is more likely to effect your data.

**Separate metadata**

The metadata of objects in ceph is handled on **metadata servers (
ceph-mds )** that maintain and broker access to inodes ( data structures
that describe the objects ) and directories inside the CephFS
filesystem.

**SSD’s** can be used for faster access times to the inodes.

**SSD/HDD node hierarchy**

Ceph supports a **tiered storage** approach where more frequently
accessed data can be stored on faster drives such as SSD’s whereas less
frequently accessed data can be stored on slower HDDs

**Monitoring**

Ceph also continuously maintains **maps** of the cluster state which are
critical for the different Ceph daemons to coordinate with eachother.

This is done via the **ceph monitor ( ceph-mon )** which is also
responsible for managing authentication between daemons and clients.

### Ceph Layout

![](MapReduce%207105dd071cf8442b9635aa21404686a1/Untitled%204.png)

------------------------------------------------------------------------

Distributed File System
-----------------------

💡

As mentioned before distributed file systems are simply file systems
which **manage data across several machines** and communicate generally
using some **network protocol**. A popular example of this type of file
system would be the **Hadoop distributed file system**.

### Hadoop Distributed File System (HDFS)

HDFS is a distributed file system that **manages files** across
**multiple machines** in a cluster.

Its one of the major parts of the **Apache Hadoop** framework which
provides a way of working with **large and diverse datasets** using
distributed and parallel computing.

**Large Files**

HDFS is optimizes for working with **very large files**. *Similar* to
data striping HDFS **splits data into small blocks** and then
distributes these amongst the drives in the cluster. This allows for
high I/O throughput because it allows one to leverage multiple drives.

**Streaming Data Access**

HDFS is optimized for streaming access, which is when you access a file
in a **sequential and continuous** manner this being due to the fact
MapReduce or YARN applications access data in a linear fashion, so
having streaming access be optimized means a faster execution time with
these utilities.

**Data Locality**

HDFS moves any **computations to the data** to **eliminate any latency**
that would come with the conventional approach of fetching a piece of
data and then preforming some computation. Specifically network latency
is what is being reduced.

**Scalability**

Because HDFS is fundamentally meant for distributed systems its
inherently scalable. You can **add more nodes** to the system as your
data grows.

### HDFS Fault Tolerance and Injection

**Fault Tolerance**

HDFS provides fault tolerance by **replicating data** across multiple
nodes. This removes any single point of failure.

**Fault Injection**

HDFS provides a method of artificially injecting faults as a means of
testing if your system can recover from specific error conditions.

### Apache Hadoop layout

![](MapReduce%207105dd071cf8442b9635aa21404686a1/Untitled%205.png)

------------------------------------------------------------------------

MapReduce
---------

💡

MapReduce is a Java-based distributed execution framework within the
Apache Hadoop ecosystem. It has to main components **Map** in which data
is split between parallel processing tasks and some transformation logic
is applied to each split chunk. **Reduce** which handles aggregating the
data from the Map set.

### MapReduce basics

**Key-value pairs**

In MapReduce the input to both the Map and Reduce procedure is
represented as **key-value** pairs.

**Mapper**

Fundamentally the mapper takes some set of data and converts into to
another set of data where the individual elements are broken down into
key-value pair tuples.

**Combiner / Mini-reducer**

This is an **intermediary procedure** that takes combines similar key
from the mapper into one and then passes these combined keys onto the
partitioner.

An important note is that the Combiner does **not always run.**

**Partitioner**

The partitioner is another intermediary procedure, it takes the output
from the combiner ( or Mapper ) and generates a number of aggregated
**key-value** pair partitions corresponding to the number of Reduce
tasks.

**Shuffle and Sort**

The Shuffle and sort procedure aggregates all the values from the same
keys from all generated partitions and sorts these values then passes
them on to the reducer.

**Reducer**

The reducer procedure takes the output of modifier tuples from the
mapper and generates a set of even smaller aggregated tuples.

------------------------------------------------------------------------

SSH Components
--------------

💡

**S**ecure **SH**ell is a network protocol for operating network
services (e.g. remote VMs) **securely.**

### Keys

In SSH both the client and the server have public and private keys.

**Servers public key**

This is used to authenticate the server from the client side.

The benefit of this is that it **avoids** a **man in the middle** attack
where an attack could potentially intercept an attempt at a connection.

**First connection**

When you first connect to a server via SSH the servers public key is
saved on your machine and used for any subsequent connection attempts.
This public key is saved file `.ssh/known_hosts` .

**Client public key and private key**

Likewise when the client tries to initiate a connection with the server
it passes along its public key which is saved into
`.ssh/authroized_keys` to express that this user is a valid.

The private key; which stays on the users machine; is used to verify the
identity of the user.

How this works is that

1.  During the initial connection the SSH server will encrypt a random
    piece of data and send it to the client.

 

1.  The client then uses the private key on their machine to decrypt the
    cyphertext and sends this now decrypted text back to the SSH client.

 

1.  If the SSH client can verify that this decrypted text is the same as
    the random piece of data it initially encrypted using the clients
    public key then it has verified the identity of the client and
    establishes a successful connection.

### Session keys

After the client has properly identified themselves and established a
connection session keys are exchanged for encrypting/decrypting
communication which is faster than using the public and private key.

### Tunneling

Communication occurs over a secure tunnel using these session keys
meaning the data is secured during transmission.

### Port forwarding

Port forwarding is the process of redirecting the communication from one
address and port number combination to another while packets are
traversing a gateway.

**Main usage**

The main usage is to make services (e.g. files) residing on a protected
internal network available to hosts on the opposite side of the gateway
(external network)

![](MapReduce%207105dd071cf8442b9635aa21404686a1/Untitled%206.png)
