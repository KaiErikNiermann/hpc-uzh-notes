
Anything as a Service (XaaS)
----------------------------
“Anything as a Service” describes the general category of services
related to cloud computing and remote access. Including things like
Infrastructure as a service or Containers as a service

### Cloud computing

The general idea of delivering computer systems/data storage/computing
power without direct management or intervention by the user.

**Examples**

-   Google Cloud
-   Amazon Web Services
-   Microsoft Azure
-   Openstack

### Infrastructure as a Service (IaaS)
A type of service which provides users with **virtualized servers**,
which are large computers which can run multiple OS’s and applications
as if they were separate.

**Data center**
Physical location which houses servers. Can be located anywhere in the
world as long as there is internet.

Are expensive to build and operate but they offer economies of scale.

⇒ Cost per unit of computing power decreases the more units there are.

### Containers as a Service
A type of service which provides users with **containers**, which are
**isolated environments** that can **run applications** without
interfering with each other.

Similar to virtual machines, but more lightweight and efficient.

⇒ Share same operating kernel but only contain necessary libs and deps
for application.

⇒ Can be rented by cloud provider. Same principle as IaaS but with more
flexibility and scalability.

### Containers in HPC
See above definition

**Examples**

-   Docker
-   Shifter
-   Singularity

------------------------------------------------------------------------

Virtual Machine
---------------
Compute resource that uses software instead of a physical computer to
run programs and deploy apps. Generally a single computer can run one or
more virtual “guest” machines.

### General usage
In the following I describe the general process for using a VM on the
cloud

1.  Open the dashboard to your cloud provider
1.  Find the specific service that allows you to create virtual
    machines, for AWS for example this would be the EC2 service
1.  Create a new virtual machine instance using this service
1.  Select the hardware you would want this virtualized system to have,
    generally this means
    1.  Operating system : Ubuntu, Debian, Fedora etc.
    1.  CPU cores
    1.  RAM amount
    1.  Storage amount

1.  Generate an SSH key pair as a means of remotely connecting with the
    VM
1.  Connect with the VM via SSH using something akin to the following
    command

``` code
ssh -i my-key.pem username@address 
```

### Snapshot

A snapshot refers to the captured state of a VM at a specific point in
time with all its data. Snapshots are just normal VMs with the idea
being that you can utilize this ability to save the state for various
purposes like

-   Creating multiple of the same VMs ( without having to manually setup
    new ones )
-   Creating backups of VMs in certain states
-   Many more benefits and reasons
### Organization

1.  At the very top you have the Kernel Based Virtual machine called
    `guest`

 

1.  This interfaces with its own file system, drivers and its own kernel

 

1.  This kernel has accessed to the virtualized hardware, here this is
    represented as it having
    $N$N vcpus or virtual cpu cores

 

1.  The CPU cores interface with the kernel of the host machine,
    specifically kvm.ko portion which handles this interface

 

1.  QEMU is the main piece of software that actually emulates or allows
    this virtualized hardware to work

![](Cloud%20and%20Containers%202edfce13672a46fb874887d46de25ee8/Untitled.png)

1.  The kernel of the host machine then interfaces with the bare metal
    (not virtualized) hardware

------------------------------------------------------------------------

Containers
----------

💡

Containers are isolated computing environments that usually consist of
an operating kernel and the exact dependencies to run a specific
application.

### Why a container

The primary usage of containers is when you want to run a specific
application on an incompatible operating system. Since containers are
meant to perfectly encapsulate everything to run a specific application
they are very well served for this purpose.

**Why not just set stuff up directly ?**

Sometimes its just incompatible and other times it can be alot harder
and more annoying to set something up directly on a computer vs a
container. Once you have it setup on a container it also becomes alot
more portable. Meaning you can easily run a container on various
different operating systems as long as they have docker installed.

**Example**

We want to run Ubuntu code with specific deps on an HPC with an
incompatible operating system.

Solution ⇒ Create a container to encapsulate the runtime requirements
for this application and just install docker on the remote HPC.

### What about VMs

They do work aswell for this purpose, are just far less efficient when
you want to run just 1 application because they are generally speaking a
full fledged operating system with a bunch of unnecessary dependencies

**Efficiency of containers**

Containers are far more lightweight since they generally only have the
kernel and the exact dependencies needed for an application. So they are
much more akin to just a regular executable then an entire virtualized
operating system.

### Diagram comparison

![](Cloud%20and%20Containers%202edfce13672a46fb874887d46de25ee8/Untitled%201.png)

------------------------------------------------------------------------

Docker basics
-------------

💡

Docker is one of the many container services. It works by encapsulating
everything that is required to run an application in a series of steps
called a **Dockerfile** which describe the behavior of a **Docker
Image.** This Dockerfile is executed by the docker engine, an instance
of this execution is called a **Container**.

### Basic DIY docker image

Remember that the general idea behind docker is that you are basically
just describing the steps you *would* do if you would setup everything
on your preferred system. So lets look at a basic Dockerfile.

``` code
FROM python:latest

WORKDIR /app

COPY hello.py .

CMD ["python", "hello.py"]
```

`hello.py`

``` code
print("Hello from docker!")
```

|           |                                                                                                                      |
|-----------|----------------------------------------------------------------------------------------------------------------------|
| `FROM`    | What do you want to use as your base kernel ? Do you want this kernel bundled with stuff for specific applications ? |
| `WORKDIR` | Where inside the container do you want everything to happen                                                          |
| `COPY`    | Do you want to put anything from your host machine into the container                                                |
| `CMD`     | Once you create an image ( running instance of the container ) what do you want to execute                           |

One very important thing is the type of image which you pull to use as
your base. You can use something like the latest version of ubuntu (
`ubuntu:latest` ) but often times the most intelligent choice is to pull
an image that comes prepackaged with all the setup you need for a
specific application. In our case we use a docker image that comes with
everything you need to execute a Python application.

**Example: Efficient Docker images**

Any further setup you can then do after the fact, for example if you are
developing in rust you might want to use `rust:latest` or if you want an
image that uses a small base kernel something like `rust:slim` which
aims to have all the things you need to execute a rust application but
also using a very minimal kernel to keep everything space efficient.

**Building the container**

You then want to build the container with the command

``` code
docker build --tag container_name location/to/Dockerfile
```

So in our case assuming you are in the same directory as the Dockerfile
its just

``` code
docker build -t hello_python . 
```

### Basics of working with docker containers

There are a few important things you should know about how to interact
with containers ( running images ).

**Common commands**

What are my currently running images ? - `docker ps`

How do I stop a container ? - `docker rm id_or_name` (to see this
information `docker ps`)

**How do I go into the container ?**

You can execute the shell belonging to the container using a docker
command

``` code
docker run -it hello_python /bin/bash
```

It then brings you into the container where you can explore a bit

`inside container`

``` code
root@6faeacc5c525:/app# ls
hello.py
root@6faeacc5c525:/app#
```

To get out simply type `exit` which brings you back to your own
terminal.

**How do I remove my container after running it ?**

Some containers, especially ones using bare kernels ( e.g.
`ubuntu:latest` ) dont immediately close unless you explicitly tell them
to. So you want to add the `--rm` flag to the run command which means
that the container will automatically stop once you are no longer
interacting with it.

``` code
docker run -it --rm hello_python /bin/bash
```

**I want to exchange files with the docker container while its running**

This is a decently common thing. Most often its that you want some
specific file generated inside the docker container on your host machine
but sometimes you might also want the opposite.

`-v` says that you want the the current directory to be linked to the
`/app` directory in the container.

new `hello.py`

``` code
import os
print(os.listdir('.'))
```

`command`

``` code
docker run -v $(pwd):/app hello_python
```

`output`

``` code
['text.txt', 'Dockerfile', 'hello.py']
```

This likewise leads to the reverse, if we change the python file to
create a file and write to it. Then because our current dir and the one
in the container are linked the generated file appears in our directory.

new `hello.py`

``` code
with open('hello.txt', 'w') as f:
    f.write('Hello, world!')
```

`ls` in our dir

``` code
Dockerfile  hello.py  hello.txt  text.txt
```

So in general with `-v` you are creating a mutual connection between a
directory on the host machine and in the docker container/image. In an
actual practical setting what you might more sensibly do is create some
specific directory in your image that is separate and meant just for
specific files you want just to maintain the isolation.

But I hope this at least demonstrates some of the fundamental ideas
behind getting files.
