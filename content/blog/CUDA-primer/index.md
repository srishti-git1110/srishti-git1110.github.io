---
title: "[WIP] Learning CUDA Programming: A Primer"
layout: post
date: 2024-03-31
tags:
    - CUDA, Computer Architecture, Optimization, GPU Programming, Parallel Processing
description: "A friendly primer to learning CUDA programming"
draft: false
mathjax: true
---

I want this post to be an entry point for anyone willing to learn GPU programming without a formal background in compsci/comp arch (I myself belong to this category so sorry in advance for any errors that I might make, and please email me to get them corrected). 

I will cover the architectural details of two of the several processors that empower the modern day computers - the CPUs and the GPUs.
By the end of this post, the reader should have a good understanding of the following terms - (in no particular order) chips, processors, microprocessors, cores, latency device, throughput device, clock speed, threads, processes, instructions, memory bandwidth, memory system.

## Why are we here?
A few months back, I used to visit the PyTorch forums almost daily. When I just began reading over there, terms "host memory", and "device memory" confused me (I haven't majored in CS and hence the unfamiliarity with basics).

A quick lookup did the deal but a similar thing happened as I started reading the PMPP book with a goal to learn parallel programming on GPU "devices" ;). I was able to get a hold of the code fairly easily (because C++ doesn't haunt - I like it more than Python which is what ML Researchers like myself usually use), but the jargons weren't entirely clear at once and hence this primer post.

## Core, Microprocessor/Processor, Chip
"A chip" is the physical semiconductor chip; it's "a physical integrated circuit" comprised of transistors, resistors, and capacitors.


A processor (here, think of CPU, the central "processing" unit) is a digital circuit that's implemented on a single or a few chips. Now, the term micro is appended to the beginning of processors to refer to the fact that it takes a single or a very few chips to implement a microprocessor. But this is more of a definition. In the context/scope of this post, consider 1 microprocessor = 1 chip.

The modern day computer is powered by processors of different types - CPUs, GPUs etc. I have also read the term processor chip, the meaning of which should be clear now.

Now, here's the thing: In older days, 1 processor used to mean 1 processing unit (single CPU based microprocessor) which changed around the year 2000 when microprocessors with more than one processing unit/CPU were introduced. Those are what are called as multi-core processors. Hence, a CPU "core" is basically a single processing unit within a processor chip that is capable of running instructions independently; and hence the modern day microprocessor with several cores is essentially a parallel processor benefitting from software that leverages parallel programming paradigms. Read that again until the terms Core, Microprocessor/Processor, Chip and the distinctions and synonymities between them are clear.

👉 If you google [intel core i9 processor](https://www.intel.com/content/www/us/en/products/details/processors/core/i9/products.html), the table there has a column # of "cores".

## Clock rate

## Types of Random Access Memory (RAM)
Before proceeding to study the processor architectures, it's worth discussing in brief two types of RAM - Static RAM (SRAM) and Dynamic RAM (DRAM).

1. **SRAM** - 
2. **DRAM** - 

## Processor Architectures (Hardware Design)
Let us now look into the architectures of the CPU and the GPU, and try to make sense of why the CPU is called a latency device and the GPU a throughput device.

### CPU
Let us first look at what a chip with 4 cores looks like:

![A CPU chip](cpu-chip.png#center)
*[Image source](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/design)*
A slight but important correction to note here is that while according to the figure above, the DRAM (sometimes simply referred to as RAM or system memory) appears to be located on the chip, it's not acutally the case. The DRAM is a separate hardware entity that's mounted on the motherboard.

Next, pay attention to *how* the chip area is divided among the different components.  Note also the multiple levels of cache memories present on the chip (purple and blue) -- they help to reduce the latency by decreasing the required amounts of high latency memory (DRAM) accesses. Ah, I went too fast here! To further clarify, since 


Now let's zoom into a single core:

// figure

A few main components are shown in the core above:
1. A few very powerful ALUs (Arithmetic Logic Units): A few of them are present on each core and it's where the actual computation happens. Each ALU in itself is very powerful and capable of completing a computation in a very few clock cycles; and hence are geared towards the low latency paradigm.

2. A Control Unit (CU): A major area is occupied by this component as its two main functions help greatly in reducing the latency - branch prediction and data forwarding. I won't elaborate on these two but the takeaway is that each CPU core features a sophisticated CU which again serves the low latency design.

3. A big L1 cache: Ofcourse, much smaller than the DRAM, a significant portion on each is dedicated to the L1 cache again to reduce the latency.

### GPU
From the same [source](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/design), here's what a GPU chip looks like:

![A GPU chip](gpu-chip.png#center)

As can be seen, the major chip area is now occupied by the green boxes which are the components where the computation takes place (I am refraining from giving a name to the green boxes just yet but yes they are the equivalent of the green ALUs we saw in the CPU). But what's also worth noting is that each green box is now much more smaller than 1 single ALU in the CPU core -- this actually reflects the real scenario that a single of these units on the GPU is much much less powerful than a single ALU and hence has a much longer latency.

The L1 caches and the control occupy much lesser chip area.




## Instructions, Threads, Processes

## Memory Bandwidth

### Why can't CPUs just have a higher memory bandwidth?


