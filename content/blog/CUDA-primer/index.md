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

This post covers the very basic foundation needed to learn GPU programming and/or Parallel programming on CPUs only.

I will cover the architectural details of two of the several processors that empower the modern day computer - the CPUs and the GPUs.
By the end of this post, one should have a good understanding of the following terms - (in no particular order) chips, processors, microprocessors, cores, latency device, throughput device, clock speed, threads, processes, instructions, memory bandwidth, memory system.

## Core, Microprocessor/Processor, Chip
"A chip" is the physical semiconductor chip; it's "a physical integrated circuit" comprised of transistors, resistors, and capacitors.


A processor (here, think of CPU, the central "processing" unit) is a digital circuit that's implemented on a single or a few chips. Now, the term micro is appended to the beginning of processors to refer to the fact that it takes a single or a very few chips to implement a microprocessor. But this is more of a definition. In the context/scope of this post, consider 1 microprocessor = 1 chip.

The modern day computer is powered by processors of different types - CPUs, GPUs etc. I have also read the term processor chip, the meaning of which should be clear now.

Now, here's the thing: In older days, 1 processor used to mean 1 processing unit (single CPU based microprocessor) which changed around the year 2000 when microprocessors with more than one processing unit/CPU were introduced. Those are what are called as multi-core processors. Hence, a CPU "core" is basically a single processing unit within a processor chip that is capable of running instructions independently; and hence the modern day microprocessor with several cores is essentially a parallel processor benefitting from software that leverages parallel programming paradigms. Read that again until the terms Core, Microprocessor/Processor, Chip and the distinctions and synonymities between them are clear.

👉 If you google [intel core i9 processor](https://www.intel.com/content/www/us/en/products/details/processors/core/i9/products.html), the table there has a column # of "cores".

## Clock rate
(While I won't go into a lot of details here, it is slighlty important to have at least a rough idea of what the Clock Rate actually is.)

By definition, it's the number of clock cycles per second. The term clock cycle is what actually needs an explanation: it's the time taken (for the internal oscillator) to complete one electric signal. Now, an instruction (which could be anything like adding two numbers or writing data to the memory) can be completed by the CPU within one or more than one clock cycles and hence the clock rate is significant in determining the number of instructions that the CPU can finish in a second.

Importantly, a higher clock rate doesn't always imply a faster CPU because other factors like the CPU architecture etc. also play a role. But keeping other things constant, a higher clock rate does imply a faster CPU and we'll keep that notion in mind for the rest of this blog at least.

👉 *Back in the day, increasing the clock rate used to be the primary way to increase the processor speed until it wasn't possible anymore due to high energy consumption and heating issues which is when the hardware vendors pivoted to multicore CPUs as a means to empower high speed demanding applications. And that is why every programmer wants to be able to write ***efficient*** parallel programs in order to make their applications run faster on the modern day processors.*
<!-- 
While we discussed clock rate in the context of the processor cores (which relates to the speed of instruction execution which in turn involves several steps that I am not discussing for brevity), other chip components like the caches also have their own clock rates. So the clock rate of, say, L2 cache determines how fast its internal operations like locating and fetching the requested data take place.

(I haven't discussed the CPU architecture but have already talked of the "cache" -- sorry about it. For now, just understand that cache is one of the components on the chip that refers to a memory that's faster to access as compared to the main memory which is mounted on the motherboard.)

👉 *As an aside, if we think a little more about it, it's really the cache's clock rate that's one of the factors in determining the latency of cache access!* -->


## Types of Random Access Memory (RAM)
Before proceeding to study the processor architectures, it's worth discussing in brief two types of RAM - Static RAM (SRAM) and Dynamic RAM (DRAM). 

[RAM is just a type of memory from which any data, regardless of its position, can be accessed in the same time using its address. The abstract/conceptual way to think about memory, be it on-chip or off-chip, is as an array of bytes each having its own address that can be used to access it.]

The short story is that the hardware components used to build these two types of RAM differ from each other which makes SRAM way faster but also bulkier and way more expensive as compared to DRAM.

1. **SRAM** - The design is such that a single cell requires 6 transistors -- 6 transistors are required to store a bit which makes SRAM bulky but since these transistors hold the charge permanently as long as power is supplied, SRAM doesn't need to be refreshed making it faster. Owing to its design again, a lot of chip area is required to store one bit making it expensive.

2. **DRAM** - DRAM only requires one capacitor and one transistor to store a bit where the capacitor stores the charge representing the bit (0 or 1). Over time, this charge leaks and so DRAM cannot hold the data permanently even through the time when power is supplied requiring continous refreshes to prevent data loss. This makes DRAM slower but cheaper as less area and transistors are dedicated to storing a bit.

Why it's important to know the distinction between these two is simply because modern day processors leverage various memories of which some are designed as DRAM and some as SRAM.

## Processor Architectures
Let us now look into the architectures of the CPU and the GPU, and try to make sense of why the CPU is called a latency device and the GPU, a throughput device.

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



## Memory Bus

## Memory Bandwidth

### Why can't CPUs just have a higher memory bandwidth?


