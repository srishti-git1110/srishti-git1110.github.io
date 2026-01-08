---
title: "[WIP] CPU, GPU and stuff!"
layout: post
date: 2025-11-30
tags:
    - Computer Architecture, Optimization, Parallel Processing
description: ""
draft: false
mathjax: true
---

Some beginner stuff!

## Core, Microprocessor/Processor, Chip
"A chip" is the physical semiconductor chip; it's "a physical integrated circuit" comprised of transistors, resistors, and capacitors.


A processor (here, think of CPU, the central "processing" unit) is a digital circuit that's implemented on a single or a few chips. Now, the term micro is appended to the beginning of processors to refer to the fact that it takes a single or a very few chips to implement a microprocessor. But this is more of a definition. In the context/scope of this post, consider 1 microprocessor = 1 chip.

The modern day computer is powered by processors of different types - CPUs, GPUs etc. I have also read the term processor chip being used, the meaning of which should be clear now.

Now, here's the thing: In older days, 1 processor used to mean 1 processing unit (single CPU based microprocessor) which changed around the year 2000 when microprocessors with more than one processing unit/CPU were introduced. Those are what are called as multi-core processors (The last section touches upon why this shift happened in the hardware industry).

Hence, a CPU "core" is basically a single processing unit within a processor chip that is capable of running instructions independently; and hence the modern day microprocessor with several cores is essentially a parallel processor benefitting from software that leverages parallel programming paradigms. Read that again until the terms Core, Microprocessor/Processor, Chip and the distinctions and synonymities between them are clear.

👉 If you google [intel core i9 processor](https://www.intel.com/content/www/us/en/products/details/processors/core/i9/products.html), the table there has a column # of "cores".

## Clock rate
By definition, it's the number of clock cycles per second. The term clock cycle is what actually needs an explanation: it's the time taken (for the internal oscillator) to complete one electric signal. Now, an instruction (which could be anything like adding two numbers or writing data to the memory) can be completed by the CPU within one or more than one clock cycles and hence the clock rate is significant in determining the number of instructions that the CPU can finish in a second.

Importantly, a higher clock rate doesn't always imply a faster CPU because other factors like the CPU architecture etc. also play a role. But keeping other things constant, a higher clock rate does imply a faster CPU and we'll keep that notion in mind for the rest of this blog at least.

<!-- 👉 *Back in the day, increasing the clock rate used to be the primary way to increase the processor speed until it wasn't possible anymore due to high energy consumption and heating issues which is when the hardware vendors pivoted to multicore CPUs as a means to empower high speed demanding applications. And that is why every programmer wants to be able to write ***efficient*** parallel programs in order to make their applications run faster on the modern day processors.* -->
<!-- 
While we discussed clock rate in the context of the processor cores (which relates to the speed of instruction execution which in turn involves several steps that I am not discussing for brevity), other chip components like the caches also have their own clock rates. So the clock rate of, say, L2 cache determines how fast its internal operations like locating and fetching the requested data take place.

(I haven't discussed the CPU architecture but have already talked of the "cache" -- sorry about it. For now, just understand that cache is one of the components on the chip that refers to a memory that's faster to access as compared to the main memory which is mounted on the motherboard.)

👉 *As an aside, if we think a little more about it, it's really the cache's clock rate that's one of the factors in determining the latency of cache access!* -->

Let's see an example demonstrating why the clock rate matters. Consider a machine with the following specs:

- #Processor chips: 2
- #cores per chip: 9
- Clock rate of a core: 2.9 Ghz
- floating point unit ops: 8 double precision ops (including FMA, fused multiply and add) per core per cycle

If we now want to calculate the peak number of floating point ops that can be performed by this machine per cycle, we'd do:

2.9 x 2 x 9 x 16 = 835 GFLOPS

If we could hence build a machine with a higher clock rate, that directly translates to better "peak" performance. Utilizing it as much as possible depends on one's skills. :)



## Types of Random Access Memory (RAM)
Before proceeding to study the processor architectures, it's worth discussing in brief two types of RAM - Static RAM (SRAM) and Dynamic RAM (DRAM). 

[RAM is just a type of memory from which any data, regardless of its position, can be accessed in the same time using its address. The abstract/conceptual way to think about memory, be it on-chip or off-chip, is as an array of bits  each having its own address that can be used to access it.]

The short story is that the hardware components used to build these two types of RAM differ from each other which makes SRAM way faster but also bulkier and way more expensive as compared to DRAM.

1. **SRAM** - The design is such that a single cell requires 6 transistors -- 6 transistors are required to store a bit which makes SRAM bulky but since these transistors hold the charge permanently as long as power is supplied, SRAM doesn't need to be refreshed making it faster. Owing to its design again, a lot of chip area is required to store one bit making it expensive.

2. **DRAM** - DRAM only requires one capacitor and one transistor to store a bit where the capacitor stores the charge representing the bit (0 or 1). Over time, this charge leaks and so DRAM cannot hold the data permanently even through the time when power is supplied requiring continous refreshes to prevent data loss. This makes DRAM slower but cheaper as less area and transistors are dedicated to storing a bit.

Why it's important to know the distinction between these two is simply because modern day processors leverage various memories of which some are designed as DRAM and some as SRAM. For instance, the main memory (RAM as we call it) is actually DRAM (and not SRAM because 💸).  

## Processor Architectures
Let us now look into the architectures of the CPU and the GPU, and try to make sense of why the CPU is called a latency device and the GPU, a throughput device.

### CPU
Let us first look at what a chip with 4 cores looks like:

![A CPU chip](cpu-chip.png#center)
*[Image source](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/design)*
A slight but important correction to note here is that while according to the figure above, the DRAM (sometimes simply referred to as RAM or system memory) appears to be located on the chip, it's not acutally the case. The DRAM is a separate hardware entity that's mounted on the motherboard.

Next, pay attention to *how* the chip area is divided among the different components.  Note also the multiple levels of cache memories present on the chip (purple and blue) -- they help to reduce the latency by decreasing the amount of high latency memory (DRAM) accesses. 


Now let's zoom into a single core:

// figure

A few main components are shown in the core above:
1. A few very powerful ALUs (Arithmetic Logic Units): A few of them are present on each core and it's where the actual computation happens. Each ALU in itself is very powerful and capable of completing a computation in a very few clock cycles; and hence follows the low latency design philosophy.

2. A Control Unit (CU): A major area is occupied by this component as its two main functions help greatly in reducing the latency - branch prediction and data forwarding. A larger CU again serves the low latency design.

3. Caches: A significant (i.e. significantly large when compared to GPU cache size) portion of each core is dedicated to on-chip caches as caches reduce the latency by reducing the amount of RAM accesses required. DRAM accesses are large latency accesses in that they take a lot more clock cycles to finish as compared to on-chip caches and hence cause stalling if the processor needs to access the DRAM frequently. 

<u> A bit about caches </u>

- Temporal Locality: Let's say the processor needs acess to a datum value that's not in any of the caches yet and hence needs to be fetched from the memory. When this datum is fetched from the memory for the processor to be able to use it, it's also loaded into the cache. This serves what's known as "temporal locality" which, in essence, means that the processor has the tendency to use the same datum value again in near future and hence it's valuable to store it in the cache. 

- Spatial Locality: The way data is loaded in the cache(s) is in the granularity of what's knows as <u>cache lines</u>. This means it's not just one particular datum that's loaded in the cache, but a whole "cache line" of contiguous data values is loaded along with it. So, let's say the cache line size for a particular processor is 128 bytes (which is the case for my machine - Apple M4), that's equivalent to loading a line of 4 fp32 floats. Consequently, cache eviction, which refers to removing certain data from the cache in order to feed in new data, also happens in cache lines. The question here is why we would do this. It's because this serves "spatial locality" which states that the processor is likely to use in future the data values near the current datum it's required to use.

Let's see this in action:



Understanding caches and the way they work is really important for performance engineering. One example of that is [here](https://srishti-git1110.github.io/blog/matmul-cpu/).


4. SIMD units: 







### GPU
From the same [source](https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/design), here's what a GPU chip looks like:

![A GPU chip](gpu-chip.png#center)

As can be seen, the major chip area is now occupied by the green boxes which are the components where the computation takes place. But what's also worth noting is that each green box is now much more smaller than 1 single ALU in the CPU core -- this actually reflects the real scenario that a single of these units on the GPU is much much less powerful than a single ALU and hence has a much longer latency.

The L1 caches and the control occupy much lesser chip area.



## Memory Bus

## Memory Bandwidth

### Why can't CPUs just have a higher memory bandwidth?


## Why parallelization?
Until 2004, Moore's law, that states that the number of transistors on an integrated circuit doubles roughly every two years, was full in action. This along with Dennard scaling led to faster chips, with increased clock rates at a constant power and cost requirement, coming in about every 18 months or so. And hence, optimization of sequential programs, though useful wasn't really the focus as every two years or so, the new hardware guaranteed better performance for the exact same (unoptimized sequential code) anyways. 

What changed then?
While Moore's law kept allowing for more transistors per chip (smaller in size and more in number), Dennard scaling broke. Meaning we could no longer get increased clock rates on a single core due to issues like charge leakage, heat dissipation etc. and hence hardware engineers couldn't fit more transistors onto a single core which further meant programmers could no longer depend on the hardware in order to get speedups to their programs. What the hardware industry then transitioned to is the multicore cpu chips we know today. This required the programmers to write parallel code so as to be able to continue enjoying speedups moving beyond a single cpu core. In short, moving forward the only way to get better performance is parallelism as from a hardware pove, single core performance gains have already maxed out.
