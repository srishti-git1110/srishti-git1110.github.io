---
title: "Learning CUDA Programming: A Primer"
layout: post
date: 2024-03-31
tags:
    - CUDA, Computer Architecture, Optimization, GPU Programming, Parallel processing
description: "A friendly primer to learning CUDA programming"
draft: true
mathjax: true
---

The goal of this short post is to get the reader familiar with the basic concepts/terminologies/jargons/paradigms that are relevant to learning parallel programming. We will also cover the architectural details of two of the processors that empower the modern day computers - the CPUs and the GPUs.

By the end of this post, the reader should have a basic understanding of the following terms - (in no particular order) chips, processors, microprocessors, cores, latency device, throughput device, clock speed, threads, processes, instructions, memory bandwidth, memory system.

## Why are we here?
A few months back, I used to visit the PyTorch forums almost daily. When I just began reading over there, terms "host memory", and "device memory" confused me (I haven't majored in CS and hence the unfamiliarity with basics).

A quick lookup did the deal but a similar thing happened as I started reading the PMPP book with a goal to learn parallel programming on GPU "devices" ;). I was able to get a hold of the code fairly easily but the jargons weren't clear at once and hence this primer post.

## Core, Microprocessor/Processor, Chip
"A chip" is the physical semiconductor chip; it's "a physical integrated circuit" comprised of transistors, resistors, and capacitors.


A processor (here, think of CPU, the central "processing" unit) is a digital circuit that's implemented on a single or a few chips. Now, the term micro is appended to the beginning of processors to refer to the fact that it takes a single or a very few chips to implement a microprocessor. But this is more of a definition. In the context/scope of this post, consider 1 microprocessor = 1 chip.

The modern day computer is powered by processors of different types - CPUs, GPUs etc. I have also read the term processor chip, the meaning of which should be clear now.

Now, here's the thing: In older days, 1 processor used to mean 1 processing unit (single CPU based microprocessor) which changed around the year 2000 when microprocessors with more than one processing unit/CPU were introduced. Those are what are called as multi-core processors. Hence, a CPU core is basically a single processing unit within a processor chip that is capable of running instructions independently and hence the modern day microprocessor with several cores is essentially a parallel processor benefitting from software that leverages parallel programming paradigms. 

If you google [intel core i9 processor](https://www.intel.com/content/www/us/en/products/details/processors/core/i9/products.html), the table there has a column # of "cores".

## Instructions, Threads, Processes

