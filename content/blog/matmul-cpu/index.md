---
title: "[WIP] Optimizing matmul on CPU"
layout: post
date: 2025-12-20
tags:
    - Optimization, Parallel Processing
description: ""
draft: false
mathjax: true
---

## Introduction and Setup
This blog starts with a naive implementation of matmul in C and optimizes it one step at a time. 

I am using [the following machine](https://www.apple.com/in/shop/buy-mac/macbook-pro/14-inch-space-black-standard-display-apple-m5-chip-with-10-core-cpu-and-10-core-gpu-16gb-memory-512gb) with a 10 core CPU (4 Performance cores+6 Efficiency cores). 
 

Food for thought: Is matrix multiplication on CPU compute bound or memory bound? Think about it...

## Algorithmic complexity of Matrix Multiplication: Calculating the FLOPs required
Matrix multiplication is ubiquitous in many areas of computer sciene. As the matrices grow in sizes, the amount of FLOPs required to calculate the matmul grow cubically[^1]. Let's see how. 

Consider two matrices $A (i \times k)$ and $B (k \times j)$. The product of $A$ and $B$, $AB$ is a matrix $C$ of shape $(i \times j)$.

For simplicity and without loss of generality, let's consider the matrices to be square so we have $A (n \times n)$, $B (n \times n)$ and their product $C (n \times n)$. One element $(c_1, c_2)$ of $C$ is defined as:

$$c_{c1,c2} = \sum_{x=1}^{n} a_{c1,x} b_{x,c2}$$

This is a total of 2n - 1 floating point operations (FLOPs) required to calculate one element of C -- $n$ multiplication ops + $(n-1)$ addition ops. A total of $n^2$ elements need to be calculated in $C$ and hence the total FLOPs:

$$(2n - 1) n^2 = 2n^3 - n^2$$

As $n$ grows bigger (asymptomatic, if you're feeling fancy), $n^2$ becomes pretty negligible in comparison to $2n^3$ and hence can be ignored so the total FLOPs required is roughly $2n^3$. Therefore, the computational complexity is $O(n^3)$. 

For the purpose of this blog, I am keeping $n=4096$ which translates to roughly 137 GFLOPs. That's seems like a lot of FLOPs, but is rather pretty small if compared with the peak FLOPs offered by any standard modern processor.  

To benchmark, I calculate the time required to multiply two $4096 x 4096$ numpy arrays using `np.matmul` and it comes out to 0.1042s. 

## [Naive implementation](https://github.com/srishti-git1110/SGEMM-cpu/blob/main/sgemm-cpu/matmuls/naive.c)
The matmul loop is this:

```C
for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
```

When complied with the `-O3` flag[^2] which is the maximum level with safe optimizations, the latency is 299.289 sec. Full compilation command below:

```
gcc -Wall -O3 sgemm-cpu/matmuls/naive.c -o sgemm-cpu/matmuls/naive
```

All the further optimizations use the same flags to compile the code.

### [A minor optimization: Avoiding unnecessary memory accesses](https://github.com/srishti-git1110/SGEMM-cpu/blob/main/sgemm-cpu/matmuls/naive_register_accumulation.c)
A very small thing we could do with the naive implementation is avoiding multiple reads and writes of intermediate partial sums from and to the memory/cache, like so:

```C
for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float running_sum = 0.0;
            for (int k = 0; k < N; k++) {
                running_sum += A[i][k] * B[k][j];
            }
            C[i][j] = running_sum;
        }
    }
```
That helps a bit and brings the latency down to 199.866s. This optimization is simply causing the compiler to keep the partial sum in the registers only and write back to the memory only once when the full sum is done. Meaning in this case the compiler isn't issuing separate instructions to store the partial sum back to the memory after every loop iteration and to load it back on the next iteration; and that reduces some latency. [^3] 

## [Loop reordering](https://github.com/srishti-git1110/SGEMM-cpu/blob/main/sgemm-cpu/matmuls/cache_aware.c)
The naive implementation follows the most natural mathy way to calculate a matmul $C = AB$ -- element $C[0][0]$ is *fully* calculated first via a scalar product of the first row of $A$ with the first column of $B$. Element $C[0][1]$ is *fully* calculated next via the scalar product of the first row of $A$ with the second column of $B$, and so on. 

There are two key insights over here:
1. Languages like C store matrices in the memory in a row major format like this:

![row major order](row-major.jpeg)

If we now follow the calculation of $C[0][0]$ in code, it's equivalent to completing the inner-most $k$ loop for $i=0, j=0$. 

The first part of the figure below uses 3 shades of red to color the values from $A, B, C$ that are accessed in the code during the calculation of $C[0][0]$. Focus on the location of these values in the memory by following the color. 

![values accessed](matmul-values-accessed.png)

The figure quickly makes it clear that the accessed values for $A, C$ are close to each other in the memory while they're further apart for $B$. Now, remember that data is fetched from the memory in the caches in the granularity of cache lines. One cache line on my machine is of size 128 bytes which is equivalent to 32 single precision values. 

So on the very first loop iteration $i=0, j=0, k=0$, when $A[0][0], B[0][0], C[0][0]$ are fetched from the memory, the cache looks something like:

![cache after first iter](cache.png)


This is simply because a cache line loads contiguous values from the memory where the matrices are stored in a row major format.
On the second iteration $i=0, j=0, k=1$, we need $A[0][1], B[1][0], C[0][0]$ and while $A[0][1]$ and $C[0][0]$ are found in the cache, we have a miss for $B[1][0]$ that we need to fetch from the memory. And this holds for each iteration of the loop. Easy to figure out why -- our cache line is only 128 bytes (32 floats) while  each subsequent loop iteration accesses a value from B that's 4096 values apart from the value accessed in the last iteration. And this high cache miss rate explains the high latency we saw w the naive implementation.


<!-- Hence, for the access pattern shown above, we're getting good cache hit rates for values of A and C but a very high miss rate for values of B -- at each loop iteration, we have a hit for $C[0][0]$, mostly hits for value of the first row of A but we have a cache miss for all the values of B involved ($B[0][0], B[1][0]...$). Simply because the cache line loads contigous values from the memory and owing to the row major storage in the memory, the different values of B involved in calculating $C[0][0]$ are 4096 values (4096x4 bytes) apart which well exceeds the size of a usual cache line.  -->

2. The second insight is not too difficult to understand -- if we just change the loop orders (eg. $jik$ or $jki$ etc. instead of the most natural  $ijk$), we'll still get the correct matmul. It's also why I was italicizing *fully* above. Thing is that with different loop orders we're not fully calculating each element of C in one full iteration of the innermost loop but that doesn't hurt the correctness of the matmul and that's easy to realise. Alright. Given that, we note that some loop orders have a better overall cache hit rate as compared to the naive $ijk$ order (btw some orders also have a worse rate than $ijk$!). And hence just by changing the orders, we'll be able to reduce the latency by not having to make as many high latency accesses to the memory. Experimenting w different orders, the lowest latency of 4.49s corresponds to order ikj down from 203.229s with order $ijk$ which is a ~45x improvement already!

```C
for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
```

### What was the bottleneck?
A simple loop reordering of our naive implementation provided a 45 fold improvement. That's a lot for such a simple change. What does this tell us? Obviously, we still performed total 137 GFLOPs so that part didn't change. Turns out, in the naive implementation our bottleneck was the memory bandwidth. Meaning the CPU execution units (ALUs) were bottlenecked by the latency of data transfer between the memory and the CPU, implying we were overall *memory bound in the naive implementation*. Of course, we cannot change the memory bandwidth towards a faster memory. So what did we do? We simply wrote the code such that it allows for better cache reuse in subsequent loop iters!

### What about the computation aspect?
// vectorization enabled by loop ordering - figured out by the compiler

### What about the minor optimization from [above](https://srishti-git1110.github.io/blog/matmul-cpu/#a-minor-optimization-avoiding-unnecessary-memory-accesses)?
No. Because with the reordered loops, we're not calculating the full result $C[i][j]$ in one iteration of the inner most loop, we can't avoid the loading and storing of partial sums anymore!

## Tiling
<!-- Let me say this: IMHO, tiling is simple (really, very simple) but explained too poorly (hastily?) in too many places. 

Let's try to give it some time by doing something boring. For our best loop order $ikj$, we start by looking at the values that are accessed by the inner most $j$ loop when i=0, k=0.

- $A[0][0]$
- C[0][0], C[0][1], ... C[0][4095] -->



[^1]: This is for the standard algorithm. There's other algos like [Strassen's](https://en.wikipedia.org/wiki/Strassen_algorithm) with better complexity.

[^2]: Explore all optimization levels [here](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html).

[^3]: Sometimes, the compiler might also consider it safe to skip the extra store/reload steps when we directly update $C[i][j]$ in every loop iteration. But other times it may not. So, we're just being explicit here and writing code such that the compiler would *always* consider it safe to not issue the extra store/reload instructions for the partial sums.

