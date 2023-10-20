# PP-f23 Assignment I Report
### Q1
Run `./myexp -s 10000` and sweep the vector width from 2, 4, 8, to 16. Record the resulting vector utilization.

| **VECTOR_WIDTH** |                  **Result**                   |
|:----------------:|:---------------------------------------------:|
|      **2**       | ![](https://hackmd.io/_uploads/HynVbRRWa.png) |
|      **4**       | ![](https://hackmd.io/_uploads/rymmWRRWa.png) |
|      **6**       | ![](https://hackmd.io/_uploads/H1HZ-R0WT.png) |
|      **8**       | ![](https://hackmd.io/_uploads/SyPHkCCW6.png) |

#### Q1-1: Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?
The vector utilization decreases as VECTOR_WIDTH increases. This happens because as VECTOR_WIDTH increases, there are more vector lanes available for parallel processing, but it becomes more challenging to find enough independent operations to fully utilize all these lanes. This underutilization of vector lanes leads to a decrease in vector utilization.

For example, the value of VECTOR_WIDTH affects the likelihood of the loop exiting. When VECTOR_WIDTH is smaller, such as 2, there's a higher probability of the loop exiting. This is because the `_pp_cntbits(maskIsPositive)` function counts the number of 1 in the maskIsPositive vector. With a smaller VECTOR_WIDTH, the vector is processed in smaller chunks, and it's more likely that the count will reach zero sooner, allowing the loop to exit.

For example, when analyzing the representation "1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0" with different VECTOR_WIDTH values:

- VECTOR_WIDTH = 2: The vector is processed in 2-lane chunks, and the loop is more likely to exit, resulting in a utilization of 75%.
- VECTOR_WIDTH = 4: The vector is processed in 4-lane chunks, but the loop still has a moderate chance of exiting earlier, leading to a utilization of 37.5%.
- VECTOR_WIDTH = 8: With a larger VECTOR_WIDTH, the loop is less likely to exit early, and the operations are more likely to continue until the end of the vector. However, the overall utilization is lower at 18.7%. 

So, smaller VECTOR_WIDTH values increase the likelihood of early loop exit, which can affect the vector utilization. Higher VECTOR_WIDTH values make it more challenging for the loop to exit early, potentially resulting in lower vector utilization.

---

### Q2

#### Q2-1: Fix the code to make sure it uses aligned moves for the best performance.

Because the AVX2 uses 256-bit YMM registers, in order to align it, must set __builtin_assume_aligned to 32 bytes.

```cpp!
void test(float* __restrict a, float* __restrict b, float* __restrict c, int N) {
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);
  
  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}
```    

change `vmovups` to `vmovaps`  

![](https://hackmd.io/_uploads/ryDwLRAZT.png)  



#### Q2-2

| iteration | unvectorized | vectorized  |    avx2     |
|:---------:|:------------:|:-----------:|:-----------:|
|     1     | 8.52219 sec  | 2.71923 sec | 1.44235 sec |
|     2     | 8.65261 sec  | 2.70346 sec | 1.44333 sec |
|     3     | 8.54353 sec  | 2.70105 sec | 1.44514 sec |
|     4     | 8.53896 sec  | 2.70614 sec | 1.44315 sec |
|     5     | 8.52212 sec  | 2.70419 sec | 1.44316 sec |
|  median   | 8.53896 sec  | 2.70419 sec | 1.44316 sec |

#### 1. What speedup does the vectorized code achieve over the unvectorized code? 
```
speedup: 3x (8.53896 / 2.70419 = 3.157)
```

#### 2. What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)? 
```
Compare with unvectorized
speedup: 6x (8.53896 / 1.44316 = 5.916)

Compare with vectorized
speedup: 2x (2.70419 / 1.44316 = 1.873)
```

#### 3. What can you infer about the bit width of the default vector registers on the PP machines? What about the bit width of the AVX2 vector registers.

"Float" is 32 bits, and the speedup for vectorized compared to unvectorized is 3x. This suggests that the vector width is 4. Therefore, I speculate that the register length on PP machines is 128 bits. Furthermore, when using -mavx2, it's about twice as fast as vectorized, indicating that the AVX2 register length is 256 bits.

#### Q2-3: Provide a theory for why the compiler is generating dramatically different assembly.

#### Origin

In the original approach, the values of the "a" vector are all assigned to "c" first, and then it checks if each element is less than "b" and replaces it with "b" if necessary. This meant that during the compilation phase, the compiler had to complete the assignment before proceeding to the subsequent IF condition. As a result, it wasn't possible to vectorize the later part of the code simultaneously, and it may not utilize instructions like `maxps`.  

```cpp!
void test2(float *__restrict a, float *__restrict b, float *__restrict c, int N)
{
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 16);
  b = (float *)__builtin_assume_aligned(b, 16);
  c = (float *)__builtin_assume_aligned(c, 16);

  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < N; j++)
    {
      /* max() */
      c[j] = a[j];
      if (b[j] > a[j])
        c[j] = b[j];
    }
  }
}
```   
![](https://hackmd.io/_uploads/SkeLhP1fa.png)

#### After patching

After patching, we explicitly instruct the compiler to perform "c = max(a, b)", making it clear that we want to calculate the maximum. This allows the compiler to use the `maxps` instruction to optimize the operation.  

```cpp!
void test2(float *__restrict a, float *__restrict b, float *__restrict c, int N)
{
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 16);
  b = (float *)__builtin_assume_aligned(b, 16);
  c = (float *)__builtin_assume_aligned(c, 16);

  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < N; j++)
    {
      /* max() */
      if (b[j] > a[j]) c[j] = b[j];
      else c[j] = a[j];
    }
  }
}
```  

![](https://hackmd.io/_uploads/r1DusDkM6.png)

The test2() function took 11.9307 seconds to execute before patching and 2.70528 seconds after patching. This speedup of approximately 4x suggests that vectorization was indeed achieved after patching.











