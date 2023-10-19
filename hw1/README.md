## SIMD Programming

### Part 1: Vectorizing Code Using Fake SIMD Intrinsics
Rather than craft an implementation using SSE or AVX2 vector intrinsics that map to real SIMD vector instructions on modern CPUs, to make things a little easier, we’re asking you to implement your version using PP’s “fake vector intrinsics” defined in PPintrin.h. The PPintrin.h library provides you with a set of vector instructions that operate on vector values and/or vector masks. (These functions don’t translate to real CPU vector instructions, instead we simulate these operations for you in our library, and provide feedback that makes for easier debugging.)

##### modify function in `vectorOP.cpp`
- `arraySumVector`
- `clampedExpVector`

### Part 2: Vectorizing Code with Automatic Vectorization Optimizations

Auto-vectorization is enabled by default at optimization levels -O2 and -O3. We first use -fno-vectorize to disable automatic vectorization, and start with the following simple loop (in test1.cpp):

```cpp
void test1(float* a, float* b, float* c, int N) {
  __builtin_assume(N == 1024);
  
  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}
```

We have added an outer loop over I whose purpose is to eliminate measurement error in gettime(). Notice that __builtin_assume(N == 1024) tells the compiler more about the inputs of the program—say this program is used in a mobile phone and always has the same input size—so that it can perform more optimizations.

You can compile this C++ code fragment with the following command and see the generated assembly code in assembly/test1.novec.s.

```Makefile
$ make clean; make test1.o ASSEMBLE=1
```

#### 2.1 Turning on auto-vectorization
Let’s turn the compiler optimizations on and see how much the compiler can speed up the execution of the program.

We remove -fno-vectorize from the compiler option to turn on the compiler optimizations, and add -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize to get more information from clang about why it does or does not optimize code. This was done in the makefile, and you can enable auto-vectorization by typing the following command, which generates assembly/test1.vec.s.

```Makefile
$ make clean; make test1.o ASSEMBLE=1 VECTORIZE=1
```

You should see the following output, informing you that the loop has been vectorized. Although clang does tell you this, you should always look at the assembly to see exactly how it has been vectorized, since it is not guaranteed to be using the vector registers optimally.
```Makefile
test1.cpp:14:5: remark: vectorized loop (vectorization width: 4, interleaved count: 2) [-Rpass=loop-vectorize]
    for (int j=0; j<N; j++) {
    ^
```
You can observe the difference between test1.vec.s and test1.novec.s with the following command or by changing the compiler flag on Compiler Explorer.
```Makefile
$ diff assembly/test1.vec.s assembly/test1.novec.s
```
#### 2.2 Adding the `__restrict` qualifier
Now, if you inspect the assembly code—actually, you don’t need to do that, which is out of the scope of this assignment—you will see the code first checks if there is a partial overlap between arrays a and c or arrays b and c. If there is an overlap, then it does a simple non-vectorized code. If there is no overlap, it does a vectorized version. The above can, at best, be called partially vectorized.

The problem is that the compiler is constrained by what we tell it about the arrays. If we tell it more, then perhaps it can do more optimization. The most obvious thing is to inform the compiler that no overlap is possible. This is done in standard C by using the restrict qualifier for the pointers. By adding this type qualifier, you can hint to the compiler that for the lifetime of the pointer, only the pointer itself or a value directly derived from it (such as pointer + 1) will be used to access the object to which it points.

C++ does not have standard support for restrict, but many compilers have equivalents that usually work in both C++ and C, such as the GCC’s and clang’s __restrict__ (or __restrict), and Visual C++’s __declspec(restrict).

The code after adding the __restrict qualifier is shown as follows.
```cpp
void test(float* __restrict a, float* __restrict b, float* __restrict c, int N) {
  __builtin_assume(N == 1024);

  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}
```
Let’s modify test1.cpp accordingly and recompile it again with the following command, which generates assembly/test1.vec.restr.s.

```Makefile
$ make clean; make test1.o ASSEMBLE=1 VECTORIZE=1 RESTRICT=1
```
Now you should see the generated code is better—the code for checking possible overlap is gone—but it is assuming the data are NOT 16 bytes aligned (movups is unaligned move). It also means that the loop above can not assume that the arrays are aligned.

If clang were smart, it could test for the cases where the arrays are either all aligned, or all unaligned, and have a fast inner loop. However, it is unable to do that currently. 🙁

#### 2.3 Adding the `__builtin_assume_aligned` intrinsic
In order to get the performance we are looking for, we need to tell clang that the arrays are aligned. There are a couple of ways to do that. The first is to construct a (non-portable) aligned type, and use that in the function interface. The second is to add an intrinsic or three within the function itself. The second option is easier to implement on older code bases, as other functions calling the one to be vectorized do not have to be modified. The intrinsic has for this is called __builtin_assume_aligned:
```cpp
void test(float* __restrict a, float* __restrict b, float* __restrict c, int N) {
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 16);
  b = (float *)__builtin_assume_aligned(b, 16);
  c = (float *)__builtin_assume_aligned(c, 16);
  
  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}
```
Let’s modify test1.cpp accordingly and recompile it again with the following command, which generates assembly/test1.vec.restr.align.s.
```Makefile
$ make clean; make test1.o ASSEMBLE=1 VECTORIZE=1 RESTRICT=1 ALIGN=1
```

Let’s see the difference:
```Makefile
$ diff assembly/test1.vec.restr.s assembly/test1.vec.restr.align.s
```
Now finally, we get the nice tight vectorized code (movaps is aligned move.) we were looking for, because clang has used packed SSE instructions to add 16 bytes at a time. It also manages load and store two at a time, which it did not do last time. The question is now that we understand what we need to tell the compiler, how much more complex can the loop be before auto-vectorization fails.

#### 2.4 Turning on AVX2 instructions
Next, we try to turn on AVX2 instructions using the following command, which generates assembly/test1.vec.restr.align.avx2.s
```Makefile
$ make clean; make test1.o ASSEMBLE=1 VECTORIZE=1 RESTRICT=1 ALIGN=1 AVX2=1
```
Let’s see the difference:
```Makefile
$ diff assembly/test1.vec.restr.align.s assembly/test1.vec.restr.align.avx2.s
```
We can see instructions with prefix v*. That’s good. We confirm the compiler uses AVX2 instructions; however, this code is still not aligned when using AVX2 registers.

#### 2.5 Performance impacts of vectorization
Let’s see what speedup we get from vectorization. Build and run the program with the following configurations, which run test1() many times, and record the elapsed execution time.
```Makefile
# case 1
$ make clean && make && ./test_auto_vectorize -t 1
# case 2
$ make clean && make VECTORIZE=1 && ./test_auto_vectorize -t 1
# case 3
$ make clean && make VECTORIZE=1 AVX2=1 && ./test_auto_vectorize -t 1
```
Note that you may wish to use the workstations provided by this course, which support AVX2; otherwise, you may get a message like “Illegal instruction (core dumped)”. You can check whether or not a machine supports the AVX2 instructions by looking for avx2 in the flags section of the output of cat /proc/cpuinfo.
```Makefile
$ cat /proc/cpuinfo | grep avx2
```

You may also run test2() and test3() with ./test_auto_vectorize -t 2 and ./test_auto_vectorize -t 2, respectively, before and after fixing the vectorization issues in Section 2.6.

#### 2.6 More examples
##### 2.6.1 EXAMPLE 2
Take a look at the second example below in test2.cpp:
```cpp
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
Compile the code with the following command:
```Makefile
make clean; make test2.o ASSEMBLE=1 VECTORIZE=1
```
Note that the assembly was not vectorized. Now, change the function with a patch file (test2.cpp.patch), which is shown below, by running patch -i ./test2.cpp.patch.
```Diff
--- test2.cpp
+++ test2.cpp
@@ -14,9 +14,8 @@
     for (int j = 0; j < N; j++)
     {
       /* max() */
-      c[j] = a[j];
-      if (b[j] > a[j])
-        c[j] = b[j];
+      if (b[j] > a[j]) c[j] = b[j];
+      else c[j] = a[j];
     }
   }
```
Now, you actually see the vectorized assembly with the movaps and maxps instructions.

##### 2.6.2 EXAMPLE 3
Take a look at the third example below in test3.cpp:
```cpp
double test3(double* __restrict a, int N) {
  __builtin_assume(N == 1024);
  a = (double *)__builtin_assume_aligned(a, 16);

  double b = 0;

  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      b += a[j];
    }
  }

  return b;
}
```
Compile the code with the following command:
```Makefile
$ make clean; make test3.o ASSEMBLE=1 VECTORIZE=1
```
You should see the non-vectorized code with the addsd instructions.

Notice that this does not actually vectorize as the xmm registers are operating on 8 byte chunks. The problem here is that clang is not allowed to re-order the operations we give it. Even though the the addition operation is associative with real numbers, they are not with floating point numbers. (Consider what happens with signed zeros, for example.)

Furthermore, we need to tell clang that reordering operations is okay with us. To do this, we need to add another compile-time flag, -ffast-math. Compile the program again with the following command:
```Makefile
$ make clean; make test3.o ASSEMBLE=1 VECTORIZE=1 FASTMATH=1
```
You should see the vectorized code with the addpd instructions.
