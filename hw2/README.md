## Multi-thread Programming

The purpose of this assignment is to familiarize yourself with Pthread and std::thread programming in C and C++, respectively. You will also gain experience measuring and reasoning about the performance of parallel programs (a challenging, but important, skill you will use throughout this class). This assignment involves only a small amount of programming, but a lot of analysis!

### Part 1: Parallel Counting PI Using Pthreads
#### 1.1 Problem Statement
This is a follow-up assignment from Assignment ZERO. You are asked to turn the serial program into a Pthreads program that uses a Monte Carlo method to estimate PI. The main thread should read in the total number of tosses and print the estimate. You may want to use `long long int` for the number of hits in the circle and the number of tosses, since both may have to be very large to get a reasonable estimate of PI.

Your mission is to make the program as fast as possible. You may consider a lot of methods to improve the speed, such as SIMD intrinsics or a faster random number generator. However, you cannot break the following rules: you need to implement the Monte Carlo method using Pthreads.

`Hint: You may want to use a reentrant and thread-safe random number generator.`

You are allowed to use third-party libraries in this part, such as pseudorandom number generators or SIMD intrinsics.

#### 1.2 Requirements
- Typing `make` in the `part1`directory should build the code and create an executable called `pi.out`.
- `pi.out` takes two command-line arguments, which indicate the number of threads and the number of tosses, respectively. The value of the first and second arguments will not exceed the range of `int` and `long long int`, respectively. `pi.out` should work well for all legal inputs.
- `pi.out` should output (to stdout) the estimated PI value, which is **accurate to three decimal places** (i.e., 3.141xxx) with at least `1e8` tosses.

Example:
```
$ make && ./pi.out 8 1000000000
3.1415926....
```

### Part 2: Parallel Fractal Generation Using std::thread

#### 2.1 Problem Statement
```
$ cd <your_workplace>
$ wget http://PP-f23.github.io/assignments/HW2/HW2.zip
$ unzip HW2.zip -d HW2
$ cd HW2/part2
```

Build and run the code in the `part2` directory of the code base. (Type `make` to build, and `./mandelbrot` to run it. `./mandelbrot --help` displays the usage information.)

This program produces the image file `mandelbrot-serial.ppm`, which is a visualization of a famous set of complex numbers called the Mandelbrot set. [Most platforms have a `.ppm` viewer. For example, to view the resulting images, use `tiv` command (already installed) to display them on the terminal.]

As you can see in the images below, the result is a familiar and beautiful fractal. Each pixel in the image corresponds to a value in the complex plane, and the brightness of each pixel is proportional to the computational cost of determining whether the value is contained in the Mandelbrot set. To get image 2, use the command option `--view 2`. (See function `mandelbrotSerial()` defined in `mandelbrotSerial.cpp`). You can learn more about the definition of the Mandelbrot set.

| VIEW 1         | VIEW 2         |
| --------------| --------------|
| ![view1.png](https://hackmd.io/_uploads/S1yvMJEQT.png) | ![view2.png](https://hackmd.io/_uploads/Skz_z14Q6.png) |

Your job is to parallelize the computation of the images using `std::thread`. Starter code that spawns one additional thread is provided in the function `mandelbrotThread(`) located in `mandelbrotThread.cpp`. In this function, the main application thread creates another additional thread using the constructor `std::thread` (function, args…). It waits for this thread to complete by calling `join` on the thread object.

Currently the launched thread does not do any computation and returns immediately. You should add code to `workerThreadStart` function to accomplish this task. You will not need to make use of any other `std::thread` API calls in this assignment.

#### 2.2 Requirements
1. Modify the starter code to parallelize the Mandelbrot generation using two processors. Specifically, compute the top half of the image in thread 0, and the bottom half of the image in thread 1. This type of problem decomposition is referred to as spatial decomposition since different spatial regions of the image are computed by different processors.

2. Extend your code to use 2, 3, 4 threads, partitioning the image generation work accordingly (threads should get blocks of the image). Q1: In your write-up, produce a graph of speedup compared to the reference sequential implementation as a function of the number of threads used FOR VIEW 1. Is speedup linear in the number of threads used? In your writeup hypothesize why this is (or is not) the case? (You may also wish to produce a graph for VIEW 2 to help you come up with a good answer. Hint: take a careful look at the three-thread data-point.)

3. To confirm (or disprove) your hypothesis, measure the amount of time each thread requires to complete its work by inserting timing code at the beginning and end of workerThreadStart(). Q2: How do your measurements explain the speedup graph you previously created?

4. Modify the mapping of work to threads to achieve to improve speedup to at about 3-4x on both views of the Mandelbrot set (if you’re above 3.5x that’s fine, don’t sweat it). You may not use any synchronization between threads in your solution. We are expecting you to come up with a single work decomposition policy that will work well for all thread counts—hard coding a solution specific to each configuration is not allowed! (Hint: There is a very simple static assignment that will achieve this goal, and no communication/synchronization among threads is necessary.). Q3: In your write-up, describe your approach to parallelization and report the final 4-thread speedup obtained.

5. Q4: Now run your improved code with eight threads. Is performance noticeably greater than when running with four threads? Why or why not? (Notice that the workstation server provides 4 cores 4 threads.)

