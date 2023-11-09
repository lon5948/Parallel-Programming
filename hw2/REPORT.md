# PP-f23 Assignment II Report
### Q1: Is speedup linear in the number of threads used? In your writeup hypothesize why this is (or is not) the case?

To compute the subset of rows assigned to this thread, we first compute the total number of rows per thread by dividing the total height of the image by the number of threads. We then compute the starting row for this thread by multiplying the thread ID by the total number of rows per thread. Finally, we call the mandelbrotSerial function with the appropriate input arguments to compute the Mandelbrot set for the subset of rows assigned to this thread.


```cpp
void workerThreadStart(WorkerArgs *const args)
{
    int totalRows = args->height / args->numThreads;
    int startRow = args->threadId * totalRows;
    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, startRow, totalRows, args->maxIterations, args->output);
}
```

Here are the results I measured separately for VIEW 1 and VIEW 2 with thread numbers set to 2, 3, and 4.

#### VIEW 1
```bash!
$ ./mandelbrot -t 2
[mandelbrot serial]:            [382.495] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [196.567] ms
Wrote image file mandelbrot-thread.ppm
                                (1.95x speedup from 2 threads)

$ ./mandelbrot -t 3
[mandelbrot serial]:            [382.168] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [236.456] ms
Wrote image file mandelbrot-thread.ppm
                                (1.62x speedup from 3 threads)
     
$ ./mandelbrot -t 4
[mandelbrot serial]:            [381.161] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [160.899] ms
Wrote image file mandelbrot-thread.ppm
                                (2.37x speedup from 4 threads)
```

#### VIEW 2

```bash!
$ ./mandelbrot -t 2 --view 2
[mandelbrot serial]:            [201.403] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [123.580] ms
Wrote image file mandelbrot-thread.ppm
                                (1.63x speedup from 2 threads)
                                
$ ./mandelbrot -t 3 --view 2
[mandelbrot serial]:            [199.723] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [96.896] ms
Wrote image file mandelbrot-thread.ppm
                                (2.06x speedup from 3 threads)
                                
$ ./mandelbrot -t 4 --view 2
[mandelbrot serial]:            [201.582] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [84.119] ms
Wrote image file mandelbrot-thread.ppm
                                (2.40x speedup from 4 threads)
```

| VIEW 1         | VIEW 2         |
| --------------| --------------|
| ![Q1-view1.png](https://hackmd.io/_uploads/BkwLTAQXa.png)| ![Q1-view2.png](https://hackmd.io/_uploads/BkThR07Qp.png) |

In VIEW 1, the speedup does not show a linear increase with the number of threads; instead, it decreases when the number of threads is set to 3. In VIEW 2, the speedup demonstrates linear growth as the number of threads increases.


The two images below are pictures of view1 and view2. We know that the numerical values affect the color depth; the larger the value, the lighter the color. From the images, it is evident that there are significantly more light-colored pixels in VIEW 1 compared to View2, and they are concentrated. Therefore, I suspect that the workload of each thread is related to the number of light-colored pixels in its assigned area. In VIEW 1, the computational workload for different regions should be different, leading to varying workloads among the threads, with some threads having excessively high workloads that slow down the overall speed.

| VIEW 1         | VIEW 2         |
| --------------| --------------|
| ![view1.png](https://hackmd.io/_uploads/S1yvMJEQT.png) | ![view2.png](https://hackmd.io/_uploads/Skz_z14Q6.png) |

---

### Q2: How do your measurements explain the speedup graph you previously created?


In the `workerThreadStart` function, I have included code to print out the runtime for each individual thread. This allows me to monitor and analyze the execution times of each thread.

#### VIEW 1
- number of threads = 2
    ```
    thread 0: 196.762 ms
    thread 1: 198.514 ms
    ```
- number of threads = 3
    ```
    thread 0: 77.218 ms
    thread 2: 77.606 ms
    thread 1: 238.488 ms
    ```
- number of threads = 4
    ```
    thread 0: 36.918 ms
    thread 3: 37.051 ms
    thread 1: 160.697 ms
    thread 2: 161.748 ms
    ```

Because the light pixels in VIEW 1 are mainly concentrated in the central area, the workload for the thread in the middle is heavier. 

From the results, we can see that when there are 2 threads, the workload is evenly divided, and since VIEW 1 is symmetric, the runtimes for both threads are roughly the same. However, when there are 3 threads, it's evident that thread 1 which is in the middle has the longest runtime. When there are 4 threads, again, it's threads 1 and 2 that have the longest execution times.

Therefore, because the runtime for the thread in the middle is longer, it slows down the overall execution speed, leading to a decrease in speedup.

#### VIEW 2
- number of threads = 2
    ```
    thread 1: 84.775 ms
    thread 0: 126.392 ms
    ```
- number of threads = 3
    ```
    thread 2: 54.390 ms
    thread 1: 58.747 ms
    thread 0: 99.604 ms
    ```
- number of threads = 4
    ```
    thread 3: 41.420 ms
    thread 1: 42.546 ms
    thread 2: 44.047 ms
    thread 0: 85.170 ms
    ```

The light pixels in VIEW 2 are primarily concentrated in the upper portion, resulting in a relatively higher workload for thread 0 at the beginning. From the results, it is evident that regardless of the number of threads, thread 0 consistently has the longest execution time. 

However, since the light pixels are relatively dispersed throughout the image as a whole, it doesn't significantly impact the overall execution speed, allowing us to observe linear growth in speedup.

---

### Q3: In your write-up, describe your approach to parallelization and report the final 4-thread speedup obtained.


Because VIEW 1 has a concentration of light pixels in the central region, resulting in an uneven workload distribution, to address this issue, I made a modification to the `workerThreadStart()` function. When executing `mandelbrotSerial()`, I now pass **totalRows=1**, which evenly distributes the originally concentrated area of light pixels among the three threads, preventing a situation where one thread significantly slows down the others.

The modified `workerThreadStart()` function appears as follows:

```cpp
void workerThreadStart(WorkerArgs *const args)
{
    int height = args->height;
    int width = args->width;
    int threadId = args->threadId;
    int numThreads = args->numThreads;
    
    for (int id = threadId; id < height; id += numThreads) {
        mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, width, height, id, 1, args->maxIterations, args->output);
    }
}
```

From the two images below, we can observe that whether in VIEW 1 or VIEW 2, the speedup exhibits linear growth.

| VIEW 1         | VIEW 2         |
| --------------| --------------|
| ![Q3-view1.png](https://hackmd.io/_uploads/ryQJvx4mT.png) | ![Q3-view2.png](https://hackmd.io/_uploads/SJQ1DxNQa.png) |

#### Final 4-thread speedup

##### VIEW 1
```bash!
$ ./mandelbrot -t 4

[mandelbrot serial]:            [381.010] ms
Wrote image file mandelbrot-serial.ppm

thread 0: 99.261 ms
thread 2: 99.776 ms
thread 1: 100.211 ms
thread 3: 100.407 ms

[mandelbrot thread]:            [100.209] ms
Wrote image file mandelbrot-thread.ppm
                                (3.80x speedup from 4 threads)
                                
```

##### VIEW 2
```bash!
$ ./mandelbrot -t 4 --view 2

[mandelbrot serial]:            [200.604] ms
Wrote image file mandelbrot-serial.ppm

thread 2: 52.579 ms
thread 1: 52.617 ms
thread 3: 52.723 ms
thread 0: 52.779 ms

[mandelbrot thread]:            [52.827] ms
Wrote image file mandelbrot-thread.ppm
                                (3.80x speedup from 4 threads)
```

Based on the results, we can observe that the workload is indeed evenly distributed among the four threads.

---

### Q4: Now run your improved code with eight threads. Is performance noticeably greater than when running with four threads? Why or why not? (Notice that the workstation server provides 4 cores 4 threads.)

- number of threads = 4
    ```
    $ ./mandelbrot -t 4
    [mandelbrot serial]:		[460.052] ms
    Wrote image file mandelbrot-serial.ppm
    [mandelbrot thread]:		[121.596] ms
    Wrote image file mandelbrot-thread.ppm
                    (3.78x speedup from 4 threads)

    $ ./mandelbrot -t 4 --view 2
    [mandelbrot serial]:		[290.004] ms
    Wrote image file mandelbrot-serial.ppm
    [mandelbrot thread]:		[76.628] ms
    Wrote image file mandelbrot-thread.ppm
                    (3.78x speedup from 4 threads)
    ```

- number of threads = 8
    ```
    $ ./mandelbrot -t 8
    [mandelbrot serial]:		[459.579] ms
    Wrote image file mandelbrot-serial.ppm
    [mandelbrot thread]:		[122.165] ms
    Wrote image file mandelbrot-thread.ppm
                    (3.76x speedup from 8 threads)

    $ ./mandelbrot -t 8 --view 2
    [mandelbrot serial]:		[289.983] ms
    Wrote image file mandelbrot-serial.ppm
    [mandelbrot thread]:		[76.953] ms
    Wrote image file mandelbrot-thread.ppm
                    (3.77x speedup from 8 threads)
    ```

From the results above, it can be seen that when the number of threads is 8, it is slower compared to when the number of threads is 4. Despite having more threads, the speedup does not increase. I think this is because the workstation server provides 4-core 4-thread. When using 8 threads, the cores allocated to threads may be more than 1, which can lead to the overhead of context switching.