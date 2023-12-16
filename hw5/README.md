### Problem Statement: Paralleling Fractal Generation with CUDA

Following part 2 of HW2, we are going to parallelize fractal generation by using CUDA.

This program produces the image file `mandelbrot-test.ppm`, which is a visualization of a famous set of complex numbers called the Mandelbrot set. [Most platforms have a .ppm viewer. For example, to view the resulting images, use `tiv` command (already installed) to display them on the terminal.]

As you can see in the images below, the result is a familiar and beautiful fractal. Each pixel in the image corresponds to a value in the complex plane, and the brightness of each pixel is proportional to the computational cost of determining whether the value is contained in the Mandelbrot set. To get image 2, use the command option `--view 2`. You can learn more about the definition of the Mandelbrot set.

Your job is to parallelize the computation of the images using CUDA. A starter code that spawns CUDA threads is provided in function `hostFE()`, which is located in `kernel.cu`. This function is the host front-end function that allocates the memory and launches a GPU kernel.

Currently `hostFE()` does not do any computation and returns immediately. You should add code to `hostFE()` function and finish `mandelKernel()` to accomplish this task.

The kernel will be implemented, of course, based on `mandel()` in `mandelbrotSerial.cpp`, which is shown below. You may want to customized it for your kernel implementation.

```cpp
int mandel(float c_re, float c_im, int maxIteration)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < maxIteration; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}
```

### Implement three approaches to solve the questions:

- Method 1: Each CUDA thread processes one pixel. Use malloc to allocate the host memory, and use cudaMalloc to allocate GPU memory. Name the file kernel1.cu. (Note that you are not allowed to use the image input as the host memory directly)

- Method 2: Each CUDA thread processes one pixel. Use cudaHostAlloc to allocate the host memory, and use cudaMallocPitch to allocate GPU memory. Name the file kernel2.cu.

- Method 3: Each CUDA thread processes a group of pixels. Use cudaHostAlloc to allocate the host memory, and use cudaMallocPitch to allocate GPU memory. You can try different size of the group. Name the file kernel3.cu.

### Reference
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [cudaMallocPitch API Document](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c)
- [CUDA 開發環境設定與簡易程式範例](https://tigercosmos.xyz/post/2020/12/system/cuda-basic/)
- [CPU 與 GPU 計算浮點數的差異](https://tigercosmos.xyz/post/2020/12/system/floating-number-cpu-gpu/)