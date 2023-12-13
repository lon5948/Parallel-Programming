#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#deifne N 16

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_img, int resX, int resY, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = thisY * resX + thisX;
    
    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;
    float z_re = c_re, z_im = c_im;

    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    d_img[idx] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);
    int *h_img = (int *)malloc(size);
    int *d_img;
    cudaMalloc((void **)&d_img, size);
    
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(resX / N, resY / N);

    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, d_img, resX, resY, maxIterations);
    
    cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);
    memcpy(img, h_img, size);

    free(h_img);
    cudaFree(d_img);
}
