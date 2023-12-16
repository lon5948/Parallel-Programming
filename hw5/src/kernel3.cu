#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define GROUP_SIZE 4

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_img, size_t pitch, int resX, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    float c_im = lowerY + thisY * stepY;
    for (int g = 0; g < GROUP_SIZE; g++) {
        float c_re = lowerX + thisX * stepX;
        float z_re = c_re, z_im = c_im;

        int i;
        for (i = 0; i < maxIterations; ++i) {

            if (z_re * z_re + z_im * z_im > 4.f)
                break;

            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }

        *((int*)((char*)d_img + thisY * pitch) + thisX) = i;
        thisX++;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int size = resX * resY * sizeof(int);
    int *h_img, *d_img;
    
    size_t pitch;
    cudaHostAlloc((void **)&h_img, size, cudaHostAllocDefault);
    cudaMallocPitch((void **)&d_img, &pitch, resX * sizeof(int), resY);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(resX / (BLOCK_SIZE * GROUP_SIZE), resY / BLOCK_SIZE);

    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, d_img, pitch, resX, maxIterations);
    
    cudaMemcpy2D(h_img, resX * sizeof(int), d_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, h_img, size);

    cudaFreeHost(h_img);
    cudaFree(d_img);
}
