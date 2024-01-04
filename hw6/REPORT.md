# PP-f23 Assignment VI Report
### Q1: Explain your implementation. How do you optimize the performance of convolution?

In `hostFE.c`, I set the global work size as { imageWidth, imageHeight }.

In `kernel.cl`, I use `get_global_id(0)` and `get_global_id(1)` to retrieve the pixel that each work item needs to process. Then, I utilize the `serialConv()` function to perform the convolution calculation for the corresponding pixel.

#### Modify the writing of "Convolution"
In the original `serialConv()` function, there are repetitive if statements within the convolution for-loop that continuously check if the calculations exceed the image boundaries. This results in a significant overhead. 

Here is the original version:
```cpp!
for (k = -halfFilterSize;k <= halfFilterSize;k++) {
   for (l = -halfFilterSize; l <= halfFilterSize; l++) {
        if(filter[(k + halfFilterSize) * filterWidth + l + halfFilterSize] != 0)
        {
            if (row + k >= 0 && row + k < imageHeight &&
                col + l >= 0 && col + l < imageWidth)
            {
                sum += inputImage[(row + k) * imageWidth + col + l] *
                        filter[(k + halfFilterSize) * filterWidth +
                                l + halfFilterSize];
            }
        }
    }
}
outputImage[row * imageWidth + col] = sum;
```

Therefore, I modified the original approach to calculate the starting row and column of the input image first and then use these values in the for-loop conditions. This reduces unnecessary checks and speeds up computation.

The modified approach is as follows:
```cpp!
int row_begin = idy - halfFilterSize >= 0 ? 0 : halfFilterSize - idy;
int row_end = idy + halfFilterSize < imageHeight ? filterWidth - 1 : imageHeight - idy;
int col_begin = idx - halfFilterSize >= 0 ? 0 : halfFilterSize - idx;
int col_end = idx + halfFilterSize < imageWidth ? filterWidth - 1 : imageWidth - idx;

for (int i = row_begin; i <= row_end; i++) {
    int row = idy - halfFilterSize + i;
    int col = idx - halfFilterSize;
    for (int j = col_begin; j <= col_end; j++) {
        sum += inputImage[row * imageWidth + col + j] * filter[i * filterWidth + j];
    }
}

outputImage[idy * imageWidth + idx] = sum;
```

### Q2: Rewrite the program using CUDA. 
1. Explain your CUDA implementation

    The implementation of the algorithm and optimizations in CUDA are similar to OpenCL. Each pixel is processed by a separate thread. 
    
    The convolution part is similar to OpenCL, and here is the hostFE code:

    ```cpp!
    void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
                       float *inputImage, float *outputImage)
    {
        float *d_filter, *d_inputImage, *d_outputImage;
        int filterSize = filterWidth * filterWidth * sizeof(float);
        int inputImageSize = imageHeight * imageWidth * sizeof(int);
        int outputImageSize = inputImageSize;

        cudaMalloc(&d_filter, filterSize);
        cudaMalloc(&d_inputImage, inputImageSize);
        cudaMalloc(&d_outputImage, outputImageSize);

        // cp mem to device
        cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_inputImage, inputImage, inputImageSize, cudaMemcpyHostToDevice);

        int block_size = 16;

        dim3 threadsPerBlock(block_size, block_size);
        dim3 numBlocks(imageWidth / block_size, imageHeight / block_size);
        // kernel
        convolution<<<threadsPerBlock, numBlocks>>>(filterWidth, d_filter, imageHeight, imageWidth, d_inputImage, d_outputImage);

        // cp mem to host
        cudaMemcpy(outputImage, d_outputImage, outputImageSize, cudaMemcpyDeviceToHost);

        // free mem
        cudaFree(d_outputImage);
        cudaFree(d_inputImage);
        cudaFree(d_filter);
    }
    ```

2. Plot a chart to show the performance difference between using OpenCL and CUDA

    ![image](https://hackmd.io/_uploads/ryZZCEEu6.png)

3. Explain the result

    From the chart above, it can be observed that CUDA is slightly slower than OpenCL. I believe the reason for this is that when using CUDA, the data transfer to the GPU may be slightly slower. This could be because in CUDA, the input image has already been allocated with malloc, so it's not possible to use `cudaHostAlloc()` for optimization in the implementation.




