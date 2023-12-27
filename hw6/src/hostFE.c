#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    
    int imageSize = imageHeight * imageWidth;
    size_t dataSize = imageSize * sizeof(float);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, &status);
    
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, dataSize, inputImage, &status);
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filterSize, filter, &status);
    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&filterBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&filterWidth);

    size_t global_work_size = imageSize;
    size_t local_work_size = 64;
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, dataSize, outputImage, 0, NULL, NULL);

    clReleaseCommandQueue(commandQueue);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseKernel(kernel);
}