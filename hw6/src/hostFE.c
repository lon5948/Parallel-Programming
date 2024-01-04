#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    size_t imageSize = imageHeight * imageWidth * sizeof(float);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, &status);

    cl_mem filterBuffer = clCreateBuffer(*context, 0, filterSize, NULL, &status);
    cl_mem inputBuffer = clCreateBuffer(*context, 0, imageSize, NULL, &status);
    cl_mem outputBuffer = clCreateBuffer(*context, 0, imageSize, NULL, &status);

    clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_MEM_READ_ONLY, 0, imageSize, inputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, filterBuffer, CL_MEM_READ_ONLY, 0, filterSize, filter, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&filterBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&filterWidth);

    size_t globalSize[2] = {imageWidth, imageHeight};
    clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, imageSize, outputImage, 0, NULL, NULL);

    clReleaseCommandQueue(commandQueue);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseKernel(kernel);
}