__kernel void convolution(const __global float *inputImage, __global float *outputImage, __constant float *filter,
                          const int imageHeight, const int imageWidth, const int filterWidth) 
{
    int halfFilterSize = filterWidth / 2;
    float sum = 0.0;

    int idx = get_global_id(0);
    int idy = get_global_id(1);
    
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
}