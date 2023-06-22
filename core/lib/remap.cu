__global__ void RemapKernel(float* input, float* output, int width, int height, int* xmap, int* ymap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        int mapped_x = xmap[y * width + x];
        int mapped_y = ymap[y * width + x];

        if(mapped_x >= 0 && mapped_x < width && mapped_y >= 0 && mapped_y < height)
        {
            output[y * width + x] = input[mapped_y * width + mapped_x];
        }
    }
}
