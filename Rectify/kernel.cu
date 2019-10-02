
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>
#include <math.h>

union pixel {
	unsigned char channels[3];
	struct {
		unsigned char r;
		unsigned char g;
		unsigned char b;
	} colors;
};

__global__ void rectifyKernel(union pixel *image, unsigned size, dim3 partitionSize)
{
	unsigned partitionX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned partitionZ = blockIdx.z;

	unsigned startX = partitionX * partitionSize.x, endX = (partitionX + 1) * partitionSize.x;
	unsigned startZ = partitionZ * partitionSize.z, endZ = (partitionZ + 1) * partitionSize.z;

	for (unsigned pixelX = startX; pixelX < size && pixelX < endX; pixelX++)
	{
		for (unsigned channel = startZ; channel < 3 && channel < endZ; channel++)
		{
			unsigned char* value = &(image[pixelX].channels[channel]);

			if (*value < 127)
			{
				*value = 127;
			}
		}
	}
}

unsigned getGpuProps(cudaDeviceProp *deviceProp)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		printf("There is no device supporting CUDA\n");
		return 1;
	}

	cudaSetDevice(0);
	cudaGetDeviceProperties(deviceProp, 0);
	return 0;
}

dim3 partitionThreads(unsigned numberOfThreads, size_t size)
{
	if (numberOfThreads <= size)
	{
		return dim3(numberOfThreads, 1, 1);
	}
	else
	{
		// Handle 1 channel per thread, this may actually use less threads than the number passed in
		// but the partitioning scheme will not work unless the number of threads is a multiple of 3
		// whenever numberOfThreads > size
		return dim3(numberOfThreads / 3, 1, 3);
	}
}

int main(int argc, char* argv[])
{
	if (argc != 4) {
		fprintf(stderr, "You should respect this format: ./rectify <name of input png> <name of output png> < # threads>");
		return 2;
	}
	char* inputFileName = argv[1];
	char* outputFileName = argv[2];
	unsigned numberOfThreads = atoi(argv[3]);

	cudaDeviceProp deviceProps;
	unsigned errorCode = getGpuProps(&deviceProps);
	if (errorCode > 0)
	{
		return errorCode;
	}


	union pixel *imageData;
	unsigned width, height;

	unsigned returnCode = lodepng_decode24_file((unsigned char **)&imageData, &width, &height, inputFileName);

	size_t pixelBufferSize = width * height * sizeof(union pixel);
	union pixel *deviceImageData;
	union pixel* hostResultImageData = (union pixel*)calloc(width * height, sizeof(union pixel));

	cudaMalloc((void**)& deviceImageData, pixelBufferSize);
	cudaMemcpy(deviceImageData, imageData, pixelBufferSize, cudaMemcpyHostToDevice);

	dim3 threads = partitionThreads(numberOfThreads, width * height);

	unsigned xThreadsPerBlock = (unsigned)fminf(deviceProps.maxThreadsPerBlock, threads.x);

	dim3 threadLayout = dim3(xThreadsPerBlock);
	dim3 blockLayout = dim3((threads.x + xThreadsPerBlock - 1) / xThreadsPerBlock, 1, threads.z);

	dim3 partitionSize = dim3((width * height + threads.x - 1) / threads.x, 1, 3 / threads.z);
	
	rectifyKernel<<<blockLayout, threadLayout>>>(deviceImageData, width * height, partitionSize);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(hostResultImageData, deviceImageData, pixelBufferSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, cudaGetErrorString(cudaStatus));
	}

	cudaFree(deviceImageData);

	// Save the image back to disk
	lodepng_encode24_file(outputFileName, (unsigned char *)hostResultImageData, width, height);

	free(hostResultImageData);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
