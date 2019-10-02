
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

__device__ unsigned char max(unsigned char a, unsigned char b)
{
	return a > b ? a : b;
}

__global__ void poolKernel(union pixel *inputImage, union pixel *outputImage, unsigned width, unsigned height, dim3 partitionSize)
{
	unsigned partitionX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned partitionY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned partitionZ = blockIdx.z;

	unsigned startX = partitionX * partitionSize.x, endX = (partitionX + 1) * partitionSize.x;
	unsigned startY = partitionY * partitionSize.y, endY = (partitionY + 1) * partitionSize.y;
	unsigned startZ = partitionZ * partitionSize.z, endZ = (partitionZ + 1) * partitionSize.z;

	for (unsigned pixelX = startX; pixelX < width && pixelX < endX; pixelX++)
	{
		for (unsigned pixelY = startY; pixelY < height && pixelY < endY; pixelY++)
		{
			for (unsigned channel = startZ; channel < 3 && channel < endZ; channel++)
			{
				unsigned char* output = &(outputImage[pixelX + width * pixelY].channels[channel]);

				*output = inputImage[2 * pixelX + 2 * width * (2 * pixelY)].channels[channel];
				*output = max(*output, inputImage[2 * pixelX + 1 + 2 * width * (2 * pixelY)].channels[channel]);
				*output = max(*output, inputImage[2 * pixelX + 2 * width * (2 * pixelY + 1)].channels[channel]);
				*output = max(*output, inputImage[2 * pixelX + 1 + 2 * width * (2 * pixelY + 1)].channels[channel]);
			}
		}
	}
}

unsigned getGpuProps(cudaDeviceProp* deviceProp)
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

dim3 partitionThreads(unsigned numberOfThreads, unsigned width, unsigned height)
{
	unsigned z = 1;

	if (numberOfThreads > width * height)
	{
		numberOfThreads /= 3;
		z = 3;
	}

	unsigned power = (unsigned)log2(numberOfThreads);
	return dim3(1 << (power / 2 + power % 2), 1 << (power / 2), z);
}

int main(int argc, char* argv[])
{
	if (argc != 4) {
		fprintf(stderr, "You should respect this format: ./Pool <name of input png> <name of output png> < # threads>");
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

	union pixel* imageData;
	unsigned inputWidth, inputHeight;

	unsigned returnCode = lodepng_decode24_file((unsigned char**)& imageData, &inputWidth, &inputHeight, inputFileName);

	if (returnCode != 0) {
		fprintf(stderr, "Error: reading the image file");
		return;
	}

	size_t pixelBufferSize = inputWidth * inputHeight * sizeof(union pixel);
	union pixel* deviceInputImageData;
	union pixel* deviceOutputImageData;
	union pixel* hostResultImageData = (union pixel*)calloc(inputWidth * inputHeight / 4, sizeof(union pixel));

	cudaMalloc((void**)& deviceInputImageData, pixelBufferSize);
	cudaMalloc((void**)& deviceOutputImageData, pixelBufferSize / 4);
	cudaMemcpy(deviceInputImageData, imageData, pixelBufferSize, cudaMemcpyHostToDevice);

	dim3 threads = partitionThreads(numberOfThreads, inputWidth / 2, inputHeight / 2);

	unsigned threadDimension = (unsigned)sqrtf(deviceProps.maxThreadsPerBlock);
	unsigned xThreadsPerBlock = (unsigned)fminf(threadDimension, threads.x);
	unsigned yThreadsPerBlock = (unsigned)fminf(threadDimension, threads.y);

	dim3 threadLayout = dim3(xThreadsPerBlock, yThreadsPerBlock);
	dim3 blockLayout = dim3((threads.x + xThreadsPerBlock - 1) / xThreadsPerBlock, (threads.y + yThreadsPerBlock - 1) / yThreadsPerBlock, threads.z);

	dim3 partitionSize = dim3((inputWidth / 2 + threads.x - 1) / threads.x, (inputHeight / 2 + threads.y - 1) / threads.y, 3 / threads.z);

	poolKernel<<<blockLayout, threadLayout>>>(deviceInputImageData, deviceOutputImageData, inputWidth / 2, inputHeight / 2, partitionSize);

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

	cudaStatus = cudaMemcpy(hostResultImageData, deviceOutputImageData, pixelBufferSize / 4, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, cudaGetErrorString(cudaStatus));
	}

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	// Save the image back to disk
	lodepng_encode24_file(outputFileName, (unsigned char*)hostResultImageData, inputWidth / 2, inputHeight / 2);

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



