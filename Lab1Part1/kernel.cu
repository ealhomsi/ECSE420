
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

__global__ void rectifyKernel(union pixel *image, unsigned width, unsigned height)
{
	unsigned pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned pixelY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned channelIdx = blockIdx.z;

	if (pixelX < width && pixelY < height)
	{
		unsigned pixelIndex = pixelX + width * pixelY;
		unsigned char* channel = &image[pixelIndex].channels[channelIdx];
		
		if (*channel <= 127)
		{
			*channel = 0;
		}
		else
		{
			*channel -= 127;
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

int main()
{
	cudaDeviceProp deviceProps;
	unsigned errorCode = getGpuProps(&deviceProps);
	if (errorCode > 0)
	{
		return errorCode;
	}

	union pixel *imageData;
	unsigned width, height;
	const char* inputFilePath = "C:\\capture.png";
	const char* outputFilePath = "C:\\rectified.png";

	unsigned returnCode = lodepng_decode24_file((unsigned char **)&imageData, &width, &height, inputFilePath);
	if (returnCode > 0) {
		fprintf(stderr, "file open issues");
		return;
	}

	size_t pixelBufferSize = width * height * sizeof(union pixel);
	union pixel *deviceImageData;
	union pixel* hostResultImageData = (union pixel*)calloc(width * height, sizeof(union pixel));

	cudaMalloc((void**)& deviceImageData, pixelBufferSize);
	cudaMemcpy(deviceImageData, imageData, pixelBufferSize, cudaMemcpyHostToDevice);

	unsigned threadDimension = (unsigned)sqrt((double)deviceProps.maxThreadsPerBlock);

	dim3 threadsPerBlock(threadDimension, threadDimension);
	dim3 numberOfBlocks(
		(width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y,
		3);
	
	rectifyKernel<<<numberOfBlocks, threadsPerBlock>>>(deviceImageData, width, height);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(hostResultImageData, deviceImageData, pixelBufferSize, cudaMemcpyDeviceToHost);
	cudaFree(deviceImageData);

	// Save the image back to disk
	lodepng_encode24_file(outputFilePath, (unsigned char *)hostResultImageData, width, height);

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
