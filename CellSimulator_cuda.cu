#include "CellSimulator.cuh"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "kernel_impl.cuh"
#include <cuda_runtime.h>

extern "C"
{
	void cudaInit(int argc, char **argv)
	{
		int devID;

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		devID = findCudaDevice(argc, (const char **)argv);

		if (devID < 0)
		{
			printf("No CUDA Capable devices found, exiting...\n");
			exit(EXIT_SUCCESS);
		}
	}

	void allocateArray(void **devPtr, size_t size)
	{
		checkCudaErrors(cudaMalloc(devPtr, size));
	}

	void freeArray(void *devPtr)
	{
		checkCudaErrors(cudaFree(devPtr));
	}

	void threadSync()
	{
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void copyArrayToDevice(void *device, const void *host, int offset, int size)
	{
		checkCudaErrors(cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice));
	}

	void copyArrayFromDevice(void *host, const void *device, int size)
	{
		checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
	}

	uint iDivUp(uint a, uint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
	{
		numThreads = std::min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}

	void calculateHashValue(uint  *gridParticleHash, uint  *gridParticleIndex, float4 *pos, int numParticles)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// execute the kernel
		calculateHashValueDevice << < numBlocks, numThreads >> >(	gridParticleHash,
																	gridParticleIndex,
																	pos,
																	numParticles);
		//cudaThreadSynchronize();
		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed: __calculateHashValueDevice__");
	}

	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
	{
		thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
			thrust::device_ptr<uint>(dGridParticleHash + numParticles),
			thrust::device_ptr<uint>(dGridParticleIndex));

		//cudaThreadSynchronize();
	}


	void integrateSystem(float4 *newPos, float4 * pos, float4 * acc, float deltaTime, uint numParticles, curandState *globalState)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		//execute the kernel
		integrateDevice << < numBlocks, numThreads >> > (newPos, pos, acc, deltaTime, numParticles, globalState);
		
		//cudaThreadSynchronize();

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed: __integrateDevice__ ");
	}

	void setParameters(SimulationParams *hostSimParams, PlaneParams *hostPlaneParams)
	{
		// copy parameters to constant memory
		checkCudaErrors(cudaMemcpyToSymbol(simParams, hostSimParams, sizeof(SimulationParams)));
		checkCudaErrors(cudaMemcpyToSymbol(planeParams, hostPlaneParams, sizeof(PlaneParams)));
	}

	void reorderDataAndFindCellStart(uint  *cellStart,
		uint  *cellEnd,
		float4 *sortedPos,
		float4 *sortedAcc,
		uint  *gridParticleHash,
		uint  *gridParticleIndex,
		float4 *oldPos,
		float4 *oldAcc,
		uint   numParticles,
		uint   numCells)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// set all cells to empty
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));
		checkCudaErrors(cudaMemset(cellEnd, 0xffffffff, numCells*sizeof(uint)));

		uint smemSize = sizeof(uint)*(numThreads + 1);
		reorderDataAndFindCellStartDevice << < numBlocks, numThreads, smemSize >> >(
			cellStart,
			cellEnd,
			sortedPos,
			sortedAcc,
			gridParticleHash,
			gridParticleIndex,
			oldPos,
			oldAcc,
			numParticles);

		//cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: __reorderDataAndFindCellStartDevice_");


	}

	void interact(float4 *newAcc,
		float4 *sortedPos,
		uint  *gridParticleIndex,
		uint  *cellStart,
		uint  *cellEnd,
		uint   numParticles,
		uint   numCells,
		float  damping)
	{

		// thread per particle
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		// execute the kernel
		interactDevice << < numBlocks, numThreads >> >(newAcc,
			sortedPos,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles,
			damping);

		//cudaThreadSynchronize();

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed: __integrateDevice__");
	}

	void initialazeKernelStates(curandState * state, unsigned long seed, uint numParticles)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		initializeCurandOnKernels << < numBlocks, numThreads >> >(state, seed, numParticles);

		getLastCudaError("Kernel execution failed: __initializeCurandOnKernels__");
	}
}