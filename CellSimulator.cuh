#include "SimulationParams.cuh"
#include "PlanePArams.cuh"
#include <curand.h>
#include <curand_kernel.h>

extern "C"
{
	void cudaInit(int argc, char **argv);
	void allocateArray(void **devPtr, size_t size);
	void freeArray(void *devPtr);
	void threadSync();
	void copyArrayFromDevice(void *host, const void *device, int size);
	void copyArrayToDevice(void *device, const void *host, int offset, int size);
	uint iDivUp(uint a, uint b);
	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);
	void calculateHashValue(uint  *gridParticleHash, uint  *gridParticleIndex, float4 *pos, int m_numParticles);
	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
	void integrateSystem(float4 *newPos, float4 * pos, float4 * acc, float deltaTime, uint numParticles, curandState *globalState);
	void setParameters(SimulationParams *hostSimParams, PlaneParams *hostPlaneParams);
	void initialazeKernelStates(curandState * states, unsigned long seed, uint numParticles);
	void reorderDataAndFindCellStart(uint  *cellStart,
		uint  *cellEnd,
		float4 *sortedPos,
		float4 *sortedAcc,
		uint  *gridParticleHash,
		uint  *gridParticleIndex,
		float4 *oldPos,
		float4 *oldAcc,
		uint   numParticles,
		uint   numCells);
	void interact(float4 *newAcc,
		float4 *Pos,
		uint  *gridParticleIndex,
		uint  *cellStart,
		uint  *cellEnd,
		uint   numParticles,
		uint   numCells,
		float  damping);
}
