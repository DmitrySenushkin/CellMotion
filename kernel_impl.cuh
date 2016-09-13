
#ifndef KERNEL_IMPL_CUH
#define KERNEL_IMPL_CUH

#include "SimulationParams.cuh"
#include "PlaneParams.cuh"
//#include <device_functions.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"


__constant__ SimulationParams simParams;
__constant__ PlaneParams planeParams;


__global__
void initializeCurandOnKernels(curandState * state, unsigned long seed, uint numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index < numParticles)
	{
		curand_init(seed, index, 0, &state[index]);
	}
}

__device__
float generate(curandState* globalState, int ind)
{
	//copy state to local mem
	curandState localState = globalState[ind];
	//apply uniform distribution with calculated random
	float rndval = curand_uniform(&localState);
	//update state
	globalState[ind] = localState;
	//return value
	return rndval;
}

__device__
int3 calculateGridPosition(float3 p)
{
	int3 gridPos;
	gridPos.x = floor(p.x / simParams.cellSize.x);	// 2.0f - cell size
	gridPos.y = floor(p.y / simParams.cellSize.y);
	gridPos.z = floor(p.z / simParams.cellSize.z);
	return gridPos;
}

__device__
uint calculateGridHashValue(int3 gridPos)
{
	return __umul24(__umul24(gridPos.z, simParams.gridSize.y), simParams.gridSize.x) + __umul24(gridPos.y, simParams.gridSize.x) + gridPos.x;
}

__global__
void calculateHashValueDevice(	uint   *gridParticleHash,  // output
								uint   *gridParticleIndex, // output
								float4 *pos,               // input: positions
								uint    numParticles)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	volatile float4 p = pos[index];

	// get address in grid
	int3 gridPos = calculateGridPosition(make_float3(p.x, p.y, p.z));
	uint hash = calculateGridHashValue(gridPos);



	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;

}



// interaction between two particles
__device__
float3 interactParticles(float3 posA, float3 posB, float damping)
{
	float3 relPos = posB - posA;

	float dist = length(relPos);

	float3 force = make_float3(0.0f);
	float3 norm = make_float3(0.0f);

	norm = relPos / dist;
	force = 12 * simParams.bindingEnergy / std::pow(simParams.bindingLenght, 2) *
	(std::pow(simParams.bindingLenght / dist, 14) - std::pow(simParams.bindingLenght / dist, 3)) * relPos;

	if (length(force) > 3.0f)
	{
		force /= length(force);
		return force;
	}

	return damping * force;
}

__device__
float3 interactCell(int3    gridPos,
					uint    index,
					float3  pos,
					float4 *oldPos,
					uint   *cellStart,
					uint   *cellEnd,
					float	damping)
{
	uint gridHash = calculateGridHashValue(gridPos);

	// get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j = startIndex; j<endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = make_float3(FETCH(oldPos, j));

				// interact two spheres
				force += interactParticles(pos, pos2, damping);
			}
		}
	}

	return force;
}

__global__
void interactDevice(float4 *newAcc,               // output: new acceleration
					float4 *oldPos,               // input: sorted positions
					uint   *gridParticleIndex,    // input: sorted particle indices
					uint   *cellStart,
					uint   *cellEnd,
					uint    numParticles,
					float	damping)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles) return;

	// read particle data from sorted arrays
	float3 pos = make_float3(FETCH(oldPos, index));

	// get address in grid
	int3 gridPos = calculateGridPosition(pos);

	// examine neighbouring cells
	float3 force = make_float3(0.0f);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				force += interactCell(neighbourPos, index, pos, oldPos, cellStart, cellEnd, damping);
			}
		}
	}

	// calculate index in sorted array
	uint originalIndex = gridParticleIndex[index];

	// add friction
	float4 vel = oldPos[originalIndex] - oldPos[originalIndex + numParticles];
	force -= 0.4f * make_float3(vel.x, vel.y, vel.z);

	// add repulsion from plane
	float koef_A = planeParams.normal.x;
	float koef_B = planeParams.normal.y;
	float koef_C = planeParams.normal.z;
	float koef_D =	planeParams.normal.x * planeParams.point.x +
					planeParams.normal.y * planeParams.point.y +
					planeParams.normal.z * planeParams.point.z;

	float dist = (koef_A * oldPos[originalIndex].x +
				koef_B * oldPos[originalIndex].y +
				koef_C * oldPos[originalIndex].z + koef_D) /
				std::sqrt(koef_A*koef_A + koef_B*koef_B + koef_C*koef_C);

	float3 planeForce = 50 * planeParams.normal / dist;

	// write new acceleration back to original unsorted location
	newAcc[originalIndex] = make_float4(force + 
										simParams.gravity +
										planeForce,
										0.0f);
}

__device__
float3 calculateNewPosition(float3 pos, 
							float3 prePos, 
							float3 acc, 
							float deltaTime,
							float3 rand)
{
	return 2 * pos - prePos + acc * deltaTime * deltaTime;
}

__global__
void integrateDevice(float4 *newP, float4 * pos, float4 * acc, float deltaTime, uint numParticles, curandState *globalState)
{
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float3 rand = make_float3(generate(globalState, index) * 0.3f, generate(globalState, index) * 0.3f, generate(globalState, index) * 0.3f);

	if (index < numParticles)
	{
		float3 newPos = calculateNewPosition(make_float3(pos[index].x, pos[index].y, pos[index].z),
			make_float3(pos[index + numParticles].x, pos[index + numParticles].y, pos[index + numParticles].z),
			make_float3(acc[index].x, acc[index].y, acc[index].z),
			deltaTime, rand);

		float4 prepos = FETCH(pos, index);

		/*if (newPos.z < 10)
		{
			newPos.z = 10;
			//prepos.z = 10;
		}
		if (newPos.x < 10)
		{
			newPos.x = 10;
			//prepos.x = 10;
		}
		if (newPos.y < 10)
		{
			newPos.y = 10;
			//prepos.y = 10;
		}

		if (newPos.z > simParams.gridSize.z * simParams.cellSize.z)
		{
			newPos.z = (simParams.gridSize.z - 1) * simParams.cellSize.z;
			prepos.z = (simParams.gridSize.z - 1) * simParams.cellSize.z;
		}
		if (newPos.x > simParams.gridSize.x * simParams.cellSize.x)
		{
			newPos.x = (simParams.gridSize.x - 1) * simParams.cellSize.x;
			prepos.x = (simParams.gridSize.x - 1) * simParams.cellSize.x;
		}
		if (newPos.y > simParams.gridSize.y * simParams.cellSize.y)
		{
			newPos.y = (simParams.gridSize.y - 1) * simParams.cellSize.y;
			prepos.y = (simParams.gridSize.y - 1) * simParams.cellSize.y;
		}

		*/
		newP[index + numParticles] = prepos;
		newP[index] = make_float4(newPos, 0.0f);
	}
}

__global__
void reorderDataAndFindCellStartDevice(	uint   *cellStart,        // output: cell start index
										uint   *cellEnd,          // output: cell end index
										float4 *sortedPos,        // output: sorted positions
										float4 *sortedAcc,        // output: sorted accelerations
										uint   *gridParticleHash, // input: sorted grid hashes
										uint   *gridParticleIndex,// input: sorted particle indices
										float4 *oldPos,           // input: sorted position array
										float4 *oldAcc,           // input: sorted acceleration array
										uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
		float4 prepos = FETCH(oldPos, sortedIndex + numParticles);
		float4 acc = FETCH(oldAcc, sortedIndex);       // see particles_kernel.cuh

		sortedPos[index] = pos;
		sortedPos[index + numParticles] = prepos;
		sortedAcc[index] = acc;
	}

}

#endif