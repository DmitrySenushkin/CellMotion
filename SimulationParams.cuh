#include <stdlib.h>
#include "vector_types.h"
#include <curand.h>
#include <curand_kernel.h>

#ifndef PARAMS_CUH
#define PARAMS_CUH

#define FETCH(t, i) t[i]

typedef unsigned int uint;


struct SimulationParams
{
	uint3	gridSize;
	uint	numCells;
	float3	cellSize;

	float3	gravity;
	float   bindingEnergy;
	float	bindingLenght;
	float	damping;
	float	boundaryDamping;
};

#endif // PARAMS_CUH