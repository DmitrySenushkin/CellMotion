#include "vector_types.h"
#include "SimulationParams.cuh"

#ifndef PLANE_PARAMS_CUH
#define PLANE_PARAMS_CUH

struct PlaneParams
{
	float3 normal;
	float3 point;
};


#endif