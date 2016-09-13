#include "CellSimulator.h"
#include <cstdio>

#define GRID_SIZE       256
#define NUM_PARTICLES   4096


CellSimulator * pSim = 0;

uint numParticles = 0;
uint3 gridSize;
uint numIterations = 10000;

float	gravity = 0.0003f;
float   bindingEnergy = 1.0f;
float	bindingLenght = 1.0f;;
float	damping = 1.0f;
float	boundaryDamping = 1.0f;

float timestep = 1.0f;
unsigned int g_TotalErrors = 0;

const char *SimName = "CUDA Cell Simulation";

void initializeCellSimulator(int numParticles, uint3 gridSize)
{
	pSim = new CellSimulator(numParticles, gridSize);
	pSim->reset();
}

void run(int iterations)
{
	printf("Run %u particles simulation for %d iterations...\n\n", NUM_PARTICLES, iterations);
	cudaDeviceSynchronize();

	for (int i = 0; i < iterations; ++i)
	{
		pSim->update(timestep);
		pSim->output(i);
	}

	cudaDeviceSynchronize();
}

int main()
{
	printf("%s Starting...\n\n", SimName);

	numParticles = NUM_PARTICLES;
	uint gridDim = GRID_SIZE;

	gridSize.x = gridSize.y = gridSize.z = gridDim;
	printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z);
	printf("particles: %d\n", numParticles);

	initializeCellSimulator(numParticles, gridSize);

	run(numIterations);

	if (pSim)
	{
		delete pSim;
	}
	cudaDeviceReset();
	system("pause");
	exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}

