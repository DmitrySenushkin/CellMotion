#include "CellSimulator.cuh"
#include "PlaneParams.cuh"
#include <fstream>
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

class CellSimulator
{
public:

	CellSimulator(uint _numParticles, uint3 gridSize);
	~CellSimulator();

	void update(float deltaTime);
	void output(uint numIteration);
	void reset();

protected:

	CellSimulator() {};
	void _initialize(int m_numParticles);
	void _finalize();

	void initializeGrid(uint *size, float spacing, float jitter, uint numParticles);

private:

	// data for cuRAND
	curandState* device_States;

	// data
	uint numParticles;

	// data for initialization grid
	// position data : CPU
	float4 * host_Position;
	float4 * host_Acceleration;

	// position data : GPU
	float4 * device_Position;
	float4 * device_Acceleration;

	float4 * device_cudaPosVBO;

	// data for computing Verle's alghorithm
	float4 * device_SortedPosition;
	float4 * device_SortedAcceleration;

	// hash data : CPU
	uint  *host_ParticleHash;
	uint  *host_CellStart;
	uint  *host_CellEnd;

	// grid data for sorting method
	uint  *device_GridParticleHash; // grid hash value for each particle
	uint  *device_GridParticleIndex;// particle index for each particle
	uint  *device_CellStart;        // index of start of each cell in sorted list
	uint  *device_CellEnd;          // index of end of cell

	SimulationParams simParams;		// simulation params
	PlaneParams planeParams;		// plane params
	uint3	gridSize;
	uint	numGridCells;

	uint m_solverIterations;
};