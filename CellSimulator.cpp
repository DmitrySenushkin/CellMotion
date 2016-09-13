#include "CellSimulator.h"
#include <memory.h>
#include <helper_cuda.h>
#include <helper_functions.h>
//#include "kernel_impl.cuh"  // НЕ ВКЛЮЧАТЬ !!! cuda файлы в срр не могут иметь включения, компилятор не узнает с++


CellSimulator::CellSimulator(uint _numParticles, uint3 gridSize) :
									numParticles(_numParticles),
									host_Position(0),
									//host_Acceleration(0),
									device_Position(0),
									device_Acceleration(0),
									gridSize(gridSize),
									m_solverIterations(1)
{
	numGridCells = gridSize.x*gridSize.y*gridSize.z;

	// set simulation parameters
	simParams.gridSize = gridSize;
	simParams.numCells = numGridCells;
	simParams.cellSize = make_float3(8.0f, 5.0f, 5.0f);
	simParams.bindingEnergy = 1;
	simParams.bindingLenght = 0.1;
	simParams.damping = 0.2f;
	simParams.boundaryDamping = -0.5f;
	simParams.gravity = make_float3(0, 0, -0.3f);

	// set plane parameters
	planeParams.normal = make_float3(0, 0, 1);
	planeParams.point = make_float3(10, 10, 10);

	_initialize(_numParticles);
}

CellSimulator::~CellSimulator()
{
	_finalize();
	numParticles = 0;
}

void CellSimulator::_initialize(int m_numParticles)
{
	numParticles = m_numParticles;

	// allocate host storage
	host_Position = new float4[numParticles * 2];
	host_Acceleration = new float4[numParticles];
	host_CellStart = new uint[numGridCells];
	host_CellEnd = new uint[numGridCells];

	memset(host_Position, 0, numParticles * 2 * sizeof(float4));
	memset(host_Acceleration, 0, numParticles * sizeof(float4));
	memset(host_CellStart, 0, numGridCells * sizeof(uint));
	memset(host_CellEnd, 0, numGridCells * sizeof(uint));

	// allocate GPU data
	allocateArray((void **)&device_Position, numParticles * 2 * sizeof(float4));
	allocateArray((void **)&device_Acceleration, numParticles * sizeof(float4));


	allocateArray((void **)&device_SortedPosition, numParticles * 2 * sizeof(float4));
	allocateArray((void **)&device_SortedAcceleration, numParticles * sizeof(float4));

	allocateArray((void **)&device_GridParticleHash, numParticles * sizeof(uint));
	allocateArray((void **)&device_GridParticleIndex, numParticles * sizeof(uint));

	allocateArray((void **)&device_CellStart, numGridCells*sizeof(uint));
	allocateArray((void **)&device_CellEnd, numGridCells*sizeof(uint));




	uint numThreads, numBlocks;
	computeGridSize(numParticles, 256, numBlocks, numThreads);

	allocateArray((void **)&device_States, numBlocks * numThreads * sizeof(curandState));
	
	initialazeKernelStates(device_States, time(NULL), numParticles);

}

void CellSimulator::_finalize()
{

	delete[] host_Position;
	delete[] host_Acceleration;
	delete[] host_CellStart;
	delete[] host_CellEnd;

	freeArray(device_Position);
	freeArray(device_Acceleration);
	freeArray(device_SortedPosition);
	freeArray(device_SortedAcceleration);

	freeArray(device_GridParticleHash);
	freeArray(device_GridParticleIndex);
	freeArray(device_CellStart);
	freeArray(device_CellEnd);
}

void CellSimulator::update(float deltaTime)
{

	// update constants
	//setParameters(&this->simParams);

	// calculate hash value for each particles
	calculateHashValue(device_GridParticleHash,
		device_GridParticleIndex,
		device_Position,
		numParticles); // tested 

	// sort particles based on hash
	sortParticles(device_GridParticleHash, device_GridParticleIndex, numParticles); // tested

	// reorder data based on hash and index
	reorderDataAndFindCellStart(device_CellStart,
		device_CellEnd,
		device_SortedPosition,
		device_SortedAcceleration,
		device_GridParticleHash,
		device_GridParticleIndex,
		device_Position,
		device_Acceleration,
		numParticles,
		numGridCells);	// tested

	// process interaction
	interact(device_Acceleration,
		device_SortedPosition,
		device_GridParticleIndex,
		device_CellStart,
		device_CellEnd,
		numParticles,
		numGridCells,
		0.4f);

	// calculation
	integrateSystem(device_Position,
		device_SortedPosition,
		device_Acceleration,
		deltaTime,
		numParticles,
		device_States); // tested


}

void CellSimulator::output(uint numIteration)
{
	std::ofstream file;
	std::string outPath("D:/My Programms/CellMotion/results/output.csv.");
	outPath += std::to_string(numIteration);

	file.open(outPath, std::ios_base::out);
	file << "x,y,z\n";
	if (numIteration != 0)
	{

		copyArrayFromDevice(host_Position, device_Position,numParticles * sizeof(float4));
		//copyArrayFromDevice(host_ParticleHash, device_GridParticleHash, numParticles * sizeof(uint));
		//copyArrayFromDevice(host_GridPos, device_GridPos, numParticles * sizeof(int3));
	}
	for (uint i = 0; i < numParticles; i++)
	{
		file << host_Position[i].x << "," << host_Position[i].y << "," << host_Position[i].z << "\n";
		//file << host_GridPos[i].x << "," << host_GridPos[i].y << "," << host_GridPos[i].z << "\n";
		//file << host_ParticleHash[i] << "\n";
	}


	file.close();

}

void CellSimulator::reset()
{
	uint gridSize[3];
	gridSize[0] = gridSize[1] = gridSize[2] = 16;

	initializeGrid(gridSize, 1.0f, 0.3f, numParticles);
	setParameters(&simParams, &planeParams);
}

inline float frand()
{
	return rand() / (float)RAND_MAX;
}

void CellSimulator::initializeGrid(uint *size, float spacing, float jitter, uint numParticles)
{
	srand(1973);
	uint cnt = 0;

	for (uint z = 0; z<size[2]; z++)
	{
		for (uint y = 0; y<size[1]; y++)
		{
			for (uint x = 0; x<size[0]; x++)
			{

				if (cnt < numParticles)
				{
					host_Position[cnt].x = (spacing * x) + (simParams.gridSize.x / 2 * simParams.cellSize.x); //+ (frand()*2.0f - 1.0f)*jitter;
					host_Position[cnt].y = (spacing * y) + (simParams.gridSize.y / 2 * simParams.cellSize.y);//+ (frand()*2.0f - 1.0f)*jitter;
					host_Position[cnt].z = (spacing * z) + (simParams.gridSize.z / 2 * simParams.cellSize.z);//+ (frand()*2.0f - 1.0f)*jitter;
					
					/*host_Position[cnt + numParticles].x = (spacing * x) + (frand()*2.0f - 1.0f)*jitter + 152;
					host_Position[cnt + numParticles].y = (spacing * y) + (frand()*2.0f - 1.0f)*jitter + 152;
					host_Position[cnt + numParticles].z = (spacing * z) + (frand()*2.0f - 1.0f)*jitter + 152;
					*/
					host_Position[cnt + numParticles] = host_Position[cnt];
					cnt++;
				}
			}
		}
	}

	copyArrayToDevice(device_Position, host_Position, 0, 2 * numParticles * sizeof(float4));
}
