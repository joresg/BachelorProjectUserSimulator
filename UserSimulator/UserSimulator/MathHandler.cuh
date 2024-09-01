#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CUDAMatrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <math.h>
#include <stdexcept>
#include <queue>
#include <chrono>
#include <vector>
#include <tuple>

#pragma once
class MathHandler {
public:
	MathHandler(unsigned long randomSeed);
	CUDAMatrix TransposeMatrix(CUDAMatrix inputMatrix);
	double GenerateRandomNumber(double min, double max);
	//CUDAMatrix TanhDerivative(CUDAMatrix inputMatrix);
	CUDAMatrix ActFuncDerivative(CUDAMatrix inputMatrix, LayerActivationFuncs actFunc);
	std::default_random_engine GetRandomEngine() { return _re; }
	std::vector<CUDAMatrix> CreateOneHotEncodedVectorSequence(std::vector<int> cmdIDs, int allClasses);
	std::tuple<std::vector<CUDAMatrix>, std::vector<CUDAMatrix>> CreateBatchOneHotEncodedVector(std::vector<std::vector<int>> cmdIDs, CUDAMatrix classMasks, int allClasses, int batchSize);
	std::vector<CUDAMatrix> CreateClassesMask(std::vector<CUDAMatrix> oneHotEncodedVectors, int allClasses, int batchSize);
	CUDAMatrix CreateOneHotEncodedVector(int cmdID, int allClasses);
	int OneHotEncodedVectorToClassID(CUDAMatrix oheInput);
private:
	unsigned long _randSeed;
	std::uniform_real_distribution<double> _unif;
	std::default_random_engine _re;
	dim3 _blockSize;
};