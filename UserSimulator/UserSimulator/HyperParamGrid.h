#include "MathHandler.cuh"

#pragma once
class HyperParamGrid
{
private:
	std::vector<double> _learningRate;
	std::vector<int> _hiddenUnits;
	std::vector<int> _sequenceLength;
	std::vector<int> _batchSize;
	//todo dropout rate
	//todo weight init uniform, normal, xavier, he....
public:
	HyperParamGrid(int allClasses);

	std::vector<std::tuple<double, int, int, int>> HyperParameterGridSearch();
};

