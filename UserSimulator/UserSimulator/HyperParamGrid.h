#include "MathHandler.cuh"

#pragma once
class HyperParamGrid
{
private:
	std::vector<double> _learningRate;
	std::vector<std::vector<std::tuple<int, LayerActivationFuncs>>> _hiddenUnits;
	std::vector<int> _sequenceLength;
	std::vector<int> _batchSize;
	//todo dropout rate
	//todo weight init uniform, normal, xavier, he....
public:
	HyperParamGrid(int allClasses, GatedUnits gatedCells);

	std::vector<std::tuple<double, std::vector<std::tuple<int, LayerActivationFuncs>>, int, int>> HyperParameterGridSearch();
};

