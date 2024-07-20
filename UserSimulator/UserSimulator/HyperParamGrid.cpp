#include "HyperParamGrid.h"
#include <tuple>


HyperParamGrid::HyperParamGrid(int allClasses) {
	_learningRate.push_back(0.001);
	_learningRate.push_back(0.0001);
	_learningRate.push_back(0.00001);
	_learningRate.push_back(0.000001);

	_hiddenUnits.push_back(allClasses / 2);
	_hiddenUnits.push_back(allClasses);
	_hiddenUnits.push_back(allClasses + (allClasses / 2));
	_hiddenUnits.push_back(allClasses * 2);

	_sequenceLength.push_back(5);
	_sequenceLength.push_back(10);
	_sequenceLength.push_back(15);
	_sequenceLength.push_back(20);
	_sequenceLength.push_back(25);
	_sequenceLength.push_back(30);

	_batchSize.push_back(8);
	_batchSize.push_back(16);
	_batchSize.push_back(32);
	_batchSize.push_back(64);
}

std::vector<std::tuple<double, int, int, int>> HyperParamGrid::HyperParameterGridSearch() {
	std::vector<std::tuple<double, int, int, int>> allHyperParamCombinations;

	for (const auto& lr : _learningRate) {
		for (const auto& hu : _hiddenUnits) {
			for (const auto& sl : _sequenceLength) {
				for (const auto& bs : _batchSize) {
					allHyperParamCombinations.push_back(std::make_tuple(lr, hu, sl, bs));
				}
			}
		}
	}

	return allHyperParamCombinations;
}