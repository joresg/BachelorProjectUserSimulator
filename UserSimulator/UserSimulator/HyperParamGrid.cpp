#include "HyperParamGrid.h"
#include <tuple>


HyperParamGrid::HyperParamGrid(int allClasses) {
	//_learningRate.push_back(0.01);
	//_learningRate.push_back(0.001);
	//_learningRate.push_back(0.0001);
	_learningRate.push_back(0.00001);
	_learningRate.push_back(0.000001);
	_learningRate.push_back(0.0000001);
	_learningRate.push_back(0.00000001);

	std::vector<int> hiddenNeuronsCombination1;
	std::vector<int> hiddenNeuronsCombination2;
	std::vector<int> hiddenNeuronsCombination3;
	std::vector<int> hiddenNeuronsCombination4;
	std::vector<int> hiddenNeuronsCombination5;
	std::vector<int> hiddenNeuronsCombination6;
	std::vector<int> hiddenNeuronsCombination7;
	std::vector<int> hiddenNeuronsCombination8;
	std::vector<int> hiddenNeuronsCombination9;
	std::vector<int> hiddenNeuronsCombination10;
	std::vector<int> hiddenNeuronsCombination11;
	std::vector<int> hiddenNeuronsCombination12;
	std::vector<int> hiddenNeuronsCombination13;
	std::vector<int> hiddenNeuronsCombination14;
	std::vector<int> hiddenNeuronsCombination15;
	std::vector<int> hiddenNeuronsCombination16;

	std::vector<int> hiddenNeuronsCombination17;
	std::vector<int> hiddenNeuronsCombination18;
	std::vector<int> hiddenNeuronsCombination19;
	std::vector<int> hiddenNeuronsCombination20;

	std::vector<int> hiddenNeuronsCombination21;
	std::vector<int> hiddenNeuronsCombination22;

	int hiddenNuronsCount1 = allClasses / 2;
	int hiddenNuronsCount2 = allClasses;
	int hiddenNuronsCount3 = allClasses + (allClasses / 2);

	hiddenNeuronsCombination21.push_back(hiddenNuronsCount2);

	hiddenNeuronsCombination22.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination22.push_back(hiddenNuronsCount1);

	hiddenNeuronsCombination17.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination17.push_back(hiddenNuronsCount2);

	hiddenNeuronsCombination18.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination18.push_back(hiddenNuronsCount3);

	hiddenNeuronsCombination19.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination19.push_back(hiddenNuronsCount2);

	hiddenNeuronsCombination20.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination20.push_back(hiddenNuronsCount3);

	hiddenNeuronsCombination1.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination1.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination1.push_back(hiddenNuronsCount1);

	hiddenNeuronsCombination2.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination2.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination2.push_back(hiddenNuronsCount2);

	hiddenNeuronsCombination3.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination3.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination3.push_back(hiddenNuronsCount1);

	hiddenNeuronsCombination4.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination4.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination4.push_back(hiddenNuronsCount3);

	hiddenNeuronsCombination5.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination5.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination5.push_back(hiddenNuronsCount1);

	hiddenNeuronsCombination6.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination6.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination6.push_back(hiddenNuronsCount3);

	hiddenNeuronsCombination7.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination7.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination7.push_back(hiddenNuronsCount2);

	hiddenNeuronsCombination8.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination8.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination8.push_back(hiddenNuronsCount1);

	hiddenNeuronsCombination9.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination9.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination9.push_back(hiddenNuronsCount2);

	hiddenNeuronsCombination10.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination10.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination10.push_back(hiddenNuronsCount1);

	hiddenNeuronsCombination11.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination11.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination11.push_back(hiddenNuronsCount3);

	hiddenNeuronsCombination12.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination12.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination12.push_back(hiddenNuronsCount1);

	hiddenNeuronsCombination13.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination13.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination13.push_back(hiddenNuronsCount3);

	hiddenNeuronsCombination14.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination14.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination14.push_back(hiddenNuronsCount1);

	hiddenNeuronsCombination15.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination15.push_back(hiddenNuronsCount1);
	hiddenNeuronsCombination15.push_back(hiddenNuronsCount2);

	hiddenNeuronsCombination16.push_back(hiddenNuronsCount3);
	hiddenNeuronsCombination16.push_back(hiddenNuronsCount2);
	hiddenNeuronsCombination16.push_back(hiddenNuronsCount1);

	_hiddenUnits.push_back(hiddenNeuronsCombination21);
	_hiddenUnits.push_back(hiddenNeuronsCombination22);

	_hiddenUnits.push_back(hiddenNeuronsCombination17);
	_hiddenUnits.push_back(hiddenNeuronsCombination18);
	_hiddenUnits.push_back(hiddenNeuronsCombination19);
	_hiddenUnits.push_back(hiddenNeuronsCombination20);

	_hiddenUnits.push_back(hiddenNeuronsCombination1);
	_hiddenUnits.push_back(hiddenNeuronsCombination2);
	_hiddenUnits.push_back(hiddenNeuronsCombination3);
	_hiddenUnits.push_back(hiddenNeuronsCombination4);
	_hiddenUnits.push_back(hiddenNeuronsCombination5);
	_hiddenUnits.push_back(hiddenNeuronsCombination6);
	_hiddenUnits.push_back(hiddenNeuronsCombination7);
	_hiddenUnits.push_back(hiddenNeuronsCombination8);
	_hiddenUnits.push_back(hiddenNeuronsCombination9);
	_hiddenUnits.push_back(hiddenNeuronsCombination10);
	_hiddenUnits.push_back(hiddenNeuronsCombination11);
	_hiddenUnits.push_back(hiddenNeuronsCombination12);
	_hiddenUnits.push_back(hiddenNeuronsCombination13);
	_hiddenUnits.push_back(hiddenNeuronsCombination14);
	_hiddenUnits.push_back(hiddenNeuronsCombination15);
	_hiddenUnits.push_back(hiddenNeuronsCombination16);

	_sequenceLength.push_back(5);
	_sequenceLength.push_back(10);
	//_sequenceLength.push_back(15);
	_sequenceLength.push_back(20);
	//_sequenceLength.push_back(25);
	_sequenceLength.push_back(30);

	_batchSize.push_back(8);
	_batchSize.push_back(16);
	_batchSize.push_back(32);
	_batchSize.push_back(64);
}

std::vector<std::tuple<double, std::vector<int>, int, int>> HyperParamGrid::HyperParameterGridSearch() {
	std::vector<std::tuple<double, std::vector<int>, int, int>> allHyperParamCombinations;

	/*for (const auto& lr : _learningRate) {
		for (const auto& hu : _hiddenUnits) {
			for (const auto& sl : _sequenceLength) {
				for (const auto& bs : _batchSize) {
					allHyperParamCombinations.push_back(std::make_tuple(lr, hu, sl, bs));
				}
			}
		}
	}*/

	for (const auto& bs : _batchSize) {
		for (const auto& sl : _sequenceLength) {
			for (const auto& hu : _hiddenUnits) {
				for (const auto& lr : _learningRate) {
					allHyperParamCombinations.push_back(std::make_tuple(lr, hu, sl, bs));
				}
			}
		}
	}
	return allHyperParamCombinations;
}

void HyperParamGrid::HyperParameterGridSearchParametersPrint() {
}