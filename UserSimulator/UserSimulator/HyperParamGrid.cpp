#include "HyperParamGrid.h"
#include <tuple>


HyperParamGrid::HyperParamGrid(int allClasses, GatedUnits gatedCells) {
	_learningRate.push_back(0.01);
	_learningRate.push_back(0.001);
	_learningRate.push_back(0.0001);
	_learningRate.push_back(0.00001);
	_learningRate.push_back(0.000001);

	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination1;
	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination2;
	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination3;

	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination4;
	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination5;
	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination6;

	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination8;

	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination9;
	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination10;

	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination11;
	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination12;
	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination13;
	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination14;

	std::vector<std::tuple<int, LayerActivationFuncs>> hiddenNeuronsCombination15;


	int hiddenNuronsCount1 = allClasses / 2;
	int hiddenNuronsCount5 = allClasses - (allClasses / 2);
	int hiddenNuronsCount2 = allClasses;
	int hiddenNuronsCount3 = allClasses + (allClasses / 2);
	int hiddenNuronsCount4 = allClasses * 2;

	LayerActivationFuncs reluA = reluAct;
	LayerActivationFuncs tanhA = tanhAct;
	LayerActivationFuncs leakyReLUA = leakyReLU;

	hiddenNeuronsCombination4.push_back(std::make_tuple(hiddenNuronsCount3, leakyReLUA));
	hiddenNeuronsCombination4.push_back(std::make_tuple(hiddenNuronsCount2, leakyReLUA));
	hiddenNeuronsCombination4.push_back(std::make_tuple(hiddenNuronsCount1, tanhA));
	hiddenNeuronsCombination1.push_back(std::make_tuple(hiddenNuronsCount1, tanhA));
	hiddenNeuronsCombination2.push_back(std::make_tuple(hiddenNuronsCount2, tanhA));
	hiddenNeuronsCombination5.push_back(std::make_tuple(hiddenNuronsCount2, leakyReLUA));
	hiddenNeuronsCombination3.push_back(std::make_tuple(hiddenNuronsCount2, tanhA));
	hiddenNeuronsCombination3.push_back(std::make_tuple(hiddenNuronsCount1, tanhA));
	hiddenNeuronsCombination6.push_back(std::make_tuple(hiddenNuronsCount2, leakyReLUA));
	hiddenNeuronsCombination6.push_back(std::make_tuple(hiddenNuronsCount1, leakyReLUA));
	hiddenNeuronsCombination8.push_back(std::make_tuple(hiddenNuronsCount2, leakyReLUA));
	hiddenNeuronsCombination8.push_back(std::make_tuple(hiddenNuronsCount1, tanhA));
	hiddenNeuronsCombination9.push_back(std::make_tuple(hiddenNuronsCount3, leakyReLUA));
	hiddenNeuronsCombination9.push_back(std::make_tuple(hiddenNuronsCount2, tanhA));
	hiddenNeuronsCombination14.push_back(std::make_tuple(hiddenNuronsCount4, leakyReLUA));
	hiddenNeuronsCombination14.push_back(std::make_tuple(hiddenNuronsCount3, tanhA));
	hiddenNeuronsCombination10.push_back(std::make_tuple(hiddenNuronsCount3, tanhA));

	hiddenNeuronsCombination11.push_back(std::make_tuple(hiddenNuronsCount2, leakyReLUA));
	hiddenNeuronsCombination11.push_back(std::make_tuple(hiddenNuronsCount2, tanhA));

	hiddenNeuronsCombination12.push_back(std::make_tuple(hiddenNuronsCount3, leakyReLUA));
	hiddenNeuronsCombination12.push_back(std::make_tuple(hiddenNuronsCount3, tanhA));

	hiddenNeuronsCombination13.push_back(std::make_tuple(hiddenNuronsCount2, tanhA));

	if (gatedCells == NoGates)
	{
		_hiddenUnits.push_back(hiddenNeuronsCombination13);
		_hiddenUnits.push_back(hiddenNeuronsCombination9);
		_hiddenUnits.push_back(hiddenNeuronsCombination4);

		_sequenceLength.push_back(10);
		//_sequenceLength.push_back(15);
		_sequenceLength.push_back(20);
		//_sequenceLength.push_back(25);
		_sequenceLength.push_back(30);
	}
	else {
		_hiddenUnits.push_back(hiddenNeuronsCombination1);
		_hiddenUnits.push_back(hiddenNeuronsCombination2);
		_hiddenUnits.push_back(hiddenNeuronsCombination9);
		_hiddenUnits.push_back(hiddenNeuronsCombination10);


		_sequenceLength.push_back(50);
		_sequenceLength.push_back(100);
	}

	_batchSize.push_back(512);
	_batchSize.push_back(256);
	_batchSize.push_back(128);
	_batchSize.push_back(64);
	_batchSize.push_back(32);
	_batchSize.push_back(16);
}

std::vector<std::tuple<double, std::vector<std::tuple<int, LayerActivationFuncs>>, int, int>> HyperParamGrid::HyperParameterGridSearch() {
	std::vector<std::tuple<double, std::vector<std::tuple<int, LayerActivationFuncs>>, int, int>> allHyperParamCombinations;

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