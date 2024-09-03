#include "MathHandler.cuh"
#include "crow.h"
#include "serialize_tuple.h"
#include "nlohmann/json.hpp"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <tuple>

#pragma once
class UserSimulator
{
public:
	UserSimulator(int inputNeurons, std::vector<std::tuple<int, LayerActivationFuncs>> hiddenLayerNeurons, int outputNeurons, double learningRate, int batchSize, int trainingSeqLength);
	UserSimulator();
	double EvaluateOnValidateSet(int lossType);
	std::deque<std::tuple<int, double>> PredictNextClickFromSequence(std::vector<CUDAMatrix> onehotEncodedLabels,
		bool performBackProp, bool verboseMode, bool trainMode, bool validationMode, int selectNTopClasses = 1);
	std::vector<std::vector<int>> PredictAllSequencesFromSequence(std::vector<int> startingSequence, int seqLen);
	void ForwardProp(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode, bool trainMode, bool validationMode, 
		bool forwardDirection, int fullSeqLength, CUDAMatrix* nextAction = nullptr);
	void BackProp(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode);
	void ForwardPropGated(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode, bool trainMode);
	void BackPropGated(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode);
	MathHandler* GetMathEngine() { return _mathHandler; }
	void PrintPredictedClassProbabilities();
	int GetBatchSize() { return _batchSize; }
	CUDAMatrix GetBiasesOutput() { return _biasesOutput; }
	void SetAllTrainingExamplesCount(int allTrainingExamples) { _allTrainingExamplesCount = allTrainingExamples; }
	std::vector<std::vector<int>> GetValidationSet() { return _validationSet; }
	void SetValidationSet(std::vector<std::vector<int>> validationSet) { _validationSet = validationSet; }
	void PrintAllParameters() {
		printf("_inputWeights\n");
		//_inputWeights.Print();
		printf("_hiddenWeights\n");
		//_hiddenWeights.Print();
		printf("_weightsOutput\n");
		_weightsOutput.Print();
		printf("_biasesHidden\n");
		//_biasesHidden.Print();
		printf("_biasesOutput\n");
		_biasesOutput.Print();
	}
	void CopyParameters();
	void RestoreBestParameters();
	void SetLearningRate(double lr) { _learningRate = lr; }
	double GetLearningRate() { return _learningRate; }
	void SetModelAccOnValidationData(double accuracy) { _modelAccuracy = accuracy; }
	double GetModelACcOnValidationData() { return _modelAccuracy; }
	void SetTrainingSequenceLength(int seqLength) { _trainingSeqLength = seqLength; }
	int GetTrainingSequenceLength() { return _trainingSeqLength; }
	void SetBatchSize(int newBatchSize) { _batchSize = newBatchSize; }
	void SetGatedUnits(GatedUnits gu) { _gatedUnits = gu; }
	GatedUnits GetGatedUnits() { return _gatedUnits; }
	void SetCmdIDsMap(std::map<std::string, std::tuple<int, int>> commandIDsMap) { _commandIDsMap = commandIDsMap; }
	std::map<std::string, std::tuple<int, int>> GetCmdIDsMap() { return _commandIDsMap; }
	int GetCommandIDFromName(std::string cmdName) {
		return std::get<0>(_commandIDsMap[cmdName]);
	}
	std::string GetCommandNameFromID(int cmdID) {
		for (const auto& keyval : _commandIDsMap)
		{
			if (std::get<0>(keyval.second) == cmdID)
			{
				return keyval.first;
			}
		}
	}
	int GetAllClasses() { return _allClasses; }
	void SetAllClasses(int allClasses) { _allClasses = allClasses; }
	double GetGradientClippingThreshold() { return _gradientClippingThresholdMax; }
	void SetGradientClippingThreshold(double gradientClippingThreshold) { _gradientClippingThresholdMax = gradientClippingThreshold; }
	int GetTotalNumberOfSamples() { return _totalNumberOfSamples; }
	void SetTotalNumberOfSamples(int numOfSamples) { _totalNumberOfSamples = numOfSamples; }
	CUDAMatrix GetWeightsForClasses() { return _weightsForClasses; }
	void SetWeightsForClasses(CUDAMatrix classWeights) { _weightsForClasses = classWeights; }

private:
	friend class boost::serialization::access;

	MathHandler* _mathHandler;
	GatedUnits _gatedUnits;
	std::vector<CUDAMatrix> _inputWeights;
	std::vector<CUDAMatrix> _inputWeightsBack;
	std::vector<CUDAMatrix> _hiddenWeights;
	std::vector<CUDAMatrix> _hiddenWeightsBack;
	CUDAMatrix _weightsOutput;
	CUDAMatrix _weightsOutputBack;
	std::vector<CUDAMatrix> _biasesHidden;
	std::vector<CUDAMatrix> _biasesHiddenBack;
	std::vector<CUDAMatrix> _biasesRecurrentHidden;
	std::vector<CUDAMatrix> _biasesRecurrentHiddenBack;
	CUDAMatrix _biasesOutput;
	CUDAMatrix _biasesOutputBack;
	CUDAMatrix _weightsForClasses;

	std::vector<CUDAMatrix> _velocityWeightsInput;
	std::vector<CUDAMatrix> _velocityWeightsHidden;
	std::vector<CUDAMatrix> _velocityBias;
	std::vector<CUDAMatrix> _velocityRecurrentHiddenBias;

	std::vector<CUDAMatrix> _velocityWeightsInputBack;
	std::vector<CUDAMatrix> _velocityWeightsHiddenBack;
	std::vector<CUDAMatrix> _velocityBiasBack;
	std::vector<CUDAMatrix> _velocityRecurrentHiddenBiasBack;

	std::vector<CUDAMatrix> _updateGateInput;
	std::vector<CUDAMatrix> _updateGateHidden;
	std::vector<CUDAMatrix> _updateGateBias;
	std::vector<CUDAMatrix> _resetGateInput;
	std::vector<CUDAMatrix> _resetGateHidden;
	std::vector<CUDAMatrix> _resetGateBias;
	std::vector<CUDAMatrix> _candidateActivationInput;
	std::vector<CUDAMatrix> _candidateActivationHidden;
	std::vector<CUDAMatrix> _candidateActivationBias;

	std::vector<CUDAMatrix> _inputWeightsCopy;
	std::vector<CUDAMatrix> _inputWeightsCopyBack;
	std::vector<CUDAMatrix> _hiddenWeightsCopy;
	std::vector<CUDAMatrix> _hiddenWeightsCopyBack;
	CUDAMatrix _weightsOutputCopy;
	std::vector<CUDAMatrix> _biasesHiddenCopy;
	std::vector<CUDAMatrix> _biasesHiddenCopyBack;
	std::vector<CUDAMatrix> _biasesRecurrentHiddenCopy;
	std::vector<CUDAMatrix> _biasesRecurrentHiddenCopyBack;
	CUDAMatrix _biasesOutputCopy;

	std::vector<std::vector<CUDAMatrix>> _hiddenStepValues;
	std::vector<std::vector<CUDAMatrix>> _hiddenStepValuesBack;
	std::vector<CUDAMatrix> _hiddenStepValuesCombined;
	std::vector<std::vector<CUDAMatrix>> _resetGateValues;
	std::vector<std::vector<CUDAMatrix>> _updateGateValues;
	std::vector<std::vector<CUDAMatrix>> _candidateActivationValues;
	std::vector<CUDAMatrix> _outputValuesCombined;
	std::vector<CUDAMatrix> _outputValues;
	std::vector<CUDAMatrix> _outputValuesBack;
	std::vector<CUDAMatrix> _oneHotEncodedClicks;
	std::vector<CUDAMatrix> _oneHotEncodedClicksReversed;

	std::vector<CUDAMatrix> _recurrentMasks;

	double _totalLoss;
	double _totalLossGeneral;

	int _inputNeurons;
	std::vector<int> _hiddenLayerNeurons;
	std::vector<LayerActivationFuncs> _hiddenLayerNeuronsActivationFuncs;
	int _outputNeurons;
	double _learningRate;
	int _batchSize;
	int _allTrainingExamplesCount;
	int _allClasses;
	double _modelAccuracy;
	int _trainingSeqLength;
	double _momentumCoefficient;
	double _gradientClippingThresholdMin;
	double _gradientClippingThresholdMax;

	std::vector<std::vector<int>> _validationSet;
	std::map<std::string, std::tuple<int, int>> _commandIDsMap;
	int _totalNumberOfSamples;

	double _dropoutRate;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& _gatedUnits;
		ar& _inputWeights;
		ar& _inputWeightsBack;
		ar& _hiddenWeights;
		ar& _hiddenWeightsBack;
		ar& _weightsOutput;
		ar& _biasesHidden;
		ar& _biasesHiddenBack;
		ar& _biasesRecurrentHidden;
		ar& _biasesRecurrentHiddenBack;
		ar& _biasesOutput;
		ar& _velocityWeightsInput;
		ar& _velocityWeightsHidden;
		ar& _velocityBias;
		ar& _velocityRecurrentHiddenBias;
		ar& _inputNeurons;
		ar& _hiddenLayerNeurons;
		ar& _hiddenLayerNeuronsActivationFuncs;
		ar& _outputNeurons;
		ar& _learningRate;
		ar& _batchSize;
		ar& _allTrainingExamplesCount;
		ar& _allClasses;
		ar& _modelAccuracy;
		ar& _trainingSeqLength;
		ar& _momentumCoefficient;
		ar& _commandIDsMap;
		ar& _totalNumberOfSamples;
		ar& _dropoutRate;
	}
};