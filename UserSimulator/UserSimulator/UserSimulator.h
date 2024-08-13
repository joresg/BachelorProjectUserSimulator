#include "MathHandler.cuh"

#pragma once
class UserSimulator
{
public:
	UserSimulator(int inputNeurons, std::vector<std::tuple<int, LayerActivationFuncs>> hiddenLayerNeurons, int outputNeurons, double learningRate, int batchSize, int trainingSeqLength);
	double EvaluateOnValidateSet();
	std::deque<std::tuple<int, double>> PredictNextClickFromSequence(std::vector<CUDAMatrix> onehotEncodedLabels, bool performBackProp, bool verboseMode, bool trainMode, int selectNTopClasses = 1);
	void ForwardProp(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode, bool trainMode);
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
	void SetModelAccOnValidationData(double accuracy) { _modelAccuracy = accuracy; }
	double GetModelACcOnValidationData() { return _modelAccuracy; }
	void SetTrainingSequenceLength(int seqLength) { _trainingSeqLength = seqLength; }
	int GetTrainingSequenceLength() { return _trainingSeqLength; }
	void SetBatchSize(int newBatchSize) { _batchSize = newBatchSize; }

private:
	MathHandler* _mathHandler;

	std::vector<CUDAMatrix> _inputWeights;
	std::vector<CUDAMatrix> _hiddenWeights;
	CUDAMatrix _weightsOutput;
	std::vector<CUDAMatrix> _biasesHidden;
	CUDAMatrix _biasesOutput;

	std::vector<CUDAMatrix> _velocityWeightsInput;
	std::vector<CUDAMatrix> _velocityWeightsHidden;
	std::vector<CUDAMatrix> _velocityBias;

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
	std::vector<CUDAMatrix> _hiddenWeightsCopy;
	CUDAMatrix _weightsOutputCopy;
	std::vector<CUDAMatrix> _biasesHiddenCopy;
	CUDAMatrix _biasesOutputCopy;

	std::vector<std::vector<CUDAMatrix>> _hiddenStepValues;
	std::vector<std::vector<CUDAMatrix>> _resetGateValues;
	std::vector<std::vector<CUDAMatrix>> _updateGateValues;
	std::vector<std::vector<CUDAMatrix>> _candidateActivationValues;
	std::vector<CUDAMatrix> _outputValues;
	std::vector<CUDAMatrix> _oneHotEncodedClicks;

	double _totalLoss;

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

	std::vector<std::vector<int>> _validationSet;
};