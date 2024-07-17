#include "MathHandler.cuh"

#include <vector>

#pragma once
class UserSimulator
{
public:
	UserSimulator(int inputNeurons, int hiddenLayerNeurons, int outputNeurons, double learningRate, int batchSize);
	double EvaluateOnValidateSet();
	std::deque<std::tuple<int, double>> PredictNextClickFromSequence(std::vector<CUDAMatrix> onehotEncodedLabels, bool performBackProp, bool verboseMode, bool trainMode, int selectNTopClasses = 1);
	void ForwardProp(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode, bool trainMode);
	void BackProp(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode);
	MathHandler* GetMathEngine() { return _mathHandler; }
	void PrintPredictedClassProbabilities();
	int GetBatchSize() { return _batchSize; }
	CUDAMatrix GetBiasesOutput() { return _biasesOutput; }
	void SetAllTrainingExamplesCount(int allTrainingExamples) { _allTrainingExamplesCount = allTrainingExamples; }
	std::vector<std::vector<int>> GetValidationSet() { return _validationSet; }
	void SetValidationSet(std::vector<std::vector<int>> validationSet) { _validationSet = validationSet; }
	void PrintAllParameters() {
		printf("_inputWeights\n");
		_inputWeights.Print();
		printf("_hiddenWeights\n");
		_hiddenWeights.Print();
		printf("_weightsOutput\n");
		_weightsOutput.Print();
		printf("_biasesHidden\n");
		_biasesHidden.Print();
		printf("_batchBiasesHidden\n");
		_batchBiasesHidden.Print();
		printf("_biasesOutput\n");
		_biasesOutput.Print();
		printf("_batchBiasesOutput\n");
		_batchBiasesOutput.Print();
	}

	void CopyParameters();
	void RestoreBestParameters();

private:
	MathHandler* _mathHandler;

	CUDAMatrix _inputWeights;
	CUDAMatrix _hiddenWeights;
	CUDAMatrix _weightsOutput;
	CUDAMatrix _biasesHidden;
	CUDAMatrix _batchBiasesHidden;
	CUDAMatrix _biasesOutput;
	CUDAMatrix _batchBiasesOutput;

	CUDAMatrix _inputWeightsCopy;
	CUDAMatrix _hiddenWeightsCopy;
	CUDAMatrix _weightsOutputCopy;
	CUDAMatrix _biasesHiddenCopy;
	CUDAMatrix _biasesOutputCopy;

	std::vector<CUDAMatrix> _hiddenStepValues;
	std::vector<CUDAMatrix> _outputValues;
	std::vector<CUDAMatrix> _oneHotEncodedClicks;

	double _totalLoss;

	//std::vector<CUDAMatrix> _oneHotEncodedClicks;

	int _inputNeurons;
	int _hiddenLayerNeurons;
	int _outputNeurons;
	double _learningRate;
	int _batchSize;
	int _allTrainingExamplesCount;
	int _allClasses;

	std::vector<std::vector<int>> _validationSet;
};