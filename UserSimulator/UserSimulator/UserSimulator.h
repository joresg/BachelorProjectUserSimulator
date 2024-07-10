#include "MathHandler.cuh"

#include <vector>

#pragma once
class UserSimulator
{
public:
	UserSimulator(int inputNeurons, int hiddenLayerNeurons, int outputNeurons, double learningRate, int batchSize);
	int PredictNextClickFromSequence(std::vector<CUDAMatrix> onehotEncodedLabels, bool performBackProp, bool verboseMode, bool trainMode);
	void ForwardProp(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode, bool trainMode);
	void BackProp(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode);
	MathHandler* GetMathEngine() { return _mathHandler; }
	void PrintPredictedClassProbabilities();
	int GetBatchSize() { return _batchSize; }
	CUDAMatrix GetBiasesOutput() { return _biasesOutput; }
	void SetAllTrainingExamplesCount(int allTrainingExamples) { _allTrainingExamplesCount = allTrainingExamples; }
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
		/*printf("_hiddenStepValues\n");
		for (int i = 0; i < _hiddenStepValues.size(); i++) {
			_hiddenStepValues[i].Print();
		}
		printf("_outputValues\n");
		for (int i = 0; i < _outputValues.size(); i++) {
			_outputValues[i].Print();
		}*/
	}

private:
	MathHandler* _mathHandler;

	CUDAMatrix _inputWeights;
	CUDAMatrix _hiddenWeights;
	CUDAMatrix _weightsOutput;
	CUDAMatrix _biasesHidden;
	CUDAMatrix _batchBiasesHidden;
	CUDAMatrix _biasesOutput;
	CUDAMatrix _batchBiasesOutput;

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
};