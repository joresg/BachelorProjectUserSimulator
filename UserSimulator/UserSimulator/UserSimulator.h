#include "MathHandler.cuh"

#include <vector>

#pragma once
class UserSimulator
{
public:
	UserSimulator(int inputNeurons, int hiddenLayerNeurons, int outputNeurons, double learningRate);
	int PredictNextClickFromSequence(std::vector<CUDAMatrix> onehotEncodedLabels, bool performBackProp, bool verboseMode);
	void ForwardProp(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode);
	void BackProp(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode);
	MathHandler* GetMathEngine() { return _mathHandler; }
	void PrintPredictedClassProbabilities();

private:
	MathHandler* _mathHandler;

	CUDAMatrix _inputWeights;
	CUDAMatrix _hiddenWeights;
	CUDAMatrix _weightsOutput;
	CUDAMatrix _biasesHidden;
	CUDAMatrix _biasesOutput;

	std::vector<CUDAMatrix> _hiddenStepValues;
	std::vector<CUDAMatrix> _outputValues;

	double _totalLoss;

	//std::vector<CUDAMatrix> _oneHotEncodedClicks;

	int _inputNeurons;
	int _hiddenLayerNeurons;
	int _outputNeurons;
	double _learningRate;
};