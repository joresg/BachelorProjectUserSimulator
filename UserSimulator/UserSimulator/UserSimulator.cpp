#include "UserSimulator.h"

#include <stdio.h>

int main() {
	bool verboseMode = false;
	//int allClasses = commandIDsMap.size();
	int allClasses = 5;
	double learningRate = 0.0001;
	int inputNeurons = allClasses;
	int outputNeurons = allClasses;
	int hiddenNeurons = allClasses;

	UserSimulator* userSimulator = new UserSimulator(inputNeurons, hiddenNeurons, outputNeurons, learningRate);

	double rand1 = userSimulator->GetMathEngine()->GenerateRandomNumber(0, 1);
	double rand2 = userSimulator->GetMathEngine()->GenerateRandomNumber(0, 1);
	double rand3 = userSimulator->GetMathEngine()->GenerateRandomNumber(0, 1);
	double rand4 = userSimulator->GetMathEngine()->GenerateRandomNumber(0, 1);

	CUDAMatrix mat1(2, 2);
	CUDAMatrix mat2(2, 2);
	CUDAMatrix mat3(5, 2);
	CUDAMatrix mat4(5, 1);

	mat1(0, 0) = 0.1;
	mat1(0, 1) = 0.2;
	mat1(1, 0) = 0.3;
	mat1(1, 1) = 0.4;

	/*mat1(0, 0) = 0.1;
	mat1(0, 1) = 0.2;
	mat1(1, 0) = 0.3;
	mat1(1, 1) = 0.4;*/

	mat2(0, 0) = 0.3;
	mat2(0, 1) = 0.6;
	mat2(1, 0) = 0.9;
	mat2(1, 1) = 0.12;

	mat3(0, 0) = 0;
	mat3(0, 1) = 1;
	mat3(1, 0) = 2;
	mat3(1, 1) = 3;
	mat3(2, 0) = 4;
	mat3(2, 1) = 5;
	mat3(3, 0) = 6;
	mat3(3, 1) = 7;
	mat3(4, 0) = 8;
	mat3(4, 1) = 9;

	mat4(0, 0) = 1;
	mat4(1, 0) = 4;
	mat4(2, 0) = 2;
	mat4(3, 0) = 10;
	mat4(4, 0) = 42;

	printf("printing matrices\n");
	printf("\n");
	mat1.Print();
	mat2.Print();
	mat3.Print();
	mat4.Print();

	printf("RES SUBSTRACT WITH OVERLOAD\n");
	CUDAMatrix matSub = mat1 - mat2;
	matSub.Print();
	printf("RES ADD WITH OVERLOAD\n");
	CUDAMatrix matADd = mat1 + mat2;
	matADd.Print();
	printf("MAT TRANSPOSE RESULT\n");
	CUDAMatrix matTranspose = userSimulator->GetMathEngine()->TransposeMatrix(mat3);
	matTranspose.Print();
	CUDAMatrix matTranspose1 = userSimulator->GetMathEngine()->TransposeMatrix(mat4);
	matTranspose1.Print();

	printf("MULTIPLY\n");
	CUDAMatrix matMultiply = mat1 * mat2;
	matMultiply.Print();

	printf("CONST\n");
	CUDAMatrix matMultiplyConst = mat1 * 10;
	matMultiplyConst.Print();

	printf("MULTIPLY ELEMENT WISE\n");
	CUDAMatrix matMultiplyElementWise = mat1.Array() * mat2.Array();
	matMultiplyElementWise.Print();

	mat1 = mat2;

	printf("assigned\n");
	mat1.Print();
	mat2.Print();

	(mat1 * mat2).Print();

	CUDAMatrix tanhDer = userSimulator->GetMathEngine()->TanhDerivative(mat1);
	printf("tanhder\n");
	tanhDer.Print();

	mat1.Print();

	std::vector<CUDAMatrix> testInputSequence;
	CUDAMatrix input1(5, 1);
	input1(0, 0) = 0;
	input1(1, 0) = 0;
	input1(2, 0) = 1;
	input1(3, 0) = 0;
	input1(4, 0) = 0;
	CUDAMatrix input2(5, 1);
	input2(0, 0) = 1;
	input2(1, 0) = 0;
	input2(2, 0) = 0;
	input2(3, 0) = 0;
	input2(4, 0) = 0;
	CUDAMatrix input3(5, 1);
	input3(0, 0) = 0;
	input3(1, 0) = 0;
	input3(2, 0) = 0;
	input3(3, 0) = 0;
	input3(4, 0) = 1;
	testInputSequence.push_back(input1);
	testInputSequence.push_back(input2);
	testInputSequence.push_back(input3);

	for (int i = 0; i < 500; i++) {
		printf("ITERATION: %i\n", i);
		int predictedNextClickClassID = userSimulator->PredictNextClickFromSequence(testInputSequence, true, verboseMode);
		printf("predicted class: %i\n", predictedNextClickClassID);
	}

	std::vector<CUDAMatrix> testInputSequence1;
	testInputSequence1.push_back(input1);
	int predictedNextClickClassIDTest = userSimulator->PredictNextClickFromSequence(testInputSequence1, false, verboseMode);
	printf("predicted: %i\n", predictedNextClickClassIDTest);

	return 0;
}

UserSimulator::UserSimulator(int inputNeurons, int hiddenLayerNeurons, int outputNeurons, double learningRate) {

	_learningRate = learningRate;
	_inputNeurons = inputNeurons;
	_hiddenLayerNeurons = hiddenLayerNeurons;
	_outputNeurons = outputNeurons;
	_totalLoss = 0;

	unsigned long int randSeed = 18273;
	_mathHandler = new MathHandler(randSeed);


	double lower_bound = 0;
	double upper_bound = 1;
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	std::default_random_engine re = _mathHandler->GetRandomEngine();

	// init input weigths
	_inputWeights.Resize(hiddenLayerNeurons, inputNeurons);
	for (int i = 0; i < _inputWeights.GetRows(); i++) {
		for (int j = 0; j < _inputWeights.GetColumns(); j++) {
			_inputWeights(i, j) = unif(re) / 5 - 0.1;
		}
	}

	_inputWeights.Print();

	// init hidden/recurrent weights
	_hiddenWeights.Resize(hiddenLayerNeurons, hiddenLayerNeurons);
	for (int i = 0; i < _hiddenWeights.GetRows(); i++) {
		for (int j = 0; j < _hiddenWeights.GetColumns(); j++) {
			_hiddenWeights(i, j) = unif(re) / 5 - 0.1;
		}
	}

	_hiddenWeights.Print();

	// init output weights

	_weightsOutput.Resize(outputNeurons, hiddenLayerNeurons);
	for (int i = 0; i < _weightsOutput.GetRows(); i++) {
		for (int j = 0; j < _weightsOutput.GetColumns(); j++) {
			//weightsOutput(i, j) = unif(re) / 5 - 0.1;
			// initialize big output weights

			_weightsOutput(i, j) = unif(re) * 50;
			//_weightsOutput(i, j) = unif(re) / 5 - 0.1;
		}
	}

	_weightsOutput.Print();

	// biases for hidden layer
	_biasesHidden.Resize(hiddenLayerNeurons, 1);
	for (int i = 0; i < hiddenLayerNeurons; i++) {
		_biasesHidden(i, 0) = unif(re) / 5 - 0.1;
	}

	_biasesHidden.Print();

	// biases for output layer
	_biasesOutput.Resize(outputNeurons, 1);
	for (int i = 0; i < outputNeurons; i++) {
		//biasesOutput(i, 0) = unif(re) / 5 - 0.1;
		_biasesOutput(i, 0) = unif(re);
	}

	_biasesOutput.Print();
}

void UserSimulator::ForwardProp(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode) {
	
	CUDAMatrix XI = _mathHandler->TransposeMatrix(_mathHandler->TransposeMatrix(onehotEncodedInput) * _inputWeights);
	XI.Print();
	CUDAMatrix XHCurrentTimeStep;
	// if first element in sequence XHidden at previous time step non existent, just take XI(nput)
	if (sequencePosition == 0) {
		XHCurrentTimeStep = XI;
	}
	else {
		XHCurrentTimeStep = XI + (_hiddenWeights * _hiddenStepValues[sequencePosition - 1]) + _biasesHidden;
	}

	XHCurrentTimeStep.Print();

	CUDAMatrix activatedHiddenLayer = XHCurrentTimeStep;

	activatedHiddenLayer.tanh();
	_hiddenStepValues.push_back(activatedHiddenLayer);

	activatedHiddenLayer.Print();

	// compute output

	CUDAMatrix outputValuesUnactivated = _weightsOutput * activatedHiddenLayer + _biasesOutput;

	double sumForSoftmax = 0;
	for (int i = 0; i < outputValuesUnactivated.GetRows(); i++) {
		sumForSoftmax += std::exp(outputValuesUnactivated(i, 0));
	}

	CUDAMatrix outputValuesActivated(outputValuesUnactivated.GetRows(), 1);

	for (int i = 0; i < outputValuesActivated.GetRows(); i++) {
		outputValuesActivated(i, 0) = std::exp(outputValuesUnactivated(i, 0)) / sumForSoftmax;
	}

	_outputValues.push_back(outputValuesActivated);

	outputValuesActivated.Print();

	if (verboseMode)
	{
		std::cout << "outputValuesActivated" << std::endl;
		outputValuesActivated.Print();
	}

	// test print predicted sequence
	double classID = -1;

	double maxProbability = 0;// outputValuesActivated.maxCoeff();
	for (int j = 0; j < outputValuesActivated.GetRows(); j++) {
		if (outputValuesActivated(j, 0) > maxProbability) {
			maxProbability = outputValuesActivated(j, 0);
			classID = j;
		}
	}

	// calculate loss for debugging purposes

	double crossEntropyLoss = 0;
	/*for (int j = 0; j < outputValuesActivated.rows(); j++) {

	}*/

	_totalLoss -= std::log(maxProbability);

	// free allocated resources
	//XI.Destroy();
	////XHCurrentTimeStep.Destroy();
	//activatedHiddenLayer.Destroy();
	//outputValuesUnactivated.Destroy();
	//outputValuesActivated.Destroy();
}

void UserSimulator::BackProp(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode) {
	if (verboseMode) std::cout << "Back prop" << std::endl;

	// Initialize gradients
	CUDAMatrix outputWeightsGrad = CUDAMatrix::Zero(_weightsOutput.GetRows(), _weightsOutput.GetColumns());
	CUDAMatrix outputBiasGrad = CUDAMatrix::Zero(_outputNeurons, 1);
	CUDAMatrix hiddenWeightsGrad = CUDAMatrix::Zero(_hiddenWeights.GetRows(), _hiddenWeights.GetColumns());
	CUDAMatrix hiddenBiasGrad = CUDAMatrix::Zero(_hiddenLayerNeurons, 1);
	CUDAMatrix inputWeightsGrad = CUDAMatrix::Zero(_inputWeights.GetRows(), _inputWeights.GetColumns());

	/*printf("initialized with 0\n");
	outputWeightsGrad.Print();
	outputBiasGrad.Print();
	hiddenWeightsGrad.Print();
	hiddenBiasGrad.Print();
	inputWeightsGrad.Print();*/

	CUDAMatrix nextHiddenGrad = CUDAMatrix::Zero(_hiddenLayerNeurons, 1);

	// Iterate over each timestep from last to first
	for (int i = _outputValues.size() - 1; i >= 0; i--) {
		// Cross-entropy loss gradient w.r.t. softmax input
		CUDAMatrix lossGrad = _outputValues[i] - oneHotEncodedLabels[i + 1]; // Gradient of softmax + cross-entropy

		// Gradients for the output layer
		outputWeightsGrad = outputWeightsGrad + (_hiddenStepValues[i] * _mathHandler->TransposeMatrix(lossGrad));
		outputBiasGrad = outputBiasGrad + lossGrad;

		// Backpropagate into the hidden layer
		CUDAMatrix outputGrad = _mathHandler->TransposeMatrix(_weightsOutput) * lossGrad;
		CUDAMatrix hiddenGrad = outputGrad + (_mathHandler->TransposeMatrix(_hiddenWeights) * nextHiddenGrad);
		CUDAMatrix tanhDerivative = _mathHandler->TanhDerivative(_hiddenStepValues[i]);
		hiddenGrad = hiddenGrad.Array() * tanhDerivative.Array();

		// Accumulate gradients for hidden layer weights and biases
		if (i > 0) {
			hiddenWeightsGrad = hiddenWeightsGrad + (_hiddenStepValues[i - 1] * _mathHandler->TransposeMatrix(hiddenGrad));
			hiddenBiasGrad = hiddenBiasGrad + hiddenGrad;
		}

		// Accumulate gradients for input weights
		inputWeightsGrad = inputWeightsGrad + (_oneHotEncodedClicks[i] * _mathHandler->TransposeMatrix(hiddenGrad));

		// Update nextHiddenGrad for the next iteration
		nextHiddenGrad = hiddenGrad;

		lossGrad.Destroy();
		outputGrad.Destroy();
		hiddenGrad.Destroy();
		tanhDerivative.Destroy();
	}

	// Apply learning rate adjustment
	double adjustedLearningRate = learningRate / _outputValues.size();

	// Update weights and biases
	_inputWeights = _inputWeights - (inputWeightsGrad * adjustedLearningRate);
	_hiddenWeights = _hiddenWeights - (hiddenWeightsGrad * adjustedLearningRate);
	_biasesHidden = _biasesHidden - (hiddenBiasGrad * adjustedLearningRate);
	_weightsOutput = _weightsOutput - (outputWeightsGrad * adjustedLearningRate);
	_biasesOutput = _biasesOutput - (outputBiasGrad * adjustedLearningRate);

	// free allocated resources
	//outputWeightsGrad.Destroy();
	//outputBiasGrad.Destroy();
	//hiddenWeightsGrad.Destroy();
	//hiddenBiasGrad.Destroy();
	//inputWeightsGrad.Destroy();
	////nextHiddenGrad.Destroy();
}

int UserSimulator::PredictNextClickFromSequence(std::vector<CUDAMatrix> onehotEncodedLabels, bool performBackProp, bool verboseMode) {

	if (onehotEncodedLabels.size() == 0) return -1;

	_hiddenStepValues.clear();
	_outputValues.clear();
	_oneHotEncodedClicks.clear();
	_oneHotEncodedClicks = onehotEncodedLabels;

	_totalLoss = 0;

	int allClasses = _biasesOutput.GetRows();

	for (int i = 0; i < onehotEncodedLabels.size() - (performBackProp ? 1 : 0); i++) {

		ForwardProp(onehotEncodedLabels[i], i, verboseMode);
	}

	printf("TOTAL LOSS: %f\n", _totalLoss);

	int firstClickID = -1;
	int classID = -1;

	for (int j = 0; j < onehotEncodedLabels[0].GetRows(); j++) {
		if (onehotEncodedLabels[0](j, 0) == 1.0) {
			firstClickID = j;
			break;
		}
	}
	if (performBackProp && verboseMode) std::cout << "STARTED WITH: " << firstClickID << std::endl;
	// test print predicted sequence
	for (int i = 0; i < _outputValues.size(); i++) {
		double maxProbability = 0;//_outputValues[i].maxCoeff();
		for (int j = 0; j < _outputValues[i].GetRows(); j++) {
			if (_outputValues[i](j, 0) > maxProbability) {
				maxProbability = _outputValues[i](j, 0);
				classID = j;
			}
		}
		//std::cout << outputValues[i].maxCoeff() << std::endl;
		if (performBackProp && verboseMode) std::cout << classID << std::endl;
	}

	//if (performBackProp && verboseMode) std::cout << "TOTAL LOSS: " << _totalLoss << std::endl;

	// after going through whole sequence perform backprop

	//BackProp(onehotEncodedLabels[onehotEncodedLabels.size() - 1], _learningRate);
	if (performBackProp) BackProp(onehotEncodedLabels, _learningRate, verboseMode);

	return classID;
}