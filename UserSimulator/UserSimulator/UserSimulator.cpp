#include "UserSimulator.h"
#include "FileManager.h"

#include <stdio.h>

static CUDAMatrix CreateOneHotEncodedVecNormalized(int allClases, int classID) {
	CUDAMatrix oneHotEncodedVector(allClases, 1);
	for (int i = 0; i < allClases; i++) {
		//oneHotEncodedVector(i, 0) = i == classID ? 1.0 / (allClases - 1) : 0;
		oneHotEncodedVector(i, 0) = i == classID ? 1.0 : 0;
	}

	return oneHotEncodedVector;
}

std::vector<std::string> split(const std::string& s, char delim) {
	std::vector<std::string> result;
	std::stringstream ss(s);
	std::string item;

	while (getline(ss, item, delim)) {
		result.push_back(item);
	}

	return result;
}

void CUDAMathTest(UserSimulator* userSimulator) {
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
}

int main() {
	bool verboseMode = false;

	cudaDeviceReset();

	FileManager* fileManager = new FileManager();
	std::vector<std::vector<int>> allSequencesFromFile;
	std::tuple<std::vector<std::vector<int>>, std::map<std::string, int>> readFileRes;

	readFileRes = fileManager->ReadTXTFile(R"(C:\Users\joresg\git\BachelorProject\RNN\inputData\unixCommandData)", true);
	allSequencesFromFile = std::get<0>(readFileRes);
	std::map<std::string, int> commandIDsMap = std::get<1>(readFileRes);

	for (auto it = commandIDsMap.cbegin(); it != commandIDsMap.cend(); ++it)
	{
		std::cout << it->first << " - " << it->second << "\n";
	}

	int allClasses = commandIDsMap.size();
	//int allClasses = 5;
	double learningRate = 0.00001;
	int inputNeurons = allClasses;
	int outputNeurons = allClasses;
	int hiddenNeurons = allClasses;

	UserSimulator* userSimulator = new UserSimulator(inputNeurons, hiddenNeurons, outputNeurons, learningRate);

	// train and test

	int epochs = 2;
	for (int i = 0; i < epochs; i++) {
		std::cout << "EPOCH: " << i << std::endl;
		if (verboseMode)
		{
			std::cout << "PREDICTING ";
			std::cout << i << std::endl;
		}

		/*for (int j = 0; j < allOneHotEncodedClicks.size(); j++) {
			if (j > 0 && j % (int)(allOneHotEncodedClicks.size() * 0.1) == 0) {
				std::cout << (j / (int)(allOneHotEncodedClicks.size() * 0.1)) * 10 << "%" << std::endl;
			}
			userSimulator->PredictNextClickFromSequence(allOneHotEncodedClicks[j], true, verboseMode);
		}*/

		for (int k = 0; k < allSequencesFromFile.size(); k++) {
			std::vector<CUDAMatrix> oneHotEncodedInput = userSimulator->GetMathEngine()->CreateOneHotEncodedVector(allSequencesFromFile[k], allClasses);

			if (k > 0 && k % (int)(allSequencesFromFile.size() * 0.1) == 0) {
				std::cout << (k / (int)(allSequencesFromFile.size() * 0.1)) * 10 << "%" << std::endl;
			}
			userSimulator->PredictNextClickFromSequence(oneHotEncodedInput, true, verboseMode);
			//userSimulator->PredictNextClickFromSequence(oneHotEncodedInput, false, verboseMode);

			/*for (auto i : oneHotEncodedInput) {
				i.Destroy();
			}*/
		}

		if (verboseMode) std::cout << std::endl;
	}

	// take user input for first click

	//test it feeding it only the first click.....
	int firstClickClassID = -1;
	bool readFirstClick = true;
	while (readFirstClick)
	{
		std::cout << "input first click ID" << std::endl;
		std::string input;
		std::vector<int> splitInputIntegers;
		int sequenceLength;
		std::getline(std::cin, input);

		if (std::strcmp("q", input.c_str()) == 0) {
			readFirstClick = false;
			break;
		}
		//split if sequence
		if (input.find(",") != std::string::npos) {
			std::vector<std::string> inputSplit = split(input, ',');
			for (const std::string& i : inputSplit) splitInputIntegers.push_back(std::stoi(i));
			firstClickClassID = splitInputIntegers[0];
		}
		else {
			firstClickClassID = std::stoi(input);
			splitInputIntegers.push_back(firstClickClassID);

		}

		std::cout << "input generated sequence length" << std::endl;
		std::getline(std::cin, input);
		sequenceLength = std::stoi(input);


		std::vector<CUDAMatrix> generatedSequence;
		//generatedSequence.push_back(CreateOneHotEncodedVecNormalized(allClasses, firstClickClassID));
		for (const int& i : splitInputIntegers) generatedSequence.push_back(CreateOneHotEncodedVecNormalized(allClasses, i));

		std::vector<int> generatedSequenceClassIDs;
		//generatedSequenceClassIDs.push_back(firstClickClassID);
		for (const int& i : splitInputIntegers) generatedSequenceClassIDs.push_back(i);

		std::cout << "started generating sequence with: " << firstClickClassID << std::endl;

		//predict a sequence of length n
		/*int n = sequence.size();*/

		for (int i = 0; i < sequenceLength - generatedSequenceClassIDs.size(); i++) {
			//clickPredictor->ForwardProp()
			int predictedClassID = userSimulator->PredictNextClickFromSequence(generatedSequence, false, verboseMode);

			// randomness experiment
			//generatedSequence.push_back(CreateOneHotEncodedVecNormalized(allClasses, i % 5 == 0 ? 2 : predictedClassID));
			generatedSequence.push_back(CreateOneHotEncodedVecNormalized(allClasses, predictedClassID));

			// randomness experiment
			//generatedSequenceClassIDs.push_back(i % 5 == 0 ? 2 : predictedClassID);
			generatedSequenceClassIDs.push_back(predictedClassID);

			userSimulator->PrintPredictedClassProbabilities();
			std::cout << "Predicted ClassID: " << predictedClassID << std::endl;

			for (int j = 0; j < generatedSequenceClassIDs.size(); j++) {
				std::cout << generatedSequenceClassIDs[j];

				//for (const auto& keyval : commandIDsMap) // Look at each key-value pair
				//{
				//    if (keyval.second == predictedClassID) // If the value is 0...
				//    {
				//        return keyval.first; // ...return the first element in the pair
				//    }
				//}

				//std::string cmdName;

				for (auto X : commandIDsMap)
					if (X.second == generatedSequenceClassIDs[j]) {
						//cmdName = X.first;
						std::cout << "(" << X.first << ") ";
						break;
					}


				//std::cout << "(" << cmdName << ") ";
				if (j != generatedSequenceClassIDs.size() - 1) std::cout << " -> ";
			}

			std::cout << std::endl;
		}
	}

	//CUDAMathTest();

	return 0;
}

UserSimulator::UserSimulator(int inputNeurons, int hiddenLayerNeurons, int outputNeurons, double learningRate) : _inputWeights(hiddenLayerNeurons, inputNeurons), 
	_hiddenWeights(hiddenLayerNeurons, hiddenLayerNeurons), _weightsOutput(outputNeurons, hiddenLayerNeurons), _biasesHidden(hiddenLayerNeurons, 1), _biasesOutput(outputNeurons, 1) {

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
	//_inputWeights.Resize(hiddenLayerNeurons, inputNeurons);
	for (int i = 0; i < _inputWeights.GetRows(); i++) {
		for (int j = 0; j < _inputWeights.GetColumns(); j++) {
			_inputWeights(i, j) = unif(re) / 5 - 0.1;
		}
	}

	//_inputWeights.Print();

	// init hidden/recurrent weights
	//_hiddenWeights.Resize(hiddenLayerNeurons, hiddenLayerNeurons);
	for (int i = 0; i < _hiddenWeights.GetRows(); i++) {
		for (int j = 0; j < _hiddenWeights.GetColumns(); j++) {
			_hiddenWeights(i, j) = unif(re) / 5 - 0.1;
		}
	}

	//_hiddenWeights.Print();

	// init output weights

	//_weightsOutput.Resize(outputNeurons, hiddenLayerNeurons);
	for (int i = 0; i < _weightsOutput.GetRows(); i++) {
		for (int j = 0; j < _weightsOutput.GetColumns(); j++) {
			//weightsOutput(i, j) = unif(re) / 5 - 0.1;
			// initialize big output weights

			_weightsOutput(i, j) = unif(re) * 50;
			//_weightsOutput(i, j) = unif(re) / 5 - 0.1;
		}
	}

	//_weightsOutput.Print();

	// biases for hidden layer
	//_biasesHidden.Resize(hiddenLayerNeurons, 1);
	for (int i = 0; i < hiddenLayerNeurons; i++) {
		_biasesHidden(i, 0) = unif(re) / 5 - 0.1;
	}

	//_biasesHidden.Print();

	// biases for output layer
	//_biasesOutput.Resize(outputNeurons, 1);
	for (int i = 0; i < outputNeurons; i++) {
		//biasesOutput(i, 0) = unif(re) / 5 - 0.1;
		_biasesOutput(i, 0) = unif(re);
	}

	//_biasesOutput.Print();
}

void UserSimulator::ForwardProp(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode) {
	CUDAMatrix oneHotEncodedInputT = _mathHandler->TransposeMatrix(onehotEncodedInput);
	CUDAMatrix XI = _mathHandler->TransposeMatrix(oneHotEncodedInputT * _inputWeights);
	//XI.Print();
	CUDAMatrix XHCurrentTimeStep = sequencePosition == 0 ? XI : XI + (_hiddenWeights * _hiddenStepValues[sequencePosition - 1]) + _biasesHidden;
	// if first element in sequence XHidden at previous time step non existent, just take XI(nput)
	/*if (sequencePosition == 0) {
		XHCurrentTimeStep = XI;
	}
	else {
		XHCurrentTimeStep = XI + (_hiddenWeights * _hiddenStepValues[sequencePosition - 1]) + _biasesHidden;
	}*/

	//XHCurrentTimeStep.Print();

	CUDAMatrix activatedHiddenLayer = XHCurrentTimeStep;

	activatedHiddenLayer.tanh();
	_hiddenStepValues.push_back(activatedHiddenLayer);

	//activatedHiddenLayer.Print();

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

	//outputValuesActivated.Print();

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
		//CUDAMatrix lossGradT = _mathHandler->TransposeMatrix(lossGrad);
		outputWeightsGrad += _hiddenStepValues[i] * _mathHandler->TransposeMatrix(lossGrad);
		//outputBiasGrad = outputBiasGrad + lossGrad;

		// Backpropagate into the hidden layer
		//CUDAMatrix outputGrad = _mathHandler->TransposeMatrix(_weightsOutput) * lossGrad;
		//CUDAMatrix hiddenGrad = outputGrad + _mathHandler->TransposeMatrix(_hiddenWeights) * nextHiddenGrad;
		//CUDAMatrix tanhDerivative = _mathHandler->TanhDerivative(_hiddenStepValues[i]);
		//hiddenGrad = hiddenGrad.Array() * tanhDerivative.Array();

		// Accumulate gradients for hidden layer weights and biases
		if (i > 0) {
			//hiddenWeightsGrad = hiddenWeightsGrad + (_hiddenStepValues[i - 1] * _mathHandler->TransposeMatrix(hiddenGrad));
			//hiddenBiasGrad = hiddenBiasGrad + hiddenGrad;
		}

		// Accumulate gradients for input weights
		//inputWeightsGrad = inputWeightsGrad + (oneHotEncodedLabels[i] * _mathHandler->TransposeMatrix(hiddenGrad));

		// Update nextHiddenGrad for the next iteration
		//nextHiddenGrad = hiddenGrad;
	}

	// Apply learning rate adjustment
	double adjustedLearningRate = learningRate / _outputValues.size();

	// Update weights and biases
	/*_inputWeights = _inputWeights - (inputWeightsGrad * adjustedLearningRate);
	_hiddenWeights = _hiddenWeights - (hiddenWeightsGrad * adjustedLearningRate);
	_biasesHidden = _biasesHidden - (hiddenBiasGrad * adjustedLearningRate);
	_weightsOutput = _weightsOutput - (outputWeightsGrad * adjustedLearningRate);
	_biasesOutput = _biasesOutput - (outputBiasGrad * adjustedLearningRate);*/
}

int UserSimulator::PredictNextClickFromSequence(std::vector<CUDAMatrix> onehotEncodedLabels, bool performBackProp, bool verboseMode) {

	if (onehotEncodedLabels.size() == 0) return -1;

	_hiddenStepValues.clear();
	_outputValues.clear();
	//_oneHotEncodedClicks.clear();
	//for (auto i : _oneHotEncodedClicks) i.Destroy();
	//_oneHotEncodedClicks = onehotEncodedLabels;
	
	_totalLoss = 0;

	int allClasses = _biasesOutput.GetRows();

	for (int i = 0; i < onehotEncodedLabels.size() - (performBackProp ? 1 : 0); i++) {

		ForwardProp(onehotEncodedLabels[i], i, verboseMode);
	}

	//printf("TOTAL LOSS: %f\n", _totalLoss);

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

void UserSimulator::PrintPredictedClassProbabilities() {
	CUDAMatrix lastTimeStepProbabilities(_outputNeurons, 1);
	for (int i = 0; i < _outputValues[_outputValues.size() - 1].GetRows(); i++) {
		std::cout << "classID " << i << " " << _outputValues[_outputValues.size() - 1](i, 0) * 100 << "%" << std::endl;
	}
}