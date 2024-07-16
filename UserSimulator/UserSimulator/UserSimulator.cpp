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
	CUDAMatrix mat6(5, 2);

	mat1(0, 0) = 1;
	mat1(0, 1) = 2;
	mat1(1, 0) = 3;
	mat1(1, 1) = 4;

	/*mat1(0, 0) = 0.1;
	mat1(0, 1) = 0.2;
	mat1(1, 0) = 0.3;
	mat1(1, 1) = 0.4;*/

	mat2(0, 0) = 3;
	mat2(0, 1) = 6;
	mat2(1, 0) = 9;
	mat2(1, 1) = 12;

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

	mat6(0, 0) = 0;
	mat6(0, 1) = 0.1;
	mat6(1, 0) = 0.2;
	mat6(1, 1) = -0.3;
	mat6(2, 0) = 0.4;
	mat6(2, 1) = 0.5;
	mat6(3, 0) = 0.6;
	mat6(3, 1) = -0.3;
	mat6(4, 0) = 0.8;
	mat6(4, 1) = 0.9;

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

	printf("mat1 += mat2\n");
	mat1.Print();
	mat2.Print();
	mat1 += mat2;
	printf("mat1 po +=\n");
	mat1.Print();

	(mat1 * mat2).Print();

	CUDAMatrix tanhDer = userSimulator->GetMathEngine()->TanhDerivative(mat1);
	printf("tanhder\n");
	tanhDer.Print();

	//mat1.Print();

	CUDAMatrix mat5(5, 1);
	mat5(0, 0) = 0.4;
	mat5(1, 0) = 0.7;
	mat5(2, 0) = 3.2;
	mat5(3, 0) = 1.1;
	mat5(4, 0) = 0.9;

	printf("printing matrices\n");
	printf("\n");
	mat1.Print();
	mat2.Print();
	mat3.Print();
	mat4.Print();
	mat5.Print();

	std::vector<int> idsToOHE;
	idsToOHE.push_back(2);
	idsToOHE.push_back(0);
	std::vector<CUDAMatrix> oneHotEncodedLabels = userSimulator->GetMathEngine()->CreateOneHotEncodedVector(idsToOHE, 5);
	oneHotEncodedLabels[0].Print();
	oneHotEncodedLabels[1].Print();

	printf("mat5\n");
	mat6.Print();
	printf("row average mat5....\n");
	mat6.RowAverage().Print();
	mat6.RowAverageMatrix().Print();

	printf("tanh\n");
	mat6.tanh().Print();

	mat2.Print();
	mat1.Print();
	(mat2.Array() / mat1.Array()).Print();
}

int main() {
	bool verboseMode = false;

	int trainingSeqLength = 5;

	cudaDeviceReset();

	FileManager* fileManager = new FileManager();
	std::vector<std::vector<int>> allSequencesFromFile;
	std::tuple<std::vector<std::vector<int>>, std::map<std::string, int>> readFileRes;

	readFileRes = fileManager->ReadTXTFile(R"(C:\Users\joresg\git\BachelorProject\RNN\inputData\unixCommandData)", true, trainingSeqLength);
	//readFileRes = fileManager->ReadTXTFile(R"(C:\Users\joresg\git\BachelorProject\RNN\inputData\randomData)", true);
	allSequencesFromFile = std::get<0>(readFileRes);
	std::map<std::string, int> commandIDsMap = std::get<1>(readFileRes);

	// split into training and validation set

	int trainingSetSize = allSequencesFromFile.size() * 0.8;
	int validationSetSize = allSequencesFromFile.size() - trainingSetSize;

	std::vector<std::vector<int>> trainingSet;
	std::vector<std::vector<int>> validationSet;

	int allClasses = commandIDsMap.size();
	//int allClasses = 5;
	double learningRate = 0.00001;
	int inputNeurons = allClasses;
	int outputNeurons = allClasses;
	int hiddenNeurons = allClasses;

	UserSimulator* userSimulator = new UserSimulator(inputNeurons, hiddenNeurons, outputNeurons, learningRate, 32);

	for (int i = 0; i < trainingSetSize; i++) {
		trainingSet.push_back(allSequencesFromFile[i]);
	}

	for (int i = trainingSetSize; i < allSequencesFromFile.size(); i++) {
		validationSet.push_back(allSequencesFromFile[i]);
		//userSimulator->GetValidationSet().push_back(allSequencesFromFile[i]);
	}

	userSimulator->SetValidationSet(validationSet);

	//CUDAMathTest(userSimulator);

	int batchSize = userSimulator->GetBatchSize();
	int epochs = 20;
	int currentEpoch = 0;

	while (currentEpoch <= epochs) {

		/*if (currentEpoch == epochs) {
			printf("continue training with how many epochs?\n");
			std::string input;
			std::getline(std::cin, input);
			currentEpoch += std::stoi(input);

			if (std::stoi(input) == 0) break;

			printf("adjust learning rate\n");
			std::getline(std::cin, input);
			learningRate *= std::stod(input);
		}*/

		std::cout << "EPOCH: " << currentEpoch + 1 << std::endl;

		// STOCHASTIC IMPLEMENTATION BATCH SIZE = 1

		/*for (int k = 0; k < allSequencesFromFile.size(); k++) {
			std::vector<CUDAMatrix> oneHotEncodedInput = userSimulator->GetMathEngine()->CreateOneHotEncodedVector(allSequencesFromFile[k], allClasses);

			if (k > 0 && k % (int)(allSequencesFromFile.size() * 0.1) == 0) {
				std::cout << (k / (int)(allSequencesFromFile.size() * 0.1)) * 10 << "%" << std::endl;
			}

			printf("iteration: %d\n", k);

			userSimulator->PredictNextClickFromSequence(oneHotEncodedInput, true, verboseMode);
		}*/

		// BATCH IMPLEMENTATION

		int trainingExamplesCount = trainingSet.size();
		userSimulator->SetAllTrainingExamplesCount(trainingExamplesCount);

		for (int k = 0; k < trainingExamplesCount; k += batchSize) {
			if (k + batchSize > trainingExamplesCount - 1) {
				// todo adjust final batch size if smaller
				break;
			}
			std::vector<std::vector<int>>::const_iterator first = trainingSet.begin() + k;
			std::vector<std::vector<int>>::const_iterator last = trainingSet.begin() + (k + batchSize < trainingExamplesCount ? k + batchSize : trainingExamplesCount - 1);
			//std::vector<std::vector<int>>::const_iterator last = allSequencesFromFile.begin() + (k + batchSize < allSequencesFromFile.size() ? k + batchSize: 1) + batchSize;
			std::vector<std::vector<int>> newVec(first, last);
			std::vector<CUDAMatrix> oneHotEncodedInput = userSimulator->GetMathEngine()->CreateBatchOneHotEncodedVector(newVec, allClasses, batchSize);

			if (k > 0 && k % (int)(trainingExamplesCount * 0.1) == 0) {
				std::cout << (k / (int)(trainingExamplesCount * 0.1)) * 10 << "%" << std::endl;
			}
			printf("iteration: %d / %d\n", k, trainingExamplesCount);

			userSimulator->PredictNextClickFromSequence(oneHotEncodedInput, true, verboseMode, true);
		}

		currentEpoch++;

		// run validation for early stoppping

		if (userSimulator->EvaluateOnValidateSet() >= 0.8) {
			// TODO save model parameters and revert back to previous after degradation
			printf("0.8 accuarcy reached...\n");
			break;
		}
	}

	printf("FINISHED TRAINING\n");
	userSimulator->PrintAllParameters();


	// take user input for first click

	printf("BIASES OUTPUT\n");
	userSimulator->GetBiasesOutput().Print();

	for (auto it = commandIDsMap.cbegin(); it != commandIDsMap.cend(); ++it)
	{
		std::cout << it->first << " - " << it->second << "\n";
	}

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

		for (int i = 0; i < sequenceLength - splitInputIntegers.size(); i++) {
			//clickPredictor->ForwardProp()
			int predictedClassID = userSimulator->PredictNextClickFromSequence(generatedSequence, false, verboseMode, false);

			// randomness experiment
			//generatedSequence.push_back(CreateOneHotEncodedVecNormalized(allClasses, i % 5 == 0 ? 2 : predictedClassID));
			generatedSequence.push_back(CreateOneHotEncodedVecNormalized(allClasses, predictedClassID));

			if (generatedSequence.size() > trainingSeqLength + trainingSeqLength / 2) generatedSequence.erase(generatedSequence.begin());

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

UserSimulator::UserSimulator(int inputNeurons, int hiddenLayerNeurons, int outputNeurons, double learningRate, int batchSize) : _inputWeights(hiddenLayerNeurons, inputNeurons), 
	_hiddenWeights(hiddenLayerNeurons, hiddenLayerNeurons), _weightsOutput(outputNeurons, hiddenLayerNeurons), _biasesHidden(hiddenLayerNeurons, 1), _biasesOutput(outputNeurons, 1), _batchSize(batchSize),
	_batchBiasesHidden(hiddenLayerNeurons, batchSize), _batchBiasesOutput(outputNeurons, batchSize), _allTrainingExamplesCount(-1) {

	_learningRate = learningRate;
	_inputNeurons = inputNeurons;
	_hiddenLayerNeurons = hiddenLayerNeurons;
	_outputNeurons = outputNeurons;
	_allClasses = outputNeurons;
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
		double randomBias = unif(re) / 5 - 0.1;
		for (int j = 0; j < batchSize; j++) {
			_batchBiasesHidden(i, j) = randomBias;
		}
		//_biasesHidden(i, 0) = unif(re) / 5 - 0.1;
		_biasesHidden(i, 0) = randomBias;
	}

	//_biasesHidden.Print();

	// biases for output layer
	//_biasesOutput.Resize(outputNeurons, 1);
	for (int i = 0; i < outputNeurons; i++) {
		//_biasesOutput(i, 0) = unif(re) / 5 - 0.1;
		double randomBias = unif(re);
		for (int j = 0; j < batchSize; j++) {
			_batchBiasesOutput(i, j) = randomBias;
		}
		//_biasesOutput(i, 0) = unif(re);
		_biasesOutput(i, 0) = randomBias;
	}

	printf("BIASES OUTPUT AFTER INIT\n");
	_biasesOutput.Print();
	_biasesHidden.Print();

	//_biasesOutput.Print();
}

double UserSimulator::EvaluateOnValidateSet() {

	// TODO don't just pick top class but consider others with high probability as well

	// go through all examples in validation set, break them down and test next click
	int allTests = 0;
	int correctPredictions = 0;
	std::vector<CUDAMatrix> clickSequence;
	for (int i = 0; i < _validationSet.size(); i++) {
		std::vector<CUDAMatrix> oneHotEncodedValidationClickSequenceExample = _mathHandler->CreateOneHotEncodedVector(_validationSet[i], _allClasses);
		for (int j = 0; j < _validationSet[i].size() - 1; j++) {
			clickSequence.clear();
			std::vector<CUDAMatrix>::const_iterator first = oneHotEncodedValidationClickSequenceExample.begin();
			std::vector<CUDAMatrix>::const_iterator last = oneHotEncodedValidationClickSequenceExample.begin() + j + 1;
			std::vector<CUDAMatrix> slicedClicks(first, last);

			int predictedClick = PredictNextClickFromSequence(slicedClicks, false, false, false);
			allTests++;
			if (predictedClick == _validationSet[i][j + 1]) correctPredictions++;
		}
	}

	double validationSucess = correctPredictions / (double)allTests;
	printf("testing on validation set.... accuracy: %d%%\n", (int)(validationSucess * 100));
	return validationSucess;
}

void UserSimulator::ForwardProp(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode, bool trainMode) {
	//_inputWeights.Print();
	//printf("onehotEncodedInput\n");
	//onehotEncodedInput.Print();
	CUDAMatrix XI = _inputWeights * onehotEncodedInput;
	//XI.Print();
	CUDAMatrix XHCurrentTimeStep;
	// if first element in sequence XHidden at previous time step non existent, just take XI(nput)
	if (sequencePosition == 0) {
		XHCurrentTimeStep = XI;
	}
	else {
		//XHCurrentTimeStep = XI + (_hiddenWeights * _hiddenStepValues[sequencePosition - 1]) + _biasesHidden;
		XHCurrentTimeStep = XI + (_hiddenWeights * _hiddenStepValues[sequencePosition - 1]) + (trainMode ? _batchBiasesHidden : _biasesHidden);
	}

	//CUDAMatrix XHCurrentTimeStepT = _mathHandler->TransposeMatrix(XHCurrentTimeStep);
	/*CUDAMatrix XHCurrentTimeStepTanh = XHCurrentTimeStep;
	XHCurrentTimeStepTanh.tanh();
	CUDAMatrix activatedHiddenLayer = XHCurrentTimeStepTanh;*/
	CUDAMatrix activatedHiddenLayer = XHCurrentTimeStep.tanh();
	//printf("activatedHiddenLayer\n");
	//activatedHiddenLayer.Print();
	_hiddenStepValues.push_back(activatedHiddenLayer);


	// compute output

	//CUDAMatrix outputValuesUnactivated = _weightsOutput * _mathHandler->TransposeMatrix(activatedHiddenLayer) + _biasesOutput;
	//CUDAMatrix outputValuesUnactivated = _weightsOutput * activatedHiddenLayer + _biasesOutput;
	CUDAMatrix outputValuesUnactivated = _weightsOutput * activatedHiddenLayer + (trainMode ? _batchBiasesOutput : _biasesOutput);

	//printf("calculating sum for softmax...........\n");
	//outputValuesUnactivated.Print();

	CUDAMatrix sumsForSoftmax(outputValuesUnactivated.GetRows(), outputValuesUnactivated.GetColumns());
	double sumForSoftmax = 0;
	for (int j = 0; j < sumsForSoftmax.GetColumns(); j++) {
		sumForSoftmax = 0;
		for (int i = 0; i < outputValuesUnactivated.GetRows(); i++) {
			sumForSoftmax += std::exp(outputValuesUnactivated(i, j));
		}
		for (int i = 0; i < outputValuesUnactivated.GetRows(); i++) {
			sumsForSoftmax(i, j) = sumForSoftmax;
		}
	}

	CUDAMatrix outputValuesActivated = outputValuesUnactivated.exp();
	//printf("outputValuesUnactivated\n");
	//outputValuesUnactivated.Print();
	//printf("outputValuesActivated after exp\n");
	//outputValuesActivated.Print();
	//sumsForSoftmax.Print();

	//outputValuesActivated = outputValues;
	//CUDAMatrix outputValuesActivated(outputValuesUnactivated.GetRows(), _batchSize);

	// use CUDA instead
	/*for (int i = 0; i < outputValuesActivated.GetRows(); i++) {
		outputValuesActivated(i, 0) = std::exp(outputValuesUnactivated(i, 0)) / sumForSoftmax;
	}*/

	outputValuesActivated = outputValuesActivated.Array() / sumsForSoftmax.Array();
	//printf("outputValuesActivated\n");
	//outputValuesActivated.Print();

	//printf("outputValuesActivated\n");
	//outputValuesActivated.Print();

	//std::cout << "PREDICTED: " << std::endl << outputValuesActivated << std::endl;

	_outputValues.push_back(outputValuesActivated);

	if (verboseMode)
	{
		std::cout << "_outputValuesActivated" << std::endl;
		outputValuesActivated.Print();
	}


	// test print predicted sequence
	double classID = -1;

	double maxProbability = 0;
	for (int j = 0; j < outputValuesActivated.GetRows(); j++) {
		if (outputValuesActivated(j, 0) > maxProbability) {
			maxProbability = outputValuesActivated(j, 0);
			classID = j;
		}
	}

	// calculate loss for debugging purposes
	double crossEntropyLoss = 0;
	_totalLoss -= std::log(maxProbability);
}

void UserSimulator::BackProp(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode) {
	if (verboseMode) std::cout << "Back prop" << std::endl;

	// Initialize gradients
	CUDAMatrix outputWeightsGrad = CUDAMatrix::Zero(_weightsOutput.GetRows(), _weightsOutput.GetColumns());
	CUDAMatrix outputBiasGrad = CUDAMatrix::Zero(_outputNeurons, this->_batchSize);
	//CUDAMatrix outputBiasGrad = CUDAMatrix::Zero(_outputNeurons, 1);
	CUDAMatrix hiddenWeightsGrad = CUDAMatrix::Zero(_hiddenWeights.GetRows(), _hiddenWeights.GetColumns());
	CUDAMatrix hiddenBiasGrad = CUDAMatrix::Zero(_hiddenLayerNeurons, this->_batchSize);
	//CUDAMatrix hiddenBiasGrad = CUDAMatrix::Zero(_hiddenLayerNeurons, 1);
	CUDAMatrix inputWeightsGrad = CUDAMatrix::Zero(_inputWeights.GetRows(), _inputWeights.GetColumns());

	CUDAMatrix nextHiddenGrad = CUDAMatrix::Zero(_hiddenLayerNeurons, this->_batchSize);
	//CUDAMatrix nextHiddenGrad = CUDAMatrix::Zero(_hiddenLayerNeurons, 1);

	// Iterate over each timestep from last to first
	for (int i = _outputValues.size() - 1; i >= 0; i--) {
		// Cross-entropy loss gradient w.r.t. softmax input
		CUDAMatrix lossGrad = _outputValues[i] - oneHotEncodedLabels[i + 1]; // Gradient of softmax + cross-entropy
		
		// Gradients for the output layer
		//outputWeightsGrad += lossGrad.RowAverage() * _mathHandler->TransposeMatrix(_hiddenStepValues[i]);
		outputWeightsGrad += lossGrad * _mathHandler->TransposeMatrix(_hiddenStepValues[i]);

		outputBiasGrad += lossGrad;

		// Backpropagate into the hidden layer
		CUDAMatrix outputGrad = _mathHandler->TransposeMatrix(_weightsOutput) * lossGrad;
		CUDAMatrix hiddenGrad = outputGrad + _mathHandler->TransposeMatrix(_hiddenWeights) * nextHiddenGrad;
		CUDAMatrix tanhDerivative = _mathHandler->TanhDerivative(_hiddenStepValues[i]);

		hiddenGrad = hiddenGrad.Array() * tanhDerivative.Array();

		// Accumulate gradients for hidden layer weights and biases
		if (i > 0) {
			hiddenWeightsGrad += hiddenGrad * _mathHandler->TransposeMatrix(_hiddenStepValues[i - 1]);
			hiddenBiasGrad += hiddenGrad;
		}

		// Accumulate gradients for input weights
		inputWeightsGrad += hiddenGrad * _mathHandler->TransposeMatrix(_oneHotEncodedClicks[i]);

		// Update nextHiddenGrad for the next iteration
		nextHiddenGrad = hiddenGrad;
	}

	/*printf("outputBiasGrad.........\n");
	outputBiasGrad.Print();*/

	// Apply learning rate adjustment
	double adjustedLearningRate = learningRate * std::sqrt(_batchSize);
	//double adjustedLearningRate = learningRate / _outputValues.size();

	//inputWeightsGrad.Print();

	_inputWeights -= inputWeightsGrad * (1.0 / _batchSize) * adjustedLearningRate;
	_hiddenWeights -= hiddenWeightsGrad * (1.0 / _batchSize) * adjustedLearningRate;
	_biasesHidden -= hiddenBiasGrad.RowAverage() * adjustedLearningRate;
	_weightsOutput -= outputWeightsGrad * (1.0 / _batchSize) * adjustedLearningRate;
	_biasesOutput -= outputBiasGrad.RowAverage() * adjustedLearningRate;

	_batchBiasesHidden -= hiddenBiasGrad.RowAverageMatrix() * adjustedLearningRate;
	_batchBiasesOutput -= outputBiasGrad.RowAverageMatrix() * adjustedLearningRate;
}

int UserSimulator::PredictNextClickFromSequence(std::vector<CUDAMatrix> onehotEncodedLabels, bool performBackProp, bool verboseMode, bool trainMode) {

	_hiddenStepValues.clear();
	_outputValues.clear();
	_oneHotEncodedClicks.clear();
	_oneHotEncodedClicks = onehotEncodedLabels;

	_totalLoss = 0;

	int allClasses = _biasesOutput.GetRows();

	for (int i = 0; i < onehotEncodedLabels.size() - (performBackProp ? 1 : 0); i++) {

		ForwardProp(onehotEncodedLabels[i], i, verboseMode, trainMode);
	}

	//printf("----------------------------finished forward prop-----------------------------------------\n");
	//PrintAllParameters();

	/*printf("hidden values and output values after forward prop\n");
	printf("hidden\n");
	for (int i = 0; i < _hiddenStepValues.size(); i++) {
		_hiddenStepValues[i].Print();
	}
	printf("output\n");
	for (int i = 0; i < _outputValues.size(); i++) {
		_outputValues[i].Print();
	}*/

	int firstClickID = -1;
	int classID = -1;

	if (!trainMode)
	{
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
	}

	if (performBackProp && verboseMode) std::cout << "TOTAL LOSS: " << _totalLoss << std::endl;

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