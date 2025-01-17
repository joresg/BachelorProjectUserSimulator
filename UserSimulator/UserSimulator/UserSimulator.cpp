#include "UserSimulator.h"
#include "FileManager.h"
#include "HyperParamGrid.h"
#include <numeric>
#include <string>

#include <stdio.h>

using json = nlohmann::json;

static void SerializeModel(UserSimulator* userSimulator, std::string fileName) {
	std::ofstream ofs(fileName);
	boost::archive::text_oarchive oa(ofs);
	oa << userSimulator;
}

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

	double matNorm = mat4.Norm();
	printf("%f\n", matNorm);

	printf("printing matrices\n");
	printf("\n");
	mat1.Print();
	mat2.Print();
	mat3.Print();
	mat4.Print();

	CUDAMatrix matAndVecMul = mat3 * mat4.Vec();
	matAndVecMul.Print();

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

	//CUDAMatrix tanhDer = userSimulator->GetMathEngine()->TanhDerivative(mat1);
	//printf("tanhder\n");
	//tanhDer.Print();

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
	std::vector<CUDAMatrix> oneHotEncodedLabels = userSimulator->GetMathEngine()->CreateOneHotEncodedVectorSequence(idsToOHE, 5);
	oneHotEncodedLabels[0].Print();
	oneHotEncodedLabels[1].Print();

	printf("mat5\n");
	mat6.Print();
	printf("row average mat5....\n");
	mat6.RowAverage().Print();

	printf("tanh\n");
	mat6.tanh().Print();

	mat2.Print();
	mat1.Print();
	(mat2.Array() / mat1.Array()).Print();

	printf("matrix add row vec\n");
	mat3.Print();
	mat4.Print();
	(mat3 + mat4.Vec()).Print();
}

int main() {
	UserSimulator* bestModel = nullptr;
	int allClasses;
	bool trainModel = true;

	if (trainModel)
	{
		bool verboseMode = false;

		// 0 regular 1 GRU
		int RNNType = 0;

		FileManager* fileManager = new FileManager();
		std::vector<std::vector<int>> allSequencesFromFile;
		std::map<std::string, std::tuple<int, int>> commandIDsMap;
		std::tuple<std::vector<std::vector<int>>, std::map<std::string, std::tuple<int, int>>> readFileRes;

		//const char* filePathTrainingData = R"(..\inputData\unixCommandData.txt)";
		const char* filePathTrainingData = R"(..\inputData\allSequences.txt)";

		allClasses = fileManager->AllClassesFromFile(filePathTrainingData);
		int inputNeurons = allClasses;
		int outputNeurons = allClasses;

		HyperParamGrid* paramGridSearch = new HyperParamGrid(allClasses, RNNType == 0 ? NoGates : GRU);
		bool lastModelConverged = false;
		double lastLR = 0;
		bool lrWarmup = true;
		bool useLRDecay = true;
		int lrWarmupSteps = 500;
		double lrWarmupStepIncrease = 0;
		bool useCurriculumLearning = true;
		bool useDropout = true;
		bool useRecurrentDropout = true;
		bool useWeightedLoss = true;
		// learning rate, hidden units, seq length, batch size
		for (const auto& paramCombo : paramGridSearch->HyperParameterGridSearch()) {

			if (lastModelConverged && std::get<0>(paramCombo) < lastLR) {
				/*printf("last model with same parameters converged, skipping lower learning rate...\n");
				lastLR = std::get<0>(paramCombo);
				lastModelConverged = true;
				continue;*/
			}

			lrWarmupStepIncrease = std::get<0>(paramCombo) / (double)lrWarmupSteps;

			lastLR = std::get<0>(paramCombo);
			lastModelConverged = false;

			cudaDeviceReset();

			//readFileRes = fileManager->ReadTXTFile(R"(C:\Users\joresg\git\BachelorProjectCUDA\UserSimulator\inputData\unixCommandData.txt)", true, std::get<2>(paramCombo));
			readFileRes = fileManager->ReadTXTFile(filePathTrainingData, true, std::get<2>(paramCombo));
			allSequencesFromFile = std::get<0>(readFileRes);
			commandIDsMap = std::get<1>(readFileRes);
			int samplesCnt = 0;
			for (const auto& cmd : commandIDsMap) {
				samplesCnt += std::get<1>(std::get<1>(cmd));
			}

			// split into training and validation set

			int trainingSetSize = allSequencesFromFile.size() * 0.8;

			// batch can't be too big, let it be relative to training set size
			if (std::get<3>(paramCombo) > trainingSetSize / 16) continue;

			std::string fileNameModelParameters = "..\\SavedModelParameters\\user_simulator_";
			fileNameModelParameters += split(filePathTrainingData, '\\').back();
			fileNameModelParameters += "_";
			fileNameModelParameters += std::to_string(std::get<2>(paramCombo));
			fileNameModelParameters += ".dat";

			if (bestModel == nullptr)
			{
				std::ifstream file(fileNameModelParameters);

				if (file) {
					printf("existing model parameters found, loading model %s...\n", fileNameModelParameters.c_str());
					bestModel = new UserSimulator();
					std::ifstream ifs(fileNameModelParameters);
					boost::archive::text_iarchive ia(ifs);
					ia >> bestModel;
					allClasses = bestModel->GetAllClasses();
				}
			}

			int validationSetSize = allSequencesFromFile.size() - trainingSetSize;

			std::vector<std::vector<int>> trainingSet;
			std::vector<std::vector<int>> trainingSetCurriculum;
			std::vector<std::vector<int>> validationSet;
			std::vector<std::vector<int>> validationSetCurriculum;

			UserSimulator* userSimulator = new UserSimulator(true, inputNeurons, std::get<1>(paramCombo), outputNeurons, lrWarmup ? 0 : std::get<0>(paramCombo), std::get<3>(paramCombo), std::get<2>(paramCombo));
			userSimulator->SetAllClasses(allClasses);
			if (RNNType == 1) userSimulator->SetGatedUnits(GRU);
			userSimulator->SetCmdIDsMap(commandIDsMap);
			userSimulator->SetGradientClippingThreshold(10);
			userSimulator->SetTotalNumberOfSamples(samplesCnt);
			userSimulator->SetUseDropout(useDropout, useRecurrentDropout);
			userSimulator->SetUseWeightedLoss(useWeightedLoss);

			CUDAMatrix classWeights(allClasses, 1);
			double classWeightsMin = DBL_MAX;
			double classWeightsMax = 0;
			for (int i = 0; i < classWeights.GetRows(); i++) {
				std::string className = userSimulator->GetCommandNameFromID(i);
				// inverse class frequency
				classWeights(i, 0) = (double)samplesCnt / ((double)allClasses * std::get<1>(commandIDsMap[className]));
				if (classWeights(i, 0) > classWeightsMax) classWeightsMax = classWeights(i, 0);
				else if (classWeights(i, 0) < classWeightsMin) classWeightsMin = classWeights(i, 0);
			}

			double shiftLow = 1.0;
			double shiftHigh = 2.0;

			if (useWeightedLoss)
			{
				// normalize between shiftLow and shiftHigh
				for (int i = 0; i < classWeights.GetRows(); i++) {
					classWeights(i, 0) = shiftLow + ((classWeights(i, 0) - classWeightsMin) * (shiftHigh - shiftLow) / (classWeightsMax - classWeightsMin));
				}
			}

			userSimulator->SetWeightsForClasses(classWeights);

			//classWeights.Print();

			//CUDAMathTest(userSimulator);

			std::shuffle(allSequencesFromFile.begin(), allSequencesFromFile.end(), userSimulator->GetMathEngine()->GetRandomEngine());

			for (int i = 0; i < trainingSetSize; i++) {
				trainingSet.push_back(allSequencesFromFile[i]);

				std::vector<int>::const_iterator first = allSequencesFromFile[i].begin();
				std::vector<int>::const_iterator last = allSequencesFromFile[i].begin() + std::max(userSimulator->GetTrainingSequenceLength() / 5, 2);
				std::vector<int> newVec(first, last);

				trainingSetCurriculum.push_back(newVec);
			}

			for (int i = trainingSetSize; i < allSequencesFromFile.size(); i++) {
				validationSet.push_back(allSequencesFromFile[i]);

				std::vector<int>::const_iterator first = allSequencesFromFile[i].begin();
				std::vector<int>::const_iterator last = allSequencesFromFile[i].begin() + std::max(userSimulator->GetTrainingSequenceLength() / 5, 2);
				std::vector<int> newVec(first, last);

				validationSetCurriculum.push_back(newVec);
			}

			userSimulator->SetValidationSet(validationSetCurriculum);

			int batchSize = userSimulator->GetBatchSize();
			int currentEpoch = 0;
			int maxAccEpoch = 0;
			double maxAccAchieved = DBL_MAX;
			double desiredAcc = 0.9;
			int progress = 1;
			double minLR = lastLR * 0.1;

			std::string hiddenNeuronsString = std::accumulate(std::get<1>(paramCombo).begin(), std::get<1>(paramCombo).end(), std::string{},
				[](const std::string& acc, const std::tuple<int, LayerActivationFuncs>& elem) {
					int num = std::get<0>(elem);
					const LayerActivationFuncs& obj = std::get<1>(elem);
					std::stringstream ss;
					ss << acc;
					if (!acc.empty()) {
						ss << ", ";
					}
					int pos = static_cast<LayerActivationFuncs>(obj);
					//ss << num << ":" << obj;
					std::string actFuncName = "";
					if (pos == 0) actFuncName = "ReLU";
					else if (pos == 1) actFuncName = "tanh";
					else if (pos == 2) actFuncName = "sigmoid";
					else if (pos == 3) actFuncName = "leakyReLU";
					//ss << num << ":" << (pos == 0 ? "relu" : "tanh");
					ss << num << ":" << actFuncName;

					return ss.str();
				});

			std::string RNNType = userSimulator->GetDirectionality() ? "Bi" : "";

			RNNType += userSimulator->GetGatedUnits() == NoGates ? "RNN" : "GRU";

			printf("%s, learning rate: %.10f, allClasses: %d, hidden units: {%s}, seq length: %d, batch size: %d\n", RNNType.c_str(), std::get<0>(paramCombo), allClasses, hiddenNeuronsString.c_str(), std::get<2>(paramCombo), std::get<3>(paramCombo));
			printf("cross entropy for best model so far: %f\n", bestModel != nullptr ? bestModel->GetModelACcOnValidationData() : -1);

			bool loweredLearningRate = false;
			int loweredLearningRateEpoch = 0;

			if (useCurriculumLearning)
			{
				double lossBeforeCurriculumTraining = userSimulator->EvaluateOnValidateSet(1);
				printf("loss before curriculum training: %f\n", lossBeforeCurriculumTraining);

				// curriculum learning for early timesteps which don't have much context

				userSimulator->SetLearningRate(lastLR * 8);

				printf("starting curriculum training...\n");

				while (true) {
					progress = 1;
					std::cout << "EPOCH: " << currentEpoch << std::endl;

					int trainingExamplesCount = trainingSetCurriculum.size();
					userSimulator->SetAllTrainingExamplesCount(trainingExamplesCount);

					for (int k = 0; k < trainingExamplesCount; k += batchSize) {
						std::vector<std::vector<int>>::const_iterator first = trainingSetCurriculum.begin() + k;
						std::vector<std::vector<int>>::const_iterator last = trainingSetCurriculum.begin() + (k + batchSize < trainingExamplesCount ? k + batchSize : trainingExamplesCount);
						std::vector<std::vector<int>> newVec(first, last);
						userSimulator->SetBatchSize(newVec.size());
						std::tuple< std::vector<CUDAMatrix>, std::vector<CUDAMatrix>> oneHotEncodedInputs = userSimulator->GetMathEngine()->CreateBatchOneHotEncodedVector(newVec, classWeights, allClasses, userSimulator->GetBatchSize());
						std::vector<CUDAMatrix> oneHotEncodedInput = std::get<0>(oneHotEncodedInputs);
						std::vector<CUDAMatrix> oneHotEncodedInputMask = std::get<1>(oneHotEncodedInputs);

						if (k > 0 && k / (int)(trainingExamplesCount * 0.1) >= progress && progress <= 10) {
							std::cout << (k / (int)(trainingExamplesCount * 0.1)) * 10 << "%" << std::endl;
							progress++;
						}
						//printf("iteration: %d / %d\n", k, trainingExamplesCount);

						userSimulator->PredictNextClickFromSequence(oneHotEncodedInput, true, verboseMode, true, false);
						if (lrWarmup && userSimulator->GetLearningRate() < lastLR) userSimulator->SetLearningRate(userSimulator->GetLearningRate() + lrWarmupStepIncrease);
						else lrWarmup = false;
					}

					userSimulator->SetBatchSize(batchSize);

					// run validation for early stoppping, learn while cross entropy loss on validation set is improving

					if (currentEpoch >= 0)
					{
						double currentLoss = userSimulator->EvaluateOnValidateSet(1);

						if (std::isinf(currentLoss) || currentLoss == DBL_MAX) {
							printf("error\n");
							if (maxAccAchieved < currentLoss) {
								userSimulator->RestoreBestParameters();
								userSimulator->SetModelAccOnValidationData(maxAccAchieved);
							}
							else {
								userSimulator->GetAllClasses();
								userSimulator->SetModelAccOnValidationData(currentLoss);
							}
							break;
						}
						else if (currentLoss > maxAccAchieved * 0.92 && currentEpoch - maxAccEpoch > 3 || currentLoss < 0.1 || currentEpoch > 6) {
							if (loweredLearningRate && currentEpoch - loweredLearningRateEpoch > 3 || currentLoss < 0.1 || currentEpoch > 6)
							{
								printf("learning converged....\n");

								if (currentLoss < maxAccAchieved * 0.99)
								{
									lastModelConverged = true;
									userSimulator->SetModelAccOnValidationData(currentLoss);
									break;
								}
								else {
									lastModelConverged = true;
									userSimulator->RestoreBestParameters();
									userSimulator->SetModelAccOnValidationData(maxAccAchieved);
									break;
								}
							}
							else if (!loweredLearningRate) {
								printf("learning might be in local optimum, lowering lr....\n");
								userSimulator->SetLearningRate(userSimulator->GetLearningRate() * 0.1);
								loweredLearningRateEpoch = currentEpoch;
								loweredLearningRate = true;
							}
						}
						else if (currentLoss < maxAccAchieved * 0.99) {
							maxAccAchieved = currentLoss;
							maxAccEpoch = currentEpoch;
							userSimulator->CopyParameters();
						}
						else if (currentLoss > maxAccAchieved * 1.05) {
							printf("model unstable, lowering lr\n");
							userSimulator->SetLearningRate(userSimulator->GetLearningRate() * 0.1);
						}
						else if (currentLoss > maxAccAchieved * 1.2) {
							printf("model deteriorated too much stopping training\n");
							userSimulator->RestoreBestParameters();
							break;
						}
					}
					if (!lrWarmup && useLRDecay && userSimulator->GetLearningRate() > minLR) {
						userSimulator->SetLearningRate(lastLR * std::exp(-0.05 * currentEpoch));
					}

					currentEpoch++;
				} 
			}

			userSimulator->SetBatchSize(batchSize);
			userSimulator->SetLearningRate(lastLR);
			userSimulator->SetValidationSet(validationSet);

			if (useWeightedLoss)
			{
				shiftLow = 1.0;
				shiftHigh = 2.0;

				// normalize between shiftLow and shiftHigh
				for (int i = 0; i < classWeights.GetRows(); i++) {
					classWeights(i, 0) = shiftLow + ((classWeights(i, 0) - classWeightsMin) * (shiftHigh - shiftLow) / (classWeightsMax - classWeightsMin));
				}

				userSimulator->SetWeightsForClasses(classWeights);
			}

			//userSimulator->SetWeightsForClasses(classWeights);

			printf("starting training...\n");

			double lossBeforeTraining = userSimulator->EvaluateOnValidateSet(0);
			printf("loss before training: %f\n", lossBeforeTraining);

			currentEpoch = 0;
			maxAccEpoch = 0;
			maxAccAchieved = DBL_MAX;
			desiredAcc = 0.9;
			progress = 1;
			minLR = lastLR * 0.1;

			loweredLearningRate = false;
			loweredLearningRateEpoch = 0;

			while (true) {
				progress = 1;
				std::cout << "EPOCH: " << currentEpoch << std::endl;

				// BATCH IMPLEMENTATION

				int trainingExamplesCount = trainingSet.size();
				userSimulator->SetAllTrainingExamplesCount(trainingExamplesCount);

				for (int k = 0; k < trainingExamplesCount; k += batchSize) {
					std::vector<std::vector<int>>::const_iterator first = trainingSet.begin() + k;
					std::vector<std::vector<int>>::const_iterator last = trainingSet.begin() + (k + batchSize < trainingExamplesCount ? k + batchSize : trainingExamplesCount);
					std::vector<std::vector<int>> newVec(first, last);
					userSimulator->SetBatchSize(newVec.size());
					std::tuple< std::vector<CUDAMatrix>, std::vector<CUDAMatrix>> oneHotEncodedInputs = userSimulator->GetMathEngine()->CreateBatchOneHotEncodedVector(newVec, classWeights, allClasses, userSimulator->GetBatchSize());
					std::vector<CUDAMatrix> oneHotEncodedInput = std::get<0>(oneHotEncodedInputs);
					std::vector<CUDAMatrix> oneHotEncodedInputMask = std::get<1>(oneHotEncodedInputs);

					if (k > 0 && k / (int)(trainingExamplesCount * 0.1) >= progress && progress <= 10) {
						std::cout << (k / (int)(trainingExamplesCount * 0.1)) * 10 << "%" << std::endl;
						progress++;
					}
					//printf("iteration: %d / %d\n", k, trainingExamplesCount);

					userSimulator->PredictNextClickFromSequence(oneHotEncodedInput, true, verboseMode, true, false);
					if (lrWarmup && userSimulator->GetLearningRate() < lastLR) userSimulator->SetLearningRate(userSimulator->GetLearningRate() + lrWarmupStepIncrease);
					else lrWarmup = false;
				}

				userSimulator->SetBatchSize(batchSize);

				// run validation for early stoppping, learn while cross entropy loss on validation set is improving

				if (currentEpoch >= 0)
				{
					double currentLoss = userSimulator->EvaluateOnValidateSet(0);

					if (std::isinf(currentLoss) || currentLoss == DBL_MAX) {
						printf("error\n");
						if (maxAccAchieved < currentLoss) {
							userSimulator->RestoreBestParameters();
							userSimulator->SetModelAccOnValidationData(maxAccAchieved);
						}
						else {
							userSimulator->GetAllClasses();
							userSimulator->SetModelAccOnValidationData(currentLoss);
						}
						break;
					}
					else if ((currentLoss > maxAccAchieved * 0.92 && currentEpoch - maxAccEpoch > 5) || currentLoss <= 0.05 || currentEpoch > 45) {
						if ((loweredLearningRate && currentEpoch - loweredLearningRateEpoch > 5) || currentLoss <= 0.05 || currentEpoch > 45)
						{
							printf("learning converged....\n");

							if (currentLoss < maxAccAchieved * 0.99)
							{
								lastModelConverged = true;
								userSimulator->SetModelAccOnValidationData(currentLoss);
								break;
							}
							else {
								lastModelConverged = true;
								userSimulator->RestoreBestParameters();
								userSimulator->SetModelAccOnValidationData(maxAccAchieved);
								break;
							}
						}
						else if(!loweredLearningRate) {
							printf("learning might be in local optimum, lowering lr....\n");
							userSimulator->SetLearningRate(userSimulator->GetLearningRate() * 0.1);
							loweredLearningRateEpoch = currentEpoch;
							loweredLearningRate = true;
						}
					}
					else if (currentLoss < maxAccAchieved * 0.99) {
						maxAccAchieved = currentLoss;
						maxAccEpoch = currentEpoch;
						userSimulator->CopyParameters();
					}
					else if (currentLoss > maxAccAchieved * 1.05) {
						printf("model unstable, lowering lr\n");
						userSimulator->SetLearningRate(userSimulator->GetLearningRate() * 0.1);
					}
					else if (currentLoss > maxAccAchieved * 1.2) {
						printf("model deteriorated too much stopping training\n");
						userSimulator->RestoreBestParameters();
						break;
					}
				}
				if (!lrWarmup && useLRDecay && userSimulator->GetLearningRate() > minLR) {
					userSimulator->SetLearningRate(lastLR * std::exp(-0.05 * currentEpoch));
				}

				currentEpoch++;
			}

			if (bestModel == nullptr || userSimulator->GetModelACcOnValidationData() < bestModel->GetModelACcOnValidationData()) {
				bestModel = userSimulator;
				SerializeModel(bestModel, fileNameModelParameters);
			}
		}

		printf("finished searching for optimal hpyerparameters\n");
		//SerializeModel(bestModel);
	}
	else {
		std::string fileName = "..\\SavedModelParameters\\user_simulator_allSequences.txt_5.dat";

		std::ifstream file(fileName);

		if (file) {
			printf("existing model parameters found, loading model %s...\n", fileName.c_str());
			bestModel = new UserSimulator();
			std::ifstream ifs(fileName);
			boost::archive::text_iarchive ia(ifs);
			ia >> bestModel;
			allClasses = bestModel->GetAllClasses();
		}
		else printf("couldn't load model parameters...\n");
	}

	crow::SimpleApp app;

	CROW_ROUTE(app, "/predictNextAction").methods(crow::HTTPMethod::POST)([&](const crow::request& req) {
		auto input_json = json::parse(req.body);

		// Assume input sequence is an array of integers in JSON
		std::vector<std::string> inputSequenceStrings = input_json["sequence"].get<std::vector<std::string>>();
		std::vector<int> inputSequence;
		for (const auto& cmdName : inputSequenceStrings) {
			inputSequence.push_back(bestModel->GetCommandIDFromName(cmdName));
		}

		std::vector<CUDAMatrix> oneHotEncodedSequence = bestModel->GetMathEngine()->CreateOneHotEncodedVectorSequence(inputSequence, allClasses);

		std::deque<std::tuple<int, double>> predictedSequenceIDs = bestModel->PredictNextClickFromSequence(oneHotEncodedSequence, false, false, false, false, 10);

		std::deque<std::tuple<std::string, double>> predictedSequence;

		for (const auto& predictedCommand : predictedSequenceIDs) {
			//double cmdProbability = const_cast<double&>(std::get<1>(predictedCommand));
			double cmdProbability = std::get<1>(predictedCommand);
			predictedSequence.push_back(std::make_tuple<std::string, double>(bestModel->GetCommandNameFromID(std::get<0>(predictedCommand)), cmdProbability * 1));
		}

		json response;
		response["predicted_sequence"] = predictedSequence;

		return crow::response(response.dump());
		});

	app.port(18080).multithreaded().run();

	return 0;
}

UserSimulator::UserSimulator(){
	unsigned long int randSeed = 18273;
	_mathHandler = new MathHandler(randSeed);
}

UserSimulator::UserSimulator(bool isBiRNN, int inputNeurons, std::vector<std::tuple<int, LayerActivationFuncs>> hiddenLayerNeurons, int outputNeurons, double learningRate, int batchSize, int trainingSeqLength) : 
	_weightsOutput(outputNeurons, std::get<0>(hiddenLayerNeurons[hiddenLayerNeurons.size() - 1]) * (isBiRNN ? 2 : 1)), _weightsOutputBack(outputNeurons, std::get<0>(hiddenLayerNeurons[hiddenLayerNeurons.size() - 1])),
	_batchSize(batchSize), _allTrainingExamplesCount(-1), _gradientClippingThresholdMax(10), _gradientClippingThresholdMin(10), _totalNumberOfSamples(-1), _modelAccuracy(DBL_MAX), _dropoutRate(0.1), 
	_isBiRNN(isBiRNN), _useDropout(true), _useRecurrentDropout(true) {

	_gatedUnits = NoGates;

	// on stack or heap, memory should be managed either way through destructor...
	bool firstHiddenLayer = true;
	int prevHlc = 0;
	for (const auto& i : hiddenLayerNeurons) {
		if (firstHiddenLayer) {
			_inputWeights.push_back(CUDAMatrix(std::get<0>(i), inputNeurons));
			_inputWeightsBack.push_back(CUDAMatrix(std::get<0>(i), inputNeurons));

			_updateGateInput.push_back(CUDAMatrix(std::get<0>(i), inputNeurons));
			_updateGateHidden.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(i)));
			_updateGateBias.push_back(CUDAMatrix(std::get<0>(i), 1));

			_resetGateInput.push_back(CUDAMatrix(std::get<0>(i), inputNeurons));
			_resetGateHidden.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(i)));
			_resetGateBias.push_back(CUDAMatrix(std::get<0>(i), 1));

			_candidateActivationInput.push_back(CUDAMatrix(std::get<0>(i), inputNeurons));
			_candidateActivationHidden.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(i)));
			_candidateActivationBias.push_back(CUDAMatrix(std::get<0>(i), 1));

			firstHiddenLayer = false;
		}
		else {
			_inputWeights.push_back(CUDAMatrix(std::get<0>(hiddenLayerNeurons[prevHlc]), std::get<0>(i)));
			_inputWeightsBack.push_back(CUDAMatrix(std::get<0>(hiddenLayerNeurons[prevHlc]), std::get<0>(i)));

			_updateGateInput.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(hiddenLayerNeurons[prevHlc])));
			_updateGateHidden.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(i)));
			_updateGateBias.push_back(CUDAMatrix(std::get<0>(i), 1));

			_resetGateInput.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(hiddenLayerNeurons[prevHlc])));
			_resetGateHidden.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(i)));
			_resetGateBias.push_back(CUDAMatrix(std::get<0>(i), 1));

			_candidateActivationInput.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(hiddenLayerNeurons[prevHlc])));
			_candidateActivationHidden.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(i)));
			_candidateActivationBias.push_back(CUDAMatrix(std::get<0>(i), 1));

			prevHlc++;
		}
		_hiddenWeights.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(i)));
		_hiddenWeightsBack.push_back(CUDAMatrix(std::get<0>(i), std::get<0>(i)));
		//_biasesHidden.push_back(CUDAMatrix(std::get<0>(i), 1));
		_biasesHidden.push_back(CUDAMatrix::Zero(std::get<0>(i), 1));
		_biasesHiddenBack.push_back(CUDAMatrix::Zero(std::get<0>(i), 1));
		//_biasesRecurrentHidden.push_back(CUDAMatrix(std::get<0>(i), 1));
		_biasesRecurrentHidden.push_back(CUDAMatrix::Zero(std::get<0>(i), 1));
		_biasesRecurrentHiddenBack.push_back(CUDAMatrix::Zero(std::get<0>(i), 1));

		_layerNormalizationScaleParameters.push_back(CUDAMatrix::One(std::get<0>(i), 1));
		_layerNormalizationShiftParameters.push_back(CUDAMatrix::Zero(std::get<0>(i), 1));
	}

	_momentumCoefficient = 0.9;

	_learningRate = learningRate;
	_inputNeurons = inputNeurons;

	std::vector<int> hLayerN;
	std::vector<LayerActivationFuncs> hLayerActFuncs;
	for (const auto& tup : hiddenLayerNeurons) {
		int hln = std::get<0>(tup);
		LayerActivationFuncs hlaf = std::get<1>(tup);

		hLayerN.push_back(hln);
		hLayerActFuncs.push_back(hlaf);
	}

	/*_hiddenLayerNeurons = hiddenLayerNeurons;
	_hiddenLayerNeuronsActivationFuncs = hiddenLayerNeuronsActivationFuncs;*/
	_hiddenLayerNeurons = hLayerN;
	_hiddenLayerNeuronsActivationFuncs = hLayerActFuncs;
	_outputNeurons = outputNeurons;
	_allClasses = outputNeurons;
	_totalLoss = 0; 
	_totalLossGeneral = 0;
	_modelAccuracy = 0;
	_trainingSeqLength = trainingSeqLength;


	// initialize weigths using Glorot uniform initializer

	unsigned long int randSeed = 18273;
	_mathHandler = new MathHandler(randSeed);

	double limit = 0;

	std::default_random_engine re = _mathHandler->GetRandomEngine();

	// init input weigths
	//_inputWeights.Resize(hiddenLayerNeurons, inputNeurons);
	for (int iwc = 0; iwc < hiddenLayerNeurons.size(); iwc++) {
		limit = std::sqrt(6.0 / (_inputWeights[iwc].GetRows() + _inputWeights[iwc].GetColumns()));
		std::uniform_real_distribution<double> unif(-limit, limit);
		for (int i = 0; i < _inputWeights[iwc].GetRows(); i++) {
			for (int j = 0; j < _inputWeights[iwc].GetColumns(); j++) {
				_inputWeights[iwc](i, j) = unif(re);
				_inputWeightsBack[iwc](i, j) = unif(re);
			}
		}
	}

	//_inputWeights.Print();

	// init hidden/recurrent weights
	for (int hwc = 0; hwc < hiddenLayerNeurons.size(); hwc++) {
		limit = std::sqrt(6.0 / (_hiddenWeights[hwc].GetRows() + _hiddenWeights[hwc].GetColumns()));
		std::uniform_real_distribution<double> unif(-limit, limit);
		for (int i = 0; i < _hiddenWeights[hwc].GetRows(); i++) {
			for (int j = 0; j < _hiddenWeights[hwc].GetColumns(); j++) {
				_hiddenWeights[hwc](i, j) = unif(re);
				_hiddenWeightsBack[hwc](i, j) = unif(re);
			}
		}
	}

	// init output weights
	limit = std::sqrt(6.0 / (_weightsOutput.GetRows() + _weightsOutput.GetColumns()));
	std::uniform_real_distribution<double> unif(-limit, limit);
	for (int i = 0; i < _weightsOutput.GetRows(); i++) {
		for (int j = 0; j < _weightsOutput.GetColumns(); j++) {
			_weightsOutput(i, j) = unif(re);
			//_weightsOutputBack(i, j) = unif(re);
		}
	}
	
	_biasesOutput = CUDAMatrix::Zero(outputNeurons, 1);
	_biasesOutputBack = CUDAMatrix::Zero(outputNeurons, 1);

	// init velocities for momentum

	for (int i = 0; i < _inputWeights.size() + 1; i++)
	{
		if (i < _inputWeights.size()) _velocityWeightsInput.push_back(CUDAMatrix::Zero(_inputWeights[i].GetRows(), _inputWeights[i].GetColumns()));
		else _velocityWeightsInput.push_back(CUDAMatrix::Zero(_weightsOutput.GetRows(), _weightsOutput.GetColumns()));
	}

	for (int i = 0; i < _hiddenWeights.size(); i++)
	{
		_velocityWeightsHidden.push_back(CUDAMatrix::Zero(_hiddenWeights[i].GetRows(), _hiddenWeights[i].GetColumns()));
	}

	for (int i = 0; i < _biasesHidden.size() + 1; i++)
	{
		if (i < _biasesHidden.size()) _velocityBias.push_back(CUDAMatrix::Zero(_biasesHidden[i].GetRows(), _biasesHidden[i].GetColumns()));
		else _velocityBias.push_back(CUDAMatrix::Zero(_biasesOutput.GetRows(), _biasesOutput.GetColumns()));
	}

	for (int i = 0; i < _biasesRecurrentHidden.size(); i++)
	{
		_velocityRecurrentHiddenBias.push_back(CUDAMatrix::Zero(_biasesRecurrentHidden[i].GetRows(), _biasesRecurrentHidden[i].GetColumns()));
	}


	for (int i = 0; i < _inputWeightsBack.size(); i++)
	{
		_velocityWeightsInputBack.push_back(CUDAMatrix::Zero(_inputWeightsBack[i].GetRows(), _inputWeightsBack[i].GetColumns()));
	}

	for (int i = 0; i < _hiddenWeightsBack.size(); i++)
	{
		_velocityWeightsHiddenBack.push_back(CUDAMatrix::Zero(_hiddenWeightsBack[i].GetRows(), _hiddenWeightsBack[i].GetColumns()));
	}

	for (int i = 0; i < _biasesHiddenBack.size(); i++)
	{
		_velocityBiasBack.push_back(CUDAMatrix::Zero(_biasesHiddenBack[i].GetRows(), _biasesHiddenBack[i].GetColumns()));
	}

	for (int i = 0; i < _biasesRecurrentHiddenBack.size(); i++)
	{
		_velocityRecurrentHiddenBiasBack.push_back(CUDAMatrix::Zero(_biasesRecurrentHiddenBack[i].GetRows(), _biasesRecurrentHiddenBack[i].GetColumns()));
	}

	CopyParameters();
}

double UserSimulator::EvaluateOnValidateSet(int lossType) {
	// evaluate based on cross entropy
	_totalLoss = 0;
	_totalLossGeneral = 0;
	double trainingBatchSize = _batchSize;
	_batchSize = 1024;

	std::vector<CUDAMatrix> clickSequence;
	int progress = 1;
	
	printf("testing on validation set...\n");

	int testExamplesCount = _validationSet.size();
	int testBatchSize = _batchSize;

	for (int k = 0; k < testExamplesCount; k += _batchSize) {
		std::vector<std::vector<int>>::const_iterator first = _validationSet.begin() + k;
		std::vector<std::vector<int>>::const_iterator last = _validationSet.begin() + (k + _batchSize < testExamplesCount ? k + _batchSize : testExamplesCount);
		std::vector<std::vector<int>> newVec(first, last);
		SetBatchSize(newVec.size());
		std::vector<CUDAMatrix> oneHotEncodedInput = std::get<0>(_mathHandler->CreateBatchOneHotEncodedVector(newVec, _weightsForClasses, _allClasses, _batchSize));

		if (k > 0 && k / (int)(testExamplesCount * 0.1) >= progress && progress <= 10) {
			std::cout << (k / (int)(testExamplesCount * 0.1)) * 10 << "%" << std::endl;
			progress++;
		}
		//printf("iteration: %d / %d\n", k, trainingExamplesCount);

		PredictNextClickFromSequence(oneHotEncodedInput, false, false, false, true, 20);
	}

	//_totalLoss /= _trainingSeqLength;
	//_totalLossGeneral /= _trainingSeqLength;

	_batchSize = trainingBatchSize;
	printf("testing on validation set finished total weighted cross entropy loss: %f, general loss: %f...\n", _totalLoss, _totalLossGeneral);
	return lossType == 0 ? _totalLossGeneral : _totalLoss;
}


void UserSimulator::ForwardPropGated(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode, bool trainMode) {
	CUDAMatrix updateGate;
	CUDAMatrix resetGate;
	CUDAMatrix newMemContent;
	CUDAMatrix hiddenOutput;
	std::vector<CUDAMatrix> allHiddenLayers;
	std::vector<CUDAMatrix> allresetGateValues;
	std::vector<CUDAMatrix> allUpdateGateValues;
	std::vector<CUDAMatrix> allCandidateActivationValues;
	for (int l = 0; l < _hiddenWeights.size(); l++) {
		if (l == 0) {
			updateGate = _updateGateInput[l] * onehotEncodedInput + _updateGateBias[l].Vec();
			if (sequencePosition > 0) updateGate += _updateGateHidden[l] * _hiddenStepValues[sequencePosition - 1][l];
			updateGate = updateGate.sigmoid();

			resetGate = _resetGateInput[l] * onehotEncodedInput + _resetGateBias[l].Vec();
			if (sequencePosition > 0) resetGate += _resetGateHidden[l] * _hiddenStepValues[sequencePosition - 1][l];
			resetGate = resetGate.sigmoid();

			newMemContent = _candidateActivationInput[l] * onehotEncodedInput + _candidateActivationBias[l].Vec();
			if (sequencePosition > 0) newMemContent += _candidateActivationHidden[l] * (resetGate.Array() * _hiddenStepValues[sequencePosition - 1][l].Array());
			newMemContent = newMemContent.tanh();

			hiddenOutput = updateGate.Array() * newMemContent.Array();
			if (sequencePosition > 0) hiddenOutput += (CUDAMatrix::One(updateGate.GetRows(), updateGate.GetColumns()) - updateGate).Array() * _hiddenStepValues[sequencePosition - 1][l];
		}
		else {
			updateGate = _updateGateInput[l] * _hiddenStepValues[sequencePosition][l - 1] + _updateGateBias[l].Vec();
			if (sequencePosition > 0) updateGate += _updateGateHidden[l] * _hiddenStepValues[sequencePosition - 1][l];
			updateGate = updateGate.sigmoid();

			resetGate = _resetGateInput[l] * _hiddenStepValues[sequencePosition][l - 1] + _resetGateBias[l].Vec();
			if (sequencePosition > 0) resetGate += _resetGateHidden[l] * _hiddenStepValues[sequencePosition - 1][l];
			resetGate = resetGate.sigmoid();

			newMemContent = _candidateActivationInput[l] * _hiddenStepValues[sequencePosition][l - 1] + _candidateActivationBias[l].Vec();
			if (sequencePosition > 0) newMemContent += _candidateActivationHidden[l] * (resetGate.Array() * _hiddenStepValues[sequencePosition - 1][l].Array());
			newMemContent = newMemContent.tanh();

			hiddenOutput = updateGate.Array() * newMemContent.Array();
			if (sequencePosition > 0) hiddenOutput += (CUDAMatrix::One(updateGate.GetRows(), updateGate.GetColumns()) - updateGate).Array() * _hiddenStepValues[sequencePosition - 1][l];
		}

		allHiddenLayers.push_back(hiddenOutput);
		allresetGateValues.push_back(resetGate);
		allUpdateGateValues.push_back(updateGate);
		allCandidateActivationValues.push_back(newMemContent);

	}

	_hiddenStepValues.push_back(allHiddenLayers);
	_resetGateValues.push_back(allresetGateValues);
	_updateGateValues.push_back(allUpdateGateValues);
	_candidateActivationValues.push_back(allCandidateActivationValues);

	// compute output

	CUDAMatrix outputValuesUnactivated = _weightsOutput * hiddenOutput + _biasesOutput.Vec();

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


	outputValuesActivated = outputValuesActivated.Array() / sumsForSoftmax.Array();

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

void UserSimulator::BackPropGated(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode) {

	// Initialize gradients for the output layer
	CUDAMatrix outputWeightsGrad = CUDAMatrix::Zero(_weightsOutput.GetRows(), _weightsOutput.GetColumns());
	CUDAMatrix outputBiasGrad = CUDAMatrix::Zero(_outputNeurons, this->_batchSize);

	// Initialize gradients for GRU gate parameters
	std::deque<CUDAMatrix> updateGateInputGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> updateGateHiddenGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> resetGateInputGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> resetGateHiddenGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> candidateActivationInputGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> candidateActivationHiddenGrad(_hiddenWeights.size());

	std::deque<CUDAMatrix> updateGateBiasGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> resetGateBiasGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> candidateActivationBiasGrad(_hiddenWeights.size());

	CUDAMatrix gradientUpdateGate;
	CUDAMatrix gradientResetGate;
	CUDAMatrix gradientCandidateActivation;

	LayerActivationFuncs sigAF = sigAct;
	LayerActivationFuncs tanhAF = tanhAct;

	for (int i = 0; i < _hiddenWeights.size(); i++) {
		updateGateInputGrad[i] = CUDAMatrix::Zero(_updateGateInput[i].GetRows(), _updateGateInput[i].GetColumns());
		updateGateHiddenGrad[i] = CUDAMatrix::Zero(_updateGateHidden[i].GetRows(), _updateGateHidden[i].GetColumns());
		resetGateInputGrad[i] = CUDAMatrix::Zero(_resetGateInput[i].GetRows(), _resetGateInput[i].GetColumns());
		resetGateHiddenGrad[i] = CUDAMatrix::Zero(_resetGateHidden[i].GetRows(), _resetGateHidden[i].GetColumns());
		candidateActivationInputGrad[i] = CUDAMatrix::Zero(_candidateActivationInput[i].GetRows(), _candidateActivationInput[i].GetColumns());
		candidateActivationHiddenGrad[i] = CUDAMatrix::Zero(_candidateActivationHidden[i].GetRows(), _candidateActivationHidden[i].GetColumns());

		updateGateBiasGrad[i] = CUDAMatrix::Zero(_updateGateBias[i].GetRows(), _batchSize);
		resetGateBiasGrad[i] = CUDAMatrix::Zero(_resetGateBias[i].GetRows(), _batchSize);
		candidateActivationBiasGrad[i] = CUDAMatrix::Zero(_candidateActivationBias[i].GetRows(), _batchSize);
	}

	// Initialize hidden gradients for backpropagation through time
	std::deque<CUDAMatrix> nextHiddenGrad(_hiddenWeights.size(), CUDAMatrix::Zero(_hiddenLayerNeurons[0], this->_batchSize));

	// Iterate over each timestep from last to first
	for (int t = _outputValues.size() - 1; t >= 0; t--) {
		// Cross-entropy loss gradient w.r.t. softmax input
		CUDAMatrix outputGrad = _outputValues[t] - oneHotEncodedLabels[t + 1]; // Gradient of softmax + cross-entropy

		// Gradients for the output layer
		outputWeightsGrad += outputGrad * _mathHandler->TransposeMatrix(_hiddenStepValues[t].back());
		outputBiasGrad += outputGrad;

		// Backpropagate into the hidden layers
		CUDAMatrix nextLayerGradient;
		CUDAMatrix hiddenGrad;

		for (int layer = _hiddenWeights.size() - 1; layer >= 0; layer--) {
			// Gradients for update gate, reset gate, and candidate activation

			if (layer == _hiddenWeights.size() - 1) {
				hiddenGrad = (_weightsOutput.transpose() * outputGrad) + (nextHiddenGrad[layer].Array() * _updateGateValues[t][layer].Array());
				/*if (t < _outputValues.size() - 1) {
					hiddenGrad += nextHiddenGrad[layer].Array() * _updateGateValues[t][layer].Array();
				}*/
			}
			else {
				hiddenGrad = (((_updateGateInput[layer + 1].transpose() * nextHiddenGrad[layer + 1]).Array() * _mathHandler->ActFuncDerivative(_updateGateValues[t][layer + 1], sigAF).Array())
					+ (nextHiddenGrad[layer].Array() * _updateGateValues[t][layer]) + (nextHiddenGrad[layer].Array() * (CUDAMatrix::One(_updateGateValues[t][layer].GetRows(), _updateGateValues[t][layer].GetColumns()) - _updateGateValues[t][layer]).Array()
						* _mathHandler->ActFuncDerivative(_candidateActivationValues[t][layer], tanhAF).Array()));
			}

			// gradient wrt to gates
			CUDAMatrix gradientUpdateGatePrevHiddenState = t == 0 ? CUDAMatrix::Zero(_candidateActivationValues[t][layer].GetRows(), _candidateActivationValues[t][layer].GetColumns()) : _hiddenStepValues[t - 1][layer];
			//gradientUpdateGate = hiddenGrad.Array() * (_candidateActivationValues[t][layer] - _hiddenStepValues[t - 1][layer]).Array() * _mathHandler->ActFuncDerivative(_updateGateValues[t][layer], sigAF).Array();
			gradientUpdateGate = hiddenGrad.Array() * (_candidateActivationValues[t][layer] - gradientUpdateGatePrevHiddenState).Array() * _mathHandler->ActFuncDerivative(_updateGateValues[t][layer], sigAF).Array();
			
			CUDAMatrix gatesTanhDerivativeInput = layer == 0 ? oneHotEncodedLabels[t] : _hiddenStepValues[t][layer - 1];
			CUDAMatrix gatesTanhDerivative = _mathHandler->ActFuncDerivative((_candidateActivationInput[layer] * gatesTanhDerivativeInput)
				+ (_candidateActivationHidden[layer] * (_resetGateValues[t][layer].Array() * gradientUpdateGatePrevHiddenState.Array())) + _candidateActivationBias[layer].Vec(), tanhAF);
			
			gradientResetGate = hiddenGrad.Array() * (_candidateActivationHidden[layer].Array() * _candidateActivationValues[t][layer].Array()).Array() * gatesTanhDerivative.Array()
				* _mathHandler->ActFuncDerivative(_resetGateValues[t][layer], sigAF).Array();

			gradientCandidateActivation = hiddenGrad.Array() * _updateGateValues[t][layer].Array() * gatesTanhDerivative.Array();


			// gradients wrt parameters

			if (layer == 0) {
				//updateGateInputGrad[layer] += gradientUpdateGate.Array() * oneHotEncodedLabels[t].transpose().Array();
				updateGateInputGrad[layer] += gradientUpdateGate * oneHotEncodedLabels[t].transpose();
				resetGateInputGrad[layer] += gradientResetGate * oneHotEncodedLabels[t].transpose();
				candidateActivationInputGrad[layer] += gradientCandidateActivation * oneHotEncodedLabels[t].transpose();

			}
			else {
				updateGateInputGrad[layer] += gradientUpdateGate * _hiddenStepValues[t][layer - 1].transpose();
				resetGateInputGrad[layer] += gradientResetGate * _hiddenStepValues[t][layer - 1].transpose();
				candidateActivationInputGrad[layer] += gradientCandidateActivation * _hiddenStepValues[t][layer - 1].transpose();

			}

			if (t > 0) {
				updateGateHiddenGrad[layer] += gradientUpdateGate * _hiddenStepValues[t - 1][layer].transpose();
				resetGateHiddenGrad[layer] += gradientResetGate * _hiddenStepValues[t - 1][layer].transpose();
				//candidateActivationHiddenGrad[layer] += gradientCandidateActivation * _hiddenStepValues[t - 1][layer].transpose();
			}

			updateGateBiasGrad[layer] += gradientUpdateGate.Vec();
			resetGateBiasGrad[layer] += gradientResetGate.Vec();
			candidateActivationBiasGrad[layer] += gradientCandidateActivation.Vec();

			// Update nextHiddenGrad for the next iteration of layer and timestep
			nextHiddenGrad[layer] = hiddenGrad;
			nextLayerGradient = hiddenGrad;
		}
	}

	double adjustedLearningRate = learningRate * std::sqrt(_batchSize);

	// Apply gradients to update weights and biases
	for (int i = 0; i < _hiddenWeights.size(); i++) {
		_updateGateInput[i] -= updateGateInputGrad[i] * (1.0 / _batchSize) * adjustedLearningRate;
		_updateGateHidden[i] -= updateGateHiddenGrad[i] * (1.0 / _batchSize) * adjustedLearningRate;
		_updateGateBias[i] -= updateGateBiasGrad[i].RowAverage() * adjustedLearningRate;

		_resetGateInput[i] -= resetGateInputGrad[i] * (1.0 / _batchSize) * adjustedLearningRate;
		_resetGateHidden[i] -= resetGateHiddenGrad[i] * (1.0 / _batchSize) * adjustedLearningRate;
		_resetGateBias[i] -= resetGateBiasGrad[i].RowAverage() * adjustedLearningRate;

		_candidateActivationInput[i] -= candidateActivationInputGrad[i] * (1.0 / _batchSize) * adjustedLearningRate;
		_candidateActivationHidden[i] -= candidateActivationHiddenGrad[i] * (1.0 / _batchSize) * adjustedLearningRate;
		_candidateActivationBias[i] -= candidateActivationBiasGrad[i].RowAverage() * adjustedLearningRate;
	}

	// Update output layer weights and biases
	_weightsOutput -= outputWeightsGrad * (1.0 / _batchSize) * adjustedLearningRate;
	_biasesOutput -= outputBiasGrad.RowAverage() * adjustedLearningRate;

	if (verboseMode) {
		std::cout << "Backpropagation Completed. Weights and biases updated." << std::endl;
	}
}

void UserSimulator::ForwardProp(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode, bool trainMode, bool validationMode, CUDAMatrix* nextAction) {
	std::vector<CUDAMatrix> allHiddenLayers;
	CUDAMatrix XI;
	CUDAMatrix XHCurrentTimeStep;
	CUDAMatrix activatedHiddenLayer;
	for (int l = 0; l < _hiddenWeights.size(); l++) {
		if (l == 0)
		{
			XI = _inputWeights[0] * onehotEncodedInput;
		}
		else {
			XI = _inputWeights[l].transpose() * allHiddenLayers[allHiddenLayers.size() - 1];
		}
		// if first element in sequence XHidden at previous time step non existent, just take XI(nput)
		if (sequencePosition == 0) {
			XHCurrentTimeStep = XI;
		}
		else {
			//XHCurrentTimeStep = XI + (_hiddenWeights[l] * _hiddenStepValues[sequencePosition - 1][l]);

			//seperate bias for recurrent connection
			XHCurrentTimeStep = XI + (_hiddenWeights[l] * _hiddenStepValues[sequencePosition - 1][l]) + _biasesRecurrentHidden[l].Vec();
		}

		XHCurrentTimeStep += _biasesHidden[l].Vec();

		// todo layer normalization.....

		activatedHiddenLayer = XHCurrentTimeStep.Activate(_hiddenLayerNeuronsActivationFuncs[l]);
		allHiddenLayers.push_back(activatedHiddenLayer);
	}

	_hiddenStepValues.push_back(allHiddenLayers);

	// compute output

	CUDAMatrix outputValuesUnactivated = _weightsOutput * activatedHiddenLayer + _biasesOutput.Vec();

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

	outputValuesActivated = outputValuesActivated.Array() / sumsForSoftmax.Array();

	_outputValues.push_back(outputValuesActivated);

	if (verboseMode)
	{
		std::cout << "_outputValuesActivated" << std::endl;
		outputValuesActivated.Print();
	}

	/*if (validationMode)
	{
		CUDAMatrix outputValuesMaskedWithNextActions = outputValuesActivated * (*nextAction).Array();
		bool predictionFound = false;

		for (int j = 0; j < outputValuesMaskedWithNextActions.GetColumns(); j++) {
			predictionFound = false;
			for (int i = 0; i < outputValuesMaskedWithNextActions.GetRows(); i++) {
				if (outputValuesMaskedWithNextActions(i, j) > 0) {
					_totalLoss -= std::log(outputValuesMaskedWithNextActions(i, j));
					predictionFound = true;
					continue;
				}
			}
			if (!predictionFound) {
				_totalLoss = DBL_MAX;
				break;
			}
		}

		if (_totalLoss != DBL_MAX) _totalLoss /= onehotEncodedInput.GetColumns();
	}*/
}

void UserSimulator::ForwardPropBiRNN(CUDAMatrix onehotEncodedInput, int sequencePosition, bool verboseMode, bool trainMode, bool forwardDirection, int fullSeqLength) {
	std::vector<CUDAMatrix> allHiddenLayers;
	CUDAMatrix XI;
	CUDAMatrix XHCurrentTimeStep;
	CUDAMatrix activatedHiddenLayer;
	double totalLossGeneralTimestep = 0;
	double totalLossTimestep = 0;
	for (int l = 0; l < _hiddenWeights.size(); l++) {
		if (l == 0)
		{
			XI = (forwardDirection ? _inputWeights[0] : _inputWeightsBack[0]) * onehotEncodedInput;
		}
		else {
			if (trainMode && _useDropout) XI = (forwardDirection ? _inputWeights[l].transpose() : _inputWeightsBack[l].transpose()) * (allHiddenLayers[allHiddenLayers.size() - 1] * _variationalDropoutMasks[l - 1].Array());
			else XI = (forwardDirection ? _inputWeights[l].transpose() : _inputWeightsBack[l].transpose()) * allHiddenLayers[allHiddenLayers.size() - 1];
		}
		// if first element in sequence XHidden at previous time step non existent, just take XI(nput)
		if (forwardDirection)
		{
			if (sequencePosition == 0) {
				XHCurrentTimeStep = XI;
			}
			else {
				//XHCurrentTimeStep = XI + (_hiddenWeights[l] * _hiddenStepValues[sequencePosition - 1][l]);

				//seperate bias for recurrent connection
				if (trainMode && _useRecurrentDropout) XHCurrentTimeStep = XI + (_hiddenWeights[l] * (_hiddenStepValues[sequencePosition - 1][l] * _variationalDropoutMasks[l].Array())) + _biasesRecurrentHidden[l].Vec();
				else XHCurrentTimeStep = XI + (_hiddenWeights[l] * _hiddenStepValues[sequencePosition - 1][l]) + _biasesRecurrentHidden[l].Vec();
			}
		}
		else {
			if (sequencePosition == fullSeqLength - 1) {
				XHCurrentTimeStep = XI;
			}
			else {
				//XHCurrentTimeStep = XI + (_hiddenWeights[l] * _hiddenStepValues[sequencePosition - 1][l]);

				//seperate bias for recurrent connection
				if (trainMode && _useRecurrentDropout) XHCurrentTimeStep = XI + (_hiddenWeightsBack[l] * (_hiddenStepValuesBack.front()[l] * _variationalDropoutMasks[l].Array())) + _biasesRecurrentHiddenBack[l].Vec();
				else XHCurrentTimeStep = XI + (_hiddenWeightsBack[l] * _hiddenStepValuesBack.front()[l]) + _biasesRecurrentHiddenBack[l].Vec();
			}
		}

		XHCurrentTimeStep += (forwardDirection ? _biasesHidden[l].Vec() : _biasesHiddenBack[l].Vec());

		//CUDAMatrix normalizedLayer = XHCurrentTimeStep;

		//CUDAMatrix meanBatch = CUDAMatrix::Zero(normalizedLayer.GetRows(), normalizedLayer.GetColumns());
		//for (int i = 0; i < normalizedLayer.GetRows(); i++) {
		//	for (int j = 0; j < normalizedLayer.GetColumns(); j++) {
		//		meanBatch(0, j) += normalizedLayer(i, j);
		//	}
		//}

		//for (int i = 0; i < normalizedLayer.GetRows(); i++) {
		//	for (int j = 0; j < normalizedLayer.GetColumns(); j++) {
		//		meanBatch(i, j) = meanBatch(0, j);
		//	}
		//}

		//meanBatch = meanBatch / normalizedLayer.GetRows();

		//CUDAMatrix varianceBatch = CUDAMatrix::Zero(normalizedLayer.GetRows(), normalizedLayer.GetColumns());

		//for (int i = 0; i < normalizedLayer.GetRows(); i++) {
		//	for (int j = 0; j < normalizedLayer.GetColumns(); j++) {
		//		varianceBatch(0, j) += std::pow(normalizedLayer(i, j) - meanBatch(0, j), 2);
		//	}
		//}

		//for (int i = 0; i < normalizedLayer.GetRows(); i++) {
		//	for (int j = 0; j < normalizedLayer.GetColumns(); j++) {
		//		varianceBatch(i, j) = varianceBatch(0, j);
		//	}
		//}

		//varianceBatch = varianceBatch / normalizedLayer.GetRows();

		//varianceBatch = varianceBatch.sqrt();
		//varianceBatch = varianceBatch + std::sqrt(0.000001);

		//// scale and shift


		//CUDAMatrix layerNormalizationScaleParametersBatch(_layerNormalizationScaleParameters[l].GetRows(), XHCurrentTimeStep.GetColumns());
		//for (int i = 0; i < layerNormalizationScaleParametersBatch.GetRows(); i++) {
		//	for (int j = 0; j < layerNormalizationScaleParametersBatch.GetColumns(); j++) {
		//		layerNormalizationScaleParametersBatch(i, j) = _layerNormalizationScaleParameters[l](i, 0);
		//	}
		//}

		//normalizedLayer = (layerNormalizationScaleParametersBatch / varianceBatch.Array()) * (XHCurrentTimeStep - meanBatch).Array() + _layerNormalizationShiftParameters[l].Vec();
		//activatedHiddenLayer = normalizedLayer.Activate(_hiddenLayerNeuronsActivationFuncs[l]);

		activatedHiddenLayer = XHCurrentTimeStep.Activate(_hiddenLayerNeuronsActivationFuncs[l]);
		allHiddenLayers.push_back(activatedHiddenLayer);
	}

	if (forwardDirection) _hiddenStepValues.push_back(allHiddenLayers);
	else _hiddenStepValuesBack.insert(_hiddenStepValuesBack.begin(), allHiddenLayers);
}

void UserSimulator::BackProp(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode) {
	if (verboseMode) std::cout << "Back prop" << std::endl;

	// Initialize gradients
	CUDAMatrix outputWeightsGrad = CUDAMatrix::Zero(_weightsOutput.GetRows(), _weightsOutput.GetColumns());
	CUDAMatrix outputBiasGrad = CUDAMatrix::Zero(_outputNeurons, this->_batchSize);
	std::deque<CUDAMatrix> hiddenWeightsGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> hiddenBiasGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> hiddenRecurrentBiasGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> inputWeightsGrad(_inputWeights.size());
	std::deque<CUDAMatrix> nextHiddenGrad(_hiddenWeights.size());

	for (int i = 0; i < _hiddenWeights.size(); i++) {
		hiddenWeightsGrad[i] = CUDAMatrix::Zero(_hiddenWeights[i].GetRows(), _hiddenWeights[i].GetColumns());
		hiddenBiasGrad[i] = CUDAMatrix::Zero(_hiddenLayerNeurons[i], this->_batchSize);
		hiddenRecurrentBiasGrad[i] = CUDAMatrix::Zero(_hiddenLayerNeurons[i], this->_batchSize);
		nextHiddenGrad[i] = CUDAMatrix::Zero(_hiddenLayerNeurons[i], this->_batchSize);
	}

	for (int i = 0; i < _inputWeights.size(); i++) {
		inputWeightsGrad[i] = CUDAMatrix::Zero(_inputWeights[i].GetRows(), _inputWeights[i].GetColumns());
	}

	// Iterate over each timestep from last to first
	for (int t = _outputValues.size() - 1; t >= 0; t--) {
		// Cross-entropy loss gradient w.r.t. softmax input
		CUDAMatrix outputGrad = _outputValues[t] - oneHotEncodedLabels[t + 1]; // Gradient of softmax + cross-entropy
		outputGrad = outputGrad * _weightsForClasses.Vec();

		// Gradients for the output layer
		outputWeightsGrad += outputGrad * _mathHandler->TransposeMatrix(_hiddenStepValues[t].back());
		outputBiasGrad += outputGrad;

		// Backpropagate into the hidden layers
		//CUDAMatrix nextLayerGradient;
		CUDAMatrix hiddenGrad;
		for (int layer = _hiddenWeights.size() - 1; layer >= 0; layer--) {
			if (layer == _hiddenWeights.size() - 1) {
				hiddenGrad = (_weightsOutput.transpose() * outputGrad) * _mathHandler->ActFuncDerivative(_hiddenStepValues[t][layer], _hiddenLayerNeuronsActivationFuncs[layer]).Array();
			}
			else {
				//hiddenGrad = _inputWeights[layer + 1] * nextLayerGradient + _hiddenWeights[layer].transpose() * nextHiddenGrad[layer];
				hiddenGrad = _inputWeights[layer + 1] * nextHiddenGrad[layer + 1] + _hiddenWeights[layer].transpose() * nextHiddenGrad[layer];

				hiddenGrad = hiddenGrad * _mathHandler->ActFuncDerivative(_hiddenStepValues[t][layer], _hiddenLayerNeuronsActivationFuncs[layer]).Array();
			}

			// Accumulate gradients for hidden layer weights and biases
			if (t > 0) {
				hiddenWeightsGrad[layer] += hiddenGrad * _hiddenStepValues[t - 1][layer].transpose();
				hiddenRecurrentBiasGrad[layer] += hiddenGrad;
			}
			hiddenBiasGrad[layer] += hiddenGrad;

			// Accumulate gradients for input weights
			if (layer == 0) {
				inputWeightsGrad[layer] += hiddenGrad * _oneHotEncodedClicks[t].transpose();
			}
			else if (layer < _hiddenWeights.size() - 1) {
				inputWeightsGrad[layer] += (hiddenGrad * _hiddenStepValues[t][layer - 1].transpose()).transpose();
			}
			// Update nextHiddenGrad for the next iteration of layer and timestep
			nextHiddenGrad[layer] = hiddenGrad;
			//nextLayerGradient = hiddenGrad;
		}
	}

	// Apply learning rate adjustment
	// smaller batches square root scaling, larger batches linear scaling
	double adjustedLearningRate = learningRate;
	if (_batchSize <= 256) adjustedLearningRate *= std::sqrt(_batchSize);
	else adjustedLearningRate *= _batchSize;

	// linear adjustment
	//double adjustedLearningRate = learningRate * _batchSize;

	//normalize gradients by minibatch size
	for (int i = 0; i < inputWeightsGrad.size(); i++) inputWeightsGrad[i] = inputWeightsGrad[i] * (1.0 / _batchSize);
	for (int i = 0; i < hiddenWeightsGrad.size(); i++) hiddenWeightsGrad[i] = hiddenWeightsGrad[i] * (1.0 / _batchSize);
	for (int i = 0; i < hiddenBiasGrad.size(); i++) hiddenBiasGrad[i] = hiddenBiasGrad[i].RowAverage();
	for (int i = 0; i < hiddenRecurrentBiasGrad.size(); i++) hiddenRecurrentBiasGrad[i] = hiddenRecurrentBiasGrad[i].RowAverage();
	outputWeightsGrad = outputWeightsGrad * (1.0 / _batchSize);
	outputBiasGrad = outputBiasGrad.RowAverage();

	// clip gradients
	for (int i = 0; i < inputWeightsGrad.size(); i++) inputWeightsGrad[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	for (int i = 0; i < hiddenWeightsGrad.size(); i++) hiddenWeightsGrad[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	for (int i = 0; i < hiddenBiasGrad.size(); i++) hiddenBiasGrad[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	for (int i = 0; i < hiddenRecurrentBiasGrad.size(); i++) hiddenRecurrentBiasGrad[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	outputWeightsGrad.ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	outputBiasGrad.ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);

	// update model parameters

	for (int i = 0; i < _inputWeights.size(); i++) {
		_velocityWeightsInput[i] = (_velocityWeightsInput[i] * _momentumCoefficient) + (inputWeightsGrad[i] * (1 - _momentumCoefficient));
		_inputWeights[i] -= _velocityWeightsInput[i] * adjustedLearningRate;
	}

	for (int i = 0; i < _hiddenWeights.size(); i++) {
		_velocityWeightsHidden[i] = (_velocityWeightsHidden[i] * _momentumCoefficient) + (hiddenWeightsGrad[i] * (1 - _momentumCoefficient));
		_velocityBias[i] = (_velocityBias[i] * _momentumCoefficient) + (hiddenBiasGrad[i] * (1 - _momentumCoefficient));
		_velocityRecurrentHiddenBias[i] = (_velocityRecurrentHiddenBias[i] * _momentumCoefficient) + (hiddenRecurrentBiasGrad[i] * (1 - _momentumCoefficient));

		_hiddenWeights[i] -= _velocityWeightsHidden[i] * adjustedLearningRate;
		_biasesHidden[i] -= _velocityBias[i] * adjustedLearningRate;
		_biasesRecurrentHidden[i] -= _velocityRecurrentHiddenBias[i] * adjustedLearningRate;
	}

	_velocityWeightsInput.back() = (_velocityWeightsInput.back() * _momentumCoefficient) + (outputWeightsGrad * (1 - _momentumCoefficient));
	_velocityBias.back() = (_velocityBias.back() * _momentumCoefficient) + (outputBiasGrad * (1 - _momentumCoefficient));

	_weightsOutput -= _velocityWeightsInput.back() * adjustedLearningRate;
	_biasesOutput -= _velocityBias.back() * adjustedLearningRate;
}

void UserSimulator::BackPropBiRNN(std::vector<CUDAMatrix> oneHotEncodedLabels, double learningRate, bool verboseMode) {
	if (verboseMode) std::cout << "Back prop" << std::endl;

	// Initialize gradients
	CUDAMatrix outputWeightsGrad = CUDAMatrix::Zero(_weightsOutput.GetRows(), _weightsOutput.GetColumns());
	CUDAMatrix outputBiasGrad = CUDAMatrix::Zero(_outputNeurons, this->_batchSize);
	std::deque<CUDAMatrix> hiddenWeightsGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> hiddenWeightsGradBack(_hiddenWeights.size());
	std::deque<CUDAMatrix> hiddenBiasGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> hiddenBiasGradBack(_hiddenWeights.size());
	std::deque<CUDAMatrix> hiddenRecurrentBiasGrad(_hiddenWeights.size());
	std::deque<CUDAMatrix> hiddenRecurrentBiasGradBack(_hiddenWeights.size());
	std::deque<CUDAMatrix> inputWeightsGrad(_inputWeights.size());
	std::deque<CUDAMatrix> inputWeightsGradBack(_inputWeights.size());
	std::deque<CUDAMatrix> nextHiddenGrad(_hiddenWeights.size());

	for (int i = 0; i < _hiddenWeights.size(); i++) {
		hiddenWeightsGrad[i] = CUDAMatrix::Zero(_hiddenWeights[i].GetRows(), _hiddenWeights[i].GetColumns());
		hiddenWeightsGradBack[i] = CUDAMatrix::Zero(_hiddenWeights[i].GetRows(), _hiddenWeights[i].GetColumns());
		hiddenBiasGrad[i] = CUDAMatrix::Zero(_hiddenLayerNeurons[i], this->_batchSize);
		hiddenBiasGradBack[i] = CUDAMatrix::Zero(_hiddenLayerNeurons[i], this->_batchSize);
		hiddenRecurrentBiasGrad[i] = CUDAMatrix::Zero(_hiddenLayerNeurons[i], this->_batchSize);
		hiddenRecurrentBiasGradBack[i] = CUDAMatrix::Zero(_hiddenLayerNeurons[i], this->_batchSize);
		nextHiddenGrad[i] = CUDAMatrix::Zero(_hiddenLayerNeurons[i], this->_batchSize);
	}

	for (int i = 0; i < _inputWeights.size(); i++) {
		inputWeightsGrad[i] = CUDAMatrix::Zero(_inputWeights[i].GetRows(), _inputWeights[i].GetColumns());
		inputWeightsGradBack[i] = CUDAMatrix::Zero(_inputWeights[i].GetRows(), _inputWeights[i].GetColumns());
	}

	// hiddenGrad = (_weightsOutput.transpose() * outputGrad) * _mathHandler->ActFuncDerivative(_hiddenStepValues[t][layer], _hiddenLayerNeuronsActivationFuncs[layer]).Array();

	// Iterate over each timestep from last to first
	for (int t = _outputValuesCombined.size() - 1; t >= 0; t--) {
		// Cross-entropy loss gradient w.r.t. softmax input
		CUDAMatrix outputGrad = _outputValuesCombined[t] - oneHotEncodedLabels[t + 1]; // Gradient of softmax + cross-entropy
		if(_useWeightedLoss) outputGrad = outputGrad * _weightsForClasses.Vec();

		//CUDAMatrix outputGrad = _outputValuesCombined[t] - (oneHotEncodedLabels[t + 1] * _weightsForClasses.Vec()); // Gradient of softmax + cross-entropy

		// Gradients for the output layer
		outputWeightsGrad += outputGrad * _mathHandler->TransposeMatrix(_hiddenStepValuesCombined[t]);
		outputBiasGrad += outputGrad;

		// Backpropagate into the hidden layers

		CUDAMatrix concatenatedHiddenGrad = _weightsOutput.transpose() * outputGrad * _mathHandler->ActFuncDerivative(_hiddenStepValuesCombined.back(), _hiddenLayerNeuronsActivationFuncs.back()).Array();

		CUDAMatrix hiddenGradForward(concatenatedHiddenGrad.GetRows() / 2, _batchSize);
		CUDAMatrix hiddenGradBackward(concatenatedHiddenGrad.GetRows() / 2, _batchSize);

		// split into forward and backward hidden grad
		for (int i = 0; i < concatenatedHiddenGrad.GetRows(); i++) {
			for (int j = 0; j < _batchSize; j++) {
				if (i < concatenatedHiddenGrad.GetRows() / 2) hiddenGradForward(i, j) = concatenatedHiddenGrad(i, j);
				else hiddenGradBackward(i - concatenatedHiddenGrad.GetRows() / 2, j) = concatenatedHiddenGrad(i, j);
			}
		}

		// forward RNN

		//CUDAMatrix hiddenGrad;
		for (int layer = _hiddenWeights.size() - 1; layer >= 0; layer--) {
			if (layer < _hiddenWeights.size() - 1) {
				//hiddenGradForward = (_weightsOutput.transpose() * outputGrad) * _mathHandler->ActFuncDerivative(_hiddenStepValues[t][layer], _hiddenLayerNeuronsActivationFuncs[layer]).Array();

				hiddenGradForward = _inputWeights[layer + 1] * nextHiddenGrad[layer + 1] + _hiddenWeights[layer].transpose() * nextHiddenGrad[layer];
				hiddenGradForward = hiddenGradForward * _mathHandler->ActFuncDerivative(_hiddenStepValues[t][layer], _hiddenLayerNeuronsActivationFuncs[layer]).Array();
			}
			//else {
			//	//hiddenGrad = _inputWeights[layer + 1] * nextLayerGradient + _hiddenWeights[layer].transpose() * nextHiddenGrad[layer];
			//	hiddenGradForward = _inputWeights[layer + 1] * nextHiddenGrad[layer + 1] + _hiddenWeights[layer].transpose() * nextHiddenGrad[layer];

			//	hiddenGradForward = hiddenGradForward * _mathHandler->ActFuncDerivative(_hiddenStepValues[t][layer], _hiddenLayerNeuronsActivationFuncs[layer]).Array();
			//}

			// Accumulate gradients for hidden layer weights and biases
			if (t > 0) {
				hiddenWeightsGrad[layer] += hiddenGradForward * _hiddenStepValues[t - 1][layer].transpose();
				hiddenRecurrentBiasGrad[layer] += hiddenGradForward;
			}
			hiddenBiasGrad[layer] += hiddenGradForward;

			// Accumulate gradients for input weights
			if (layer == 0) {
				inputWeightsGrad[layer] += hiddenGradForward * _oneHotEncodedClicks[t].transpose();
			}
			else if (layer < _hiddenWeights.size() - 1) {
				inputWeightsGrad[layer] += (hiddenGradForward * _hiddenStepValues[t][layer - 1].transpose()).transpose();
			}
			// Update nextHiddenGrad for the next iteration of layer and timestep
			nextHiddenGrad[layer] = hiddenGradForward;
		}
	}


	for (int i = 0; i < _hiddenWeights.size(); i++) {
		nextHiddenGrad[i] = CUDAMatrix::Zero(_hiddenLayerNeurons[i], this->_batchSize);
	}


	for (int t = 0; t < _outputValuesCombined.size() - 1; t++) {
		// Cross-entropy loss gradient w.r.t. softmax input
		CUDAMatrix outputGrad = _outputValuesCombined[t] - oneHotEncodedLabels[t + 1]; // Gradient of softmax + cross-entropy
		if (_useWeightedLoss) outputGrad = outputGrad * _weightsForClasses.Vec();

		//CUDAMatrix outputGrad = _outputValuesCombined[t] - (oneHotEncodedLabels[t + 1] * _weightsForClasses.Vec()); // Gradient of softmax + cross-entropy

		// Gradients for the output layer
		outputWeightsGrad += outputGrad * _mathHandler->TransposeMatrix(_hiddenStepValuesCombined[t]);
		outputBiasGrad += outputGrad;

		// Backpropagate into the hidden layers

		CUDAMatrix concatenatedHiddenGrad = _weightsOutput.transpose() * outputGrad * _mathHandler->ActFuncDerivative(_hiddenStepValuesCombined.back(), _hiddenLayerNeuronsActivationFuncs.back()).Array();
		//CUDAMatrix concatenatedHiddenGrad = _weightsOutput.transpose() * outputGrad;

		CUDAMatrix hiddenGradForward(concatenatedHiddenGrad.GetRows() / 2, _batchSize);
		CUDAMatrix hiddenGradBackward(concatenatedHiddenGrad.GetRows() / 2, _batchSize);

		// split into forward and backward hidden grad
		for (int i = 0; i < concatenatedHiddenGrad.GetRows(); i++) {
			for (int j = 0; j < _batchSize; j++) {
				if (i < concatenatedHiddenGrad.GetRows() / 2) hiddenGradForward(i, j) = concatenatedHiddenGrad(i, j);
				else hiddenGradBackward(i - concatenatedHiddenGrad.GetRows() / 2, j) = concatenatedHiddenGrad(i, j);
			}
		}

		// backward RNN

		//CUDAMatrix hiddenGrad;
		for (int layer = _hiddenWeights.size() - 1; layer >= 0; layer--) {
			if (layer < _hiddenWeights.size() - 1) {
				//hiddenGradForward = (_weightsOutput.transpose() * outputGrad) * _mathHandler->ActFuncDerivative(_hiddenStepValues[t][layer], _hiddenLayerNeuronsActivationFuncs[layer]).Array();

				hiddenGradBackward = _inputWeightsBack[layer + 1] * nextHiddenGrad[layer + 1] + _hiddenWeightsBack[layer].transpose() * nextHiddenGrad[layer];
				hiddenGradBackward = hiddenGradBackward * _mathHandler->ActFuncDerivative(_hiddenStepValuesBack[t][layer], _hiddenLayerNeuronsActivationFuncs[layer]).Array();
			}

			// Accumulate gradients for hidden layer weights and biases
			if (t < _outputValuesCombined.size() - 1) {
				hiddenWeightsGradBack[layer] += hiddenGradBackward * _hiddenStepValuesBack[t + 1][layer].transpose();
				hiddenRecurrentBiasGradBack[layer] += hiddenGradBackward;
			}
			hiddenBiasGradBack[layer] += hiddenGradBackward;

			// Accumulate gradients for input weights
			if (layer == 0) {
				inputWeightsGradBack[layer] += hiddenGradBackward * _oneHotEncodedClicksReversed[t].transpose();
			}
			else if (layer < _hiddenWeights.size() - 1) {
				inputWeightsGradBack[layer] += (hiddenGradBackward * _hiddenStepValuesBack[t][layer - 1].transpose()).transpose();
			}
			// Update nextHiddenGrad for the next iteration of layer and timestep
			nextHiddenGrad[layer] = hiddenGradBackward;
		}
	}


	// Apply learning rate adjustment
	// smaller batches square root scaling, larger batches linear scaling
	double adjustedLearningRate = learningRate;
	if (_batchSize <= 256) adjustedLearningRate *= std::sqrt(_batchSize);
	else adjustedLearningRate *= _batchSize;

	// linear adjustment
	//double adjustedLearningRate = learningRate * _batchSize;

	//normalize gradients by minibatch size
	for (int i = 0; i < inputWeightsGrad.size(); i++) inputWeightsGrad[i] = inputWeightsGrad[i] * (1.0 / _batchSize);
	for (int i = 0; i < hiddenWeightsGrad.size(); i++) hiddenWeightsGrad[i] = hiddenWeightsGrad[i] * (1.0 / _batchSize);
	for (int i = 0; i < hiddenBiasGrad.size(); i++) hiddenBiasGrad[i] = hiddenBiasGrad[i].RowAverage();
	for (int i = 0; i < hiddenRecurrentBiasGrad.size(); i++) hiddenRecurrentBiasGrad[i] = hiddenRecurrentBiasGrad[i].RowAverage();
	outputWeightsGrad = outputWeightsGrad * (1.0 / _batchSize);
	outputBiasGrad = outputBiasGrad.RowAverage();

	for (int i = 0; i < inputWeightsGradBack.size(); i++) inputWeightsGradBack[i] = inputWeightsGradBack[i] * (1.0 / _batchSize);
	for (int i = 0; i < hiddenWeightsGradBack.size(); i++) hiddenWeightsGradBack[i] = hiddenWeightsGradBack[i] * (1.0 / _batchSize);
	for (int i = 0; i < hiddenBiasGradBack.size(); i++) hiddenBiasGradBack[i] = hiddenBiasGradBack[i].RowAverage();
	for (int i = 0; i < hiddenRecurrentBiasGradBack.size(); i++) hiddenRecurrentBiasGradBack[i] = hiddenRecurrentBiasGradBack[i].RowAverage();

	// clip gradients
	for (int i = 0; i < inputWeightsGrad.size(); i++) inputWeightsGrad[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	for (int i = 0; i < hiddenWeightsGrad.size(); i++) hiddenWeightsGrad[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	for (int i = 0; i < hiddenBiasGrad.size(); i++) hiddenBiasGrad[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	for (int i = 0; i < hiddenRecurrentBiasGrad.size(); i++) hiddenRecurrentBiasGrad[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	outputWeightsGrad.ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	outputBiasGrad.ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);

	for (int i = 0; i < inputWeightsGradBack.size(); i++) inputWeightsGradBack[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	for (int i = 0; i < hiddenWeightsGradBack.size(); i++) hiddenWeightsGradBack[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	for (int i = 0; i < hiddenBiasGradBack.size(); i++) hiddenBiasGradBack[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);
	for (int i = 0; i < hiddenRecurrentBiasGradBack.size(); i++) hiddenRecurrentBiasGradBack[i].ClipByNorm(_gradientClippingThresholdMin, _gradientClippingThresholdMax);

	// update model parameters

	for (int i = 0; i < _inputWeights.size(); i++) {
		_velocityWeightsInput[i] = (_velocityWeightsInput[i] * _momentumCoefficient) + (inputWeightsGrad[i] * (1 - _momentumCoefficient));
		_velocityWeightsInputBack[i] = (_velocityWeightsInputBack[i] * _momentumCoefficient) + (inputWeightsGradBack[i] * (1 - _momentumCoefficient));

		_inputWeights[i] -= _velocityWeightsInput[i] * adjustedLearningRate;

		_inputWeightsBack[i] -= _velocityWeightsInputBack[i] * adjustedLearningRate;
	}

	for (int i = 0; i < _hiddenWeights.size(); i++) {
		_velocityWeightsHidden[i] = (_velocityWeightsHidden[i] * _momentumCoefficient) + (hiddenWeightsGrad[i] * (1 - _momentumCoefficient));
		_velocityBias[i] = (_velocityBias[i] * _momentumCoefficient) + (hiddenBiasGrad[i] * (1 - _momentumCoefficient));
		_velocityRecurrentHiddenBias[i] = (_velocityRecurrentHiddenBias[i] * _momentumCoefficient) + (hiddenRecurrentBiasGrad[i] * (1 - _momentumCoefficient));

		_velocityWeightsHiddenBack[i] = (_velocityWeightsHiddenBack[i] * _momentumCoefficient) + (hiddenWeightsGradBack[i] * (1 - _momentumCoefficient));
		_velocityBiasBack[i] = (_velocityBiasBack[i] * _momentumCoefficient) + (hiddenBiasGradBack[i] * (1 - _momentumCoefficient));
		_velocityRecurrentHiddenBiasBack[i] = (_velocityRecurrentHiddenBiasBack[i] * _momentumCoefficient) + (hiddenRecurrentBiasGradBack[i] * (1 - _momentumCoefficient));

		_hiddenWeights[i] -= _velocityWeightsHidden[i] * adjustedLearningRate;
		_biasesHidden[i] -= _velocityBias[i] * adjustedLearningRate;
		_biasesRecurrentHidden[i] -= _velocityRecurrentHiddenBias[i] * adjustedLearningRate;

		_hiddenWeightsBack[i] -= _velocityWeightsHiddenBack[i] * adjustedLearningRate;
		_biasesHiddenBack[i] -= _velocityBiasBack[i] * adjustedLearningRate;
		_biasesRecurrentHiddenBack[i] -= _velocityRecurrentHiddenBiasBack[i] * adjustedLearningRate;
	}

	_velocityWeightsInput.back() = (_velocityWeightsInput.back() * _momentumCoefficient) + (outputWeightsGrad * (1 - _momentumCoefficient));
	_velocityBias.back() = (_velocityBias.back() * _momentumCoefficient) + (outputBiasGrad * (1 - _momentumCoefficient));

	_weightsOutput -= _velocityWeightsInput.back() * adjustedLearningRate;
	_biasesOutput -= _velocityBias.back() * adjustedLearningRate;
}

std::vector<std::vector<int>> UserSimulator::PredictAllSequencesFromSequence(std::vector<int> startingSequence, int seqLen) {
	std::vector<std::vector<int>> allSequences;

	std::vector<std::vector<CUDAMatrix>> allSequencesOHE;

	//std::vector<CUDAMatrix> initialSeqOHE;

	allSequencesOHE.push_back(_mathHandler->CreateOneHotEncodedVectorSequence(startingSequence, _allClasses));

	for (int i = 0; i < seqLen; i++) {
		std::vector<std::vector<CUDAMatrix>> allSequencesOHEToAdd;
		for (auto& seq : allSequencesOHE) {
			std::deque<std::tuple<int, double>> nextActions = PredictNextClickFromSequence(seq, false, false, false, false, 20);
			if (nextActions.size() == 1) {
				// if only one action just push back
				seq.push_back(_mathHandler->CreateOneHotEncodedVector(std::get<0>(nextActions.front()), _allClasses));
			}
			else if (nextActions.size() > 1) {
				std::vector<CUDAMatrix> newSeq = seq;
				seq.push_back(_mathHandler->CreateOneHotEncodedVector(std::get<0>(nextActions.front()), _allClasses));
				// else create all the other new sequences with all possible next clicks
				for (auto mIter = std::next(nextActions.begin()); mIter != nextActions.end(); ++mIter) {
					newSeq.push_back(_mathHandler->CreateOneHotEncodedVector(std::get<0>(*mIter), _allClasses));
					allSequencesOHEToAdd.push_back(newSeq);
					newSeq.pop_back();
				}
			}
		}
		for (const auto& sequenceToAdd : allSequencesOHEToAdd) {
			allSequencesOHE.push_back(sequenceToAdd);
		}


	}

	for (const auto& oheSeq : allSequencesOHE) {
		std::vector<int> sequenceIDs;
		for (const auto& oheSeqElement : oheSeq) {
			sequenceIDs.push_back(_mathHandler->OneHotEncodedVectorToClassID(oheSeqElement));
		}
		allSequences.push_back(sequenceIDs);
	}
	return allSequences;
}

std::deque<std::tuple<int, double>> UserSimulator::PredictNextClickFromSequence(std::vector<CUDAMatrix> onehotEncodedLabels, bool performBackProp, bool verboseMode, bool trainMode, bool validationMode, int selectNTopClasses) {

	_hiddenStepValues.clear();
	_hiddenStepValuesBack.clear();
	_outputValues.clear();
	_outputValuesCombined.clear();
	_outputValuesBack.clear();
	_oneHotEncodedClicks.clear();
	_oneHotEncodedClicksReversed.clear();
	_resetGateValues.clear();
	_updateGateValues.clear();
	_candidateActivationValues.clear();
	_hiddenStepValuesCombined.clear();
	_variationalDropoutMasks.clear();
	_oneHotEncodedClicks = onehotEncodedLabels;
	std::vector<CUDAMatrix> oheLabelsRev = onehotEncodedLabels;
	std::reverse(oheLabelsRev.begin(), oheLabelsRev.end());
	_oneHotEncodedClicksReversed = oheLabelsRev;

	std::vector<CUDAMatrix> onehotEncodedLabelsReversed = onehotEncodedLabels;
	std::reverse(onehotEncodedLabelsReversed.begin(), onehotEncodedLabelsReversed.end());

	//_totalLoss = 0;

	double lower_bound = 0;
	double upper_bound = 1;
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	std::default_random_engine re = _mathHandler->GetRandomEngine();

	int allClasses = _biasesOutput.GetRows();

	if (trainMode)
	{
		// same mask across all timesteps
		for (int i = 0; i < _hiddenLayerNeurons.size(); i++) {
			CUDAMatrix mask = CUDAMatrix::One(_hiddenLayerNeurons[i], _batchSize);
			for (int j = 0; j < mask.GetColumns(); j++) {
				for (int k = 0; k < mask.GetRows(); k++) {
					// different for every sample in batch
					double isDropout = unif(re);
					if (isDropout < _dropoutRate) {
						mask(k, j) = 0.0;
					}
					else {
						// scale other inputs to account for dropped out neurons
						mask(k, j) = mask(k, j) * (1.0 / (1.0 - _dropoutRate));
					}
				}
			}
			_variationalDropoutMasks.push_back(mask);
		}
	}


	for (int i = 0; i < onehotEncodedLabels.size() - (performBackProp || validationMode ? 1 : 0); i++) {

		if (_gatedUnits == NoGates) {
			// bidirectional compute values for forward and backward direction si
			if (_isBiRNN)
			{
				ForwardPropBiRNN(onehotEncodedLabels[i], i, verboseMode, trainMode, true, onehotEncodedLabels.size() - (performBackProp || validationMode ? 1 : 0));
				ForwardPropBiRNN(onehotEncodedLabelsReversed[i + (performBackProp || validationMode ? 1 : 0)], onehotEncodedLabels.size() - (performBackProp || validationMode ? 2 : 1) - i, verboseMode, trainMode, false, onehotEncodedLabels.size() - (performBackProp || validationMode ? 1 : 0));
			}else ForwardProp(onehotEncodedLabels[i], i, verboseMode, trainMode, validationMode, validationMode ? &onehotEncodedLabels[i + 1] : nullptr);
		}
		else if (_gatedUnits == GRU) ForwardPropGated(onehotEncodedLabels[i], i, verboseMode, trainMode);
	}

	// compute outputs by combining forward and backward RNNs

	double totalLossGeneralTimestep = 0;
	double totalLossTimestep = 0;

	for (int i = 0; i < _hiddenStepValues.size(); i++) {
		if (_isBiRNN)
		{
			//_outputValuesCombined[i] += _outputValuesBack[_outputValues.size() - i - 1];
			CUDAMatrix hiddenStateCombined(_hiddenStepValues[i].back().GetRows() * 2, _hiddenStepValues[i].back().GetColumns());
			for (int k = 0; k < _hiddenStepValues[i].back().GetColumns(); k++)
			{
				for (int j = 0; j < _hiddenStepValues[i].back().GetRows(); j++) {
					hiddenStateCombined(j, k) = _hiddenStepValues[i].back()(j, k);
					hiddenStateCombined(_hiddenStepValues[i].back().GetRows() + j, k) = _hiddenStepValuesBack[i].back()(j, k);
				}
			}

			_hiddenStepValuesCombined.push_back(hiddenStateCombined);

			// softmax

			CUDAMatrix outputValuesUnactivated = _weightsOutput * hiddenStateCombined + _biasesOutput.Vec();

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

			outputValuesActivated = outputValuesActivated.Array() / sumsForSoftmax.Array();

			_outputValuesCombined.push_back(outputValuesActivated);
		}

		if (validationMode)
		{
			// calculate cross entropy loss

			bool predictionFound = false;

			CUDAMatrix outputValuesWithNextActions = (_isBiRNN ? _outputValuesCombined[i] : _outputValues[i]) * onehotEncodedLabels[i + 1].Array();

			for (int j = 0; j < outputValuesWithNextActions.GetColumns(); j++) {
				predictionFound = false;
				for (int i = 0; i < outputValuesWithNextActions.GetRows(); i++) {
					if (outputValuesWithNextActions(i, j) > 0) {
						totalLossGeneralTimestep -= std::log(outputValuesWithNextActions(i, j));
						predictionFound = true;
						continue;
					}
				}
				if (!predictionFound) {
					totalLossGeneralTimestep = DBL_MAX;
					//break;
				}
			}

			//if (_totalLossGeneral != DBL_MAX) _totalLossGeneral /= onehotEncodedInput.GetColumns();
			if (totalLossGeneralTimestep != DBL_MAX) _totalLossGeneral += totalLossGeneralTimestep / onehotEncodedLabels[i].GetColumns();
			else _totalLossGeneral = DBL_MAX;


			predictionFound = false;

			CUDAMatrix outputValuesMaskedWithNextActions = (((_isBiRNN ? _outputValuesCombined.back() : _outputValues.back()) * onehotEncodedLabels[i + 1].Array())).log() * _weightsForClasses.Vec();

			for (int j = 0; j < outputValuesMaskedWithNextActions.GetColumns(); j++) {
				predictionFound = false;
				for (int i = 0; i < outputValuesMaskedWithNextActions.GetRows(); i++) {
					if (outputValuesMaskedWithNextActions(i, j) < 0) {
						//_totalLoss -= std::log(outputValuesMaskedWithNextActions(i, j));
						totalLossTimestep -= outputValuesMaskedWithNextActions(i, j);
						predictionFound = true;
						continue;
					}
				}
				if (!predictionFound) {
					totalLossTimestep = DBL_MAX;
					//break;
				}
			}

			if (totalLossTimestep != DBL_MAX) _totalLoss += totalLossTimestep / onehotEncodedLabels[i].GetColumns();
			else _totalLoss = DBL_MAX;
		}
	}

	if (totalLossGeneralTimestep != DBL_MAX) _totalLossGeneral /= (_trainingSeqLength - 1);
	if (totalLossTimestep != DBL_MAX) _totalLoss /= (_trainingSeqLength - 1);


	int firstClickID = -1;
	int classID = -1;

	//only add them if probability is high enough eg. equal distribution most extreme case
	double minTopNProbabilitiesThreshold = selectNTopClasses > 1 ? 1.0 / (double)allClasses : 0;
	// class id --------- probability
	std::deque<std::tuple<int, double>> topNIDs;

	if (!trainMode)
	{
		std::vector<CUDAMatrix> outValues = _isBiRNN ? _outputValuesCombined : _outputValues;

		//for (int x = 0; x < outValues.size(); x++) outValues[x].Print();

		for (int j = 0; j < onehotEncodedLabels[0].GetRows(); j++) {
			if (onehotEncodedLabels[0](j, 0) == 1.0) {
				firstClickID = j;
				break;
			}
		}
		//if (performBackProp && verboseMode) std::cout << "STARTED WITH: " << firstClickID << std::endl;

		if (selectNTopClasses == 1)
		{
			double maxProbability = 0;
			for (int i = 0; i < outValues.size(); i++) {
				maxProbability = 0;//_outputValues[i].maxCoeff();
				for (int j = 0; j < outValues[i].GetRows(); j++) {
					if (outValues[i](j, 0) > maxProbability) {
						maxProbability = outValues[i](j, 0);
						classID = j;
					}
				}
				//std::cout << outputValues[i].maxCoeff() << std::endl;
				if (performBackProp && verboseMode) std::cout << classID << std::endl;
			}

			topNIDs.push_front(std::make_tuple(classID, maxProbability));
		}
		else {
			//CUDAMatrix lastTimeStep = outValues[outValues.size() - 1];
			CUDAMatrix lastTimeStep = outValues.back();
			for (int i = 0; i < std::min(selectNTopClasses, allClasses); i++) {
				topNIDs.push_front(std::make_tuple(i, lastTimeStep(i, 0)));
			}
			std::sort(topNIDs.begin(), topNIDs.end(), [](const auto& a, const auto& b) {
				return std::get<1>(a) < std::get<1>(b);
			});
			//topNIDs.push(std::make_tuple(0, lastTimeStep(0, 0)));
			for (int i = selectNTopClasses; i < lastTimeStep.GetRows(); i++) {

				// figure out at which position to insert if probability is high enough

				if (lastTimeStep(i, 0) > std::get<1>(topNIDs.front()))
				{
					std::tuple<int, double>& tupleToChange = topNIDs.front();
					//classID = -1;
					//double classProbability = 0;

					for (std::tuple<int, double>& tuple : topNIDs) {
						if (lastTimeStep(i, 0) > std::get<1>(tuple)) tupleToChange = tuple;
						else break;
						/*classID = i;
						classProbability = lastTimeStep(i, 0);*/
					}

					/*std::get<0>(tupleToChange) = classID;
					std::get<1>(tupleToChange) = classProbability;*/

					std::get<0>(tupleToChange) = i;
					std::get<1>(tupleToChange) = lastTimeStep(i, 0);
				}
			}
		}
	}

	if (performBackProp && verboseMode) std::cout << "TOTAL LOSS: " << _totalLoss << std::endl;

	// after going through whole sequence perform backprop

	if (performBackProp && _gatedUnits == NoGates) {
		if (_isBiRNN) BackPropBiRNN(onehotEncodedLabels, _learningRate, verboseMode);
		else BackProp(onehotEncodedLabels, _learningRate, verboseMode);
	}
	else if (performBackProp && _gatedUnits == GRU) BackPropGated(onehotEncodedLabels, _learningRate, verboseMode);

	std::sort(topNIDs.begin(), topNIDs.end(), [](const auto& a, const auto& b) {
		return std::get<1>(a) < std::get<1>(b);
	});
	
	// todo make min threshold optional
	while (!topNIDs.empty()) {
		//if (std::get<1>(topNIDs.front()) > minTopNProbabilitiesThreshold) break;
		//if (std::get<1>(topNIDs.front()) > 1.0 / allClasses * 10.0) break;
		if (std::get<1>(topNIDs.front()) >= 1.0 / allClasses * 5.0) break;
		topNIDs.pop_front();
	}

	return topNIDs;
}

void UserSimulator::PrintPredictedClassProbabilities() {
	CUDAMatrix lastTimeStepProbabilities(_outputNeurons, 1);
	for (int i = 0; i < _outputValues[_outputValues.size() - 1].GetRows(); i++) {
		std::cout << "classID " << i << " " << _outputValues[_outputValues.size() - 1](i, 0) * 100 << "%" << std::endl;
	}
}

void UserSimulator::CopyParameters() {
	_inputWeightsCopy = _inputWeights;
	_inputWeightsCopyBack = _inputWeightsBack;
	_hiddenWeightsCopy = _hiddenWeights;
	_hiddenWeightsCopyBack = _hiddenWeightsBack;
	_weightsOutputCopy = _weightsOutput;
	_biasesHiddenCopy = _biasesHidden;
	_biasesHiddenCopyBack = _biasesHiddenBack;
	_biasesRecurrentHiddenCopy = _biasesRecurrentHidden;
	_biasesRecurrentHiddenCopyBack = _biasesRecurrentHiddenBack;
	_biasesOutputCopy = _biasesOutput;
}

void UserSimulator::RestoreBestParameters() {
	_inputWeights = _inputWeightsCopy;
	_inputWeightsBack = _inputWeightsCopyBack;
	_hiddenWeights = _hiddenWeightsCopy;
	_hiddenWeightsBack = _hiddenWeightsCopyBack;
	_weightsOutput = _weightsOutputCopy;
	_biasesHidden = _biasesHiddenCopy;
	_biasesHiddenBack = _biasesHiddenCopyBack;
	_biasesRecurrentHidden = _biasesRecurrentHiddenCopy;
	_biasesRecurrentHiddenBack = _biasesRecurrentHiddenCopyBack;
	_biasesOutput = _biasesOutputCopy;
}