#include "FileManager.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <windows.h>
#include <time.h>
#include <map>
#include <algorithm>
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

//Eigen::MatrixXi FileManager::ReadCSVFile(const char* filePath, bool isTrainData) {
//
//	std::cout << "reading file " << filePath << std::endl;
//
//	Eigen::MatrixXi res(42000, isTrainData ? 785 : 784);
//
//	std::ifstream infile(filePath);
//	std::string line;
//	bool firstLine = true;
//	int resRow = 0;
//
//	while (std::getline(infile, line)) {
//		int resCol = 0;
//		if (firstLine) {
//			firstLine = false;
//			continue;
//		}
//		char* p;
//		char* cstr = &line[0];
//		p = strtok(cstr, ",");
//		while (p != NULL) {
//			res(resRow, resCol) = std::stoi(p);
//			resCol++;
//			p = strtok(NULL, ",");
//		}
//		resRow++;
//	}
//
//	return res;
//}

std::tuple<std::vector<std::vector<int>>, std::map<std::string, int>> FileManager::ReadTXTFile(const char* filePath, bool isTrainData, int trainingSeqLength) {
	std::vector<std::vector<int>> allSequences;

	std::ifstream infile(filePath);
	std::string line;
	int seqLength = trainingSeqLength;
	// sliding window seems to work best to split up very large sequences
	int seqOverlap = seqLength - 1;
	int cmdID = -1;
	int generatedID = 0;
	int limitLines = 20000;

	std::cout << "READING TXT FILE: " << filePath << std::endl;

	std::vector<int> seq;
	std::vector<int> seqOverlapVec;
	std::map<std::string, int> cmdIDs;

	while (std::getline(infile, line)) {
		if (limitLines == 0) break;

		// check if cmd already in map if not add with new ID

		if (cmdIDs.count(line)) {
			cmdID = cmdIDs[line];
		}
		else {
			cmdID = generatedID;
			cmdIDs.insert({ line, cmdID });
			generatedID++;
		}

		if (seq.size() < seqLength) seq.push_back(cmdID);
		else {
			//// delete first and add last
			//seq.erase(seq.begin());
			//seq.push_back(cmdID);

			allSequences.push_back(seq);

			seqOverlapVec.clear();

			for (int i = seq.size() - seqOverlap; i < seq.size(); i++) {
				seqOverlapVec.push_back(seq[i]);
			}

			seq.clear();

			/*for (int i = 0; i < seqOverlapVec.size(); i++) {
				seq.push_back(seqOverlapVec[i]);
			}*/

			for (auto i : seqOverlapVec) seq.push_back(i);
			seq.push_back(cmdID);
		}

		limitLines--;
	}

	return std::tuple<std::vector<std::vector<int>>, std::map<std::string, int>>(allSequences, cmdIDs);
}

std::map<std::string, int> FileManager::AllClassesFromFile(const char* filePath) {
	std::vector<std::vector<int>> allSequences;

	std::ifstream infile(filePath);
	std::string line;
	int cmdID = -1;
	int generatedID = 0;
	int limitLines = 20000;

	std::cout << "READING TXT FILE: " << filePath << std::endl;

	std::map<std::string, int> cmdIDs;

	while (std::getline(infile, line)) {
		if (limitLines == 0) break;

		// check if cmd already in map if not add with new ID

		if (cmdIDs.count(line)) {
			cmdID = cmdIDs[line];
		}
		else {
			cmdID = generatedID;
			cmdIDs.insert({ line, cmdID });
			generatedID++;
		}

		limitLines--;
	}

	return cmdIDs;
}