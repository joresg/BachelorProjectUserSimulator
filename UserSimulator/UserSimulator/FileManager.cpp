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

std::tuple<std::vector<std::vector<int>>, std::map<std::string, int>> FileManager::ReadTXTFile(const char* filePath, bool isTrainData) {
	std::vector<std::vector<int>> allSequences;

	std::ifstream infile(filePath);
	std::string line;
	int seqLength = 30;
	int cmdID = -1;
	int generatedID = 0;

	std::cout << "READING TXT FILE: " << filePath << std::endl;

	std::vector<int> seq;
	std::map<std::string, int> cmdIDs;

	while (std::getline(infile, line)) {

		// check if cmd already in map if not add with new ID

		if (cmdIDs.count(line)) {
			cmdID = cmdIDs[line];
		}
		else {
			cmdID = generatedID;

			cmdIDs.insert({ line, generatedID });

			generatedID++;

		}

		if (seq.size() < seqLength) seq.push_back(cmdID);
		else {
			// delete first and add last
			seq.erase(seq.begin());
			seq.push_back(cmdID);
		}

		if (seq.size() == seqLength) {
			allSequences.push_back(seq);
			//seq.clear();
		}

		//std::cout << line << std::endl;
	}

	return std::tuple<std::vector<std::vector<int>>, std::map<std::string, int>>(allSequences, cmdIDs);
}