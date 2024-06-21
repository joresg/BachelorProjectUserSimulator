#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <map>

#pragma once
class FileManager
{
public:
	//Eigen::MatrixXi ReadCSVFile(const char* filePath, bool isTrainData);
	std::tuple<std::vector<std::vector<int>>, std::map<std::string, int>> ReadTXTFile(const char* filePath, bool isTrainData);
};