#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

using std::vector;

int32_t readInt(std::ifstream &);
vector<vector<uint8_t>> readImages(const std::string &, int &, int &, int &);
vector<uint8_t> readLabels(const std::string &, int &);
__global__ void forwardPass(const uint8_t *, const float *, float *, int, int, int);
__global__ void updateWeights(const uint8_t *, const uint8_t *, const float *, float *, int, int, int, float);
void saveWeights(const std::string &, const float *, int);
void loadWeights(const std::string &, float *, int);
