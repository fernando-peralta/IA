#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

using std::vector;

int32_t readInt(std::ifstream &);
void readImages(uint8_t[][28][28], const std::string &, int);
void readLabels(uint8_t[], const std::string &, int);
