#include "helper.h"

int32_t 
readInt(std::ifstream &file) {
    int32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    value = __builtin_bswap32(value); // Convert to host byte order
    return value;
}

vector<vector<uint8_t>> 
readImages(const std::string &filename, int &numImages, int &rows, int &cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filename << std::endl;
        exit(1);
    }

    int magicNumber = readInt(file);
    if (magicNumber != 0x00000803) {
        std::cerr << "Invalid magic number in " << filename << std::endl;
        exit(1);
    }

    numImages = readInt(file);
    rows = readInt(file);
    cols = readInt(file);

    std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(rows * cols));

    for (int i = 0; i < numImages; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }

    return images;
}

vector<uint8_t>
readLabels(const std::string &filename, int &numLabels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filename << std::endl;
        exit(1);
    }

    int magicNumber = readInt(file);
    if (magicNumber != 0x00000801) {
        std::cerr << "Invalid magic number in " << filename << std::endl;
        exit(1);
    }

    numLabels = readInt(file);
    vector<uint8_t> labels(numLabels);
    file.read(reinterpret_cast<char*>(labels.data()), numLabels);

    return labels;
}

__global__ void 
forwardPass(const uint8_t *images, const float *weights, float *outputs, int numImages, int imgSize, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numImages) {
        for (int c = 0; c < numClasses; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < imgSize; ++i) {
                sum += images[idx * imgSize + i] * weights[c * imgSize + i];
            }
            outputs[idx * numClasses + c] = sum;
        }
    }
}

__global__ void 
updateWeights(const uint8_t *images, const uint8_t *labels, const float *outputs, float *weights, int numImages, int imgSize, int numClasses, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numImages) {
        int label = labels[idx];
        for (int c = 0; c < numClasses; ++c) {
            float error = (c == label) - outputs[idx * numClasses + c];
            for (int i = 0; i < imgSize; ++i) {
                atomicAdd(&weights[c * imgSize + i], learningRate * error * images[idx * imgSize + i]);
            }
        }
    }
}

void 
saveWeights(const std::string &filename, const float *weights, int size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filename << std::endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(weights), size * sizeof(float));
    file.close();
}

void
loadWeights(const std::string &filename, float *weights, int size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(weights), size * sizeof(float));
    file.close();
}


