#include "helper.h"

int32_t 
readInt(std::ifstream &file) {
    int32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    value = ((value & 0xFF000000) >> 24) |
            ((value & 0x00FF0000) >> 8) |
            ((value & 0x0000FF00) << 8) |
            ((value & 0x000000FF) << 24);
    return value;
}

void 
readImages(uint8_t images[][28][28], const std::string &filename, int numImages) {
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

    int readNumImages = readInt(file);
    int rows = readInt(file);
    int cols = readInt(file);

    if (rows != 28 || cols != 28) {
        std::cerr << "Invalid image dimensions: " << rows << "x" << cols << std::endl;
        exit(1);
    }

    if (numImages > readNumImages) {
        std::cerr << "Number of requested images exceeds the available images in file" << std::endl;
        exit(1);
    }

    for (int i = 0; i < numImages; ++i) {
        if (!file.read(reinterpret_cast<char*>(images[i]), rows * cols)) {
            std::cerr << "Error reading image " << i << " from file " << filename << std::endl;
            exit(1);
        }
    }
}

void 
readLabels(uint8_t labels[], const std::string &filename, int numLabels) {
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

    int readNumLabels = readInt(file);

    if (numLabels > readNumLabels) {
        std::cerr << "Number of requested labels exceeds the available labels in file" << std::endl;
        exit(1);
    }

    if (!file.read(reinterpret_cast<char*>(labels), numLabels)) {
        std::cerr << "Error reading labels from file " << filename << std::endl;
        exit(1);
    }
}
