#include "helper.h"

int main() {
    const std::string trainImagesFile = "train-images.idx3-ubyte";
    const std::string trainLabelsFile = "train-labels.idx1-ubyte";
    const std::string weightsFile = "perceptron_weights.bin";
    const int numClasses = 10;
    const float learningRate = 0.0001;
    const int numEpochs = 5;
    
    int numImages, rows, cols, numLabels;
    vector<vector<uint8_t>> images = readImages(trainImagesFile, numImages, rows, cols);
    vector<uint8_t> labels = readLabels(trainLabelsFile, numLabels);

    int imgSize = rows * cols;
    int weightsSize = numClasses * imgSize;
    vector<float> weights(weightsSize, 0.0f);

    uint8_t *d_images, *d_labels;
    float *d_weights, *d_outputs;
    cudaMalloc(&d_images, numImages * imgSize * sizeof(uint8_t));
    cudaMalloc(&d_labels, numLabels * sizeof(uint8_t));
    cudaMalloc(&d_weights, weightsSize * sizeof(float));
    cudaMalloc(&d_outputs, numImages * numClasses * sizeof(float));

    cudaMemcpy(d_images, images[0].data(), numImages * imgSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), numLabels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), weightsSize * sizeof(float), cudaMemcpyHostToDevice);

    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        forwardPass<<<(numImages + 255) / 256, 256>>>(d_images, d_weights, d_outputs, numImages, imgSize, numClasses);
        cudaDeviceSynchronize();
        updateWeights<<<(numImages + 255) / 256, 256>>>(d_images, d_labels, d_outputs, d_weights, numImages, imgSize, numClasses, learningRate);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(weights.data(), d_weights, weightsSize * sizeof(float), cudaMemcpyDeviceToHost);

    saveWeights(weightsFile, weights.data(), weightsSize);

    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_weights);
    cudaFree(d_outputs);

    return 0;
}

