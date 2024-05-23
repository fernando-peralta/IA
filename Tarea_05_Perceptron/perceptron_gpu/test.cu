#include "helper.h"
#include "float.h"

using std::cout;
using std::endl;

int 
main() 
{
    const std::string testImagesFile = "t10k-images.idx3-ubyte";
    const std::string testLabelsFile = "t10k-labels.idx1-ubyte";
    const std::string weightsFile = "perceptron_weights.bin";
    const int numClasses = 10;
    
    int numTestImages, rows, cols, numTestLabels;
    vector<vector<uint8_t>> testImages = readImages(testImagesFile, numTestImages, rows, cols);
    vector<uint8_t> testLabels = readLabels(testLabelsFile, numTestLabels);

    int imgSize = rows * cols;
    int weightsSize = numClasses * imgSize;
    vector<float> weights(weightsSize);

    loadWeights(weightsFile, weights.data(), weightsSize);

    uint8_t *d_testImages;
    float *d_weights, *d_outputs;
    cudaMalloc(&d_testImages, numTestImages * imgSize * sizeof(uint8_t));
    cudaMalloc(&d_weights, weightsSize * sizeof(float));
    cudaMalloc(&d_outputs, numTestImages * numClasses * sizeof(float));

    cudaMemcpy(d_testImages, testImages[0].data(), numTestImages * imgSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), weightsSize * sizeof(float), cudaMemcpyHostToDevice);

    forwardPass<<<(numTestImages + 255) / 256, 256>>>(d_testImages, d_weights, d_outputs, numTestImages, imgSize, numClasses);
    cudaDeviceSynchronize();

    vector<float> outputs(numTestImages * numClasses);
    cudaMemcpy(outputs.data(), d_outputs, numTestImages * numClasses * sizeof(float), cudaMemcpyDeviceToHost);

    int correct = 0;
    for (int i = 0; i < numTestImages; ++i) {
        int predictedLabel = -1;
        float maxScore = -FLT_MAX;
        for (int c = 0; c < numClasses; ++c) {
            if (outputs[i * numClasses + c] > maxScore) {
                maxScore = outputs[i * numClasses + c];
                predictedLabel = c;
            }
        }
        if (predictedLabel == testLabels[i]) {
            ++correct;
        }
    }
    float accuracy = static_cast<float>(correct) / numTestImages * 100.0f;
    cout << "Accuracy: " << accuracy << "%" << endl;

    cudaFree(d_testImages);
    cudaFree(d_weights);
    cudaFree(d_outputs);

    return 0;
}

