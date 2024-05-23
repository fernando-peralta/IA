#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"

using std::string;

int main(void)
{
    int nimgs = 60000;
    string imgPath = "mnist/train-images.idx3-ubyte";
    string lblPath = "mnist/train-labels.idx1-ubyte";

	uint8_t (*images)[28][28] = (uint8_t (*)[28][28])malloc(nimgs * 784 * sizeof(uint8_t));
	uint8_t *labels = (uint8_t*)malloc(nimgs * sizeof(uint8_t));
	load_dataset(images, labels, imgPath, lblPath, nimgs);

	float weight[10][28][28], constant_weight[10];
	load_weight(weight, constant_weight, "perceptron.bin");
	int count = 0;

	for (int i = 0; i < nimgs; ++i) {
		int pred = predict_digit(images[i], weight, constant_weight);
		if (pred != labels[i])
			++count;
	}

	printf("%f\n", (float) count / (float) nimgs * 100.0f);
	return 0;
}
