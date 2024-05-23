#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include "perceptron.h"

using std::string;

int main(int argc, char *argv[])
{
	int iterations = 5;
    int nimgs = 60000;

    if (argc == 3) {
        iterations = atoi(argv[1]);
        nimgs = atoi(argv[2]);
    }

	uint8_t (*images)[28][28] = (uint8_t (*)[28][28])malloc(nimgs * 784 * sizeof(uint8_t));
	uint8_t *labels = (uint8_t*)malloc(nimgs * sizeof(uint8_t));

	float weight[10][28][28], constant_weight[10];
	uint8_t expected_result[10];

    string imgPath = "mnist/train-images.idx3-ubyte";
    string lblPath = "mnist/train-labels.idx1-ubyte";

	load_dataset(images, labels, imgPath, lblPath, nimgs);

	memset(weight, 0, sizeof(weight));
	memset(constant_weight, 0, sizeof(constant_weight));

	#pragma omp parallel for
	for (int digit = 0; digit < 10; ++digit) {
	    for (int t = 0; t < iterations; ++t) {
		    for (int i = 0; i < nimgs; ++i) {
			    for (int j = 0; j < 10; ++j)
				    expected_result[j] = labels[i] == j;
			    update_weight(digit, images[i], labels[i], weight, constant_weight);
		    }
	    }
	}

	save_weight(weight, constant_weight, "perceptron.bin");
	free(images);
	return 0;
}
