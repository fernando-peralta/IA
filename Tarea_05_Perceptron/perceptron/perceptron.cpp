#include "perceptron.h"
#include "helper.h"

using std::string;

void 
update_weight(int digit, uint8_t input[28][28], uint8_t label,	float weight[10][28][28], float constant_weight[10])
{
	float result = constant_weight[digit];
	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j)
			result += (float) input[i][j] * weight[digit][i][j];
	}

	float error = (digit == label) - (result > 0.0f);

	constant_weight[digit] += 1e-4f * error;
	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j)
			weight[digit][i][j] += 1e-4f * (float) input[i][j] * error;
	}
}

int 
predict_digit(uint8_t input[28][28], float weight[10][28][28], float constant_weight[10])
{
	float max_result = -1e18f;
	int prediction = -1;
	for (int d = 0; d < 10; ++d) {
		float result = 0;
		for (int i = 0; i < 28; ++i) {
			for (int j = 0; j < 28; ++j)
				result += (float) input[i][j] * weight[d][i][j];
		}
		result += constant_weight[d];
		if (max_result < result) {
			prediction = d;
			max_result = result;
		}
	}

	return prediction;
}

void
load_weight(float weight[10][28][28], float constant_weight[10], const char *path)
{
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "File can't be opened: %s\n", path);
		exit(1);
	}

	fread(weight, sizeof(float), 10*28*28, file);
	fread(constant_weight, sizeof(float), 10, file);
	fclose(file);
}

void 
save_weight(float weight[10][28][28], float constant_weight[10], const char *path)
{
	FILE *file = fopen(path, "wb+");
	if (!file) {
		fprintf(stderr, "File can't be opened: %s\n", path);
		exit(1);
	}

	fwrite(weight, sizeof(float), 10*28*28, file);
	fwrite(constant_weight, sizeof(float), 10, file);
	fclose(file);
}

void 
load_dataset(uint8_t images[][28][28], uint8_t labels[], std::string images_path, std::string labels_path, int numImgs) 
{
    readImages(images, images_path, numImgs);   
    readLabels(labels, labels_path, numImgs);
}
