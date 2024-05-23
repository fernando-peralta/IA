#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <string>

void update_weight(int, uint8_t[28][28], uint8_t, float[10][28][28], float[10]);
int predict_digit(uint8_t[28][28], float[10][28][28], float[10]);
void load_weight(float[10][28][28],	float[10], const char *path);
void save_weight(float[10][28][28], float[10], const char *path);
void load_dataset(uint8_t[][28][28], uint8_t[], std::string, std::string, int);
