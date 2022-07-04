#include <iostream>
#include "cuda_runtime.h"

#define KERNEL_SIZE 3
#define OFFSET 1

void canny_edge_detect(const uint8_t* input_image, int height, int width, int high_threshold, int low_threshold, uint8_t* output_image);
__global__ void gaussian_blur(const uint8_t* input_image, int height, int width, uint8_t* output_image);
void gradient_magnitude_direction(const uint8_t* input_image, int height, int width, double* magnitude, uint8_t* direction);
void non_max_suppression(double* gradient_magnitude, uint8_t* gradient_direction, int height, int width, double* output_image);
void thresholding(double* suppressed_image, int height, int width, int high_threshold, int low_threshold, uint8_t* output_image);
void hysteresis(uint8_t* input_image, int height, int width, uint8_t* output_image);


