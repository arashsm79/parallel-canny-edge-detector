#include "canny.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <iostream>
#include <math.h>

// Easier error handling
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

using namespace std;
__device__ __constant__ double gaussian_kernel[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
__device__ __constant__ int gaussian_kernel_sum = 16;
__device__ __constant__ int8_t Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__device__ __constant__ int8_t Gy[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

void canny_edge_detect(const uint8_t *input_image, int height, int width,
                       int high_threshold, int low_threshold,
                       uint8_t *output_image) {

  size_t image_size = height * width;
  size_t n_threads = 256;
  size_t n_blocks = ceil(image_size / n_threads);

  double *nms_output = new double[image_size];
  uint8_t *double_thresh_output = new uint8_t[image_size];



  // Gaussian Blur
  uint8_t *gaussian_blur_input;
  uint8_t *gaussian_blur_output;
  checkCudaErrors(
      cudaMalloc((void **)&gaussian_blur_input, image_size * sizeof(uint8_t)));
  checkCudaErrors(
      cudaMalloc((void **)&gaussian_blur_output, image_size * sizeof(uint8_t)));
  checkCudaErrors(cudaMemcpy(gaussian_blur_input, input_image,
                             image_size * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));
  cout << "Launching gaussian blur kernel..." << endl;
  gaussian_blur<<<n_blocks, n_threads>>>(gaussian_blur_input, height, width,
                                         gaussian_blur_output);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  cout << "Launching gaussian blur kernel finished." << endl;
  checkCudaErrors(cudaFree(gaussian_blur_input));




  // Gradient Magnitude and Direction
  double *gradient_magnitude;
  uint8_t *gradient_direction;
  checkCudaErrors(
      cudaMalloc((void **)&gradient_magnitude, image_size * sizeof(double)));
  checkCudaErrors(
      cudaMalloc((void **)&gradient_direction, image_size * sizeof(uint8_t)));
  cout << "Launching gradient magnitide and direction kernel..." << endl;
  gradient_magnitude_direction<<<n_blocks, n_threads>>>(
      gaussian_blur_output, height, width, gradient_magnitude,
      gradient_direction);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  cout << "Launching gradient magnitide and direction kernel finished." << endl;
  double *gradient_magnitude_h = new double[image_size];
  uint8_t *gradient_direction_h = new uint8_t[image_size];
  checkCudaErrors(cudaMemcpy(gradient_magnitude_h, gradient_magnitude,
                             image_size * sizeof(double),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(gradient_direction_h, gradient_direction,
                             image_size * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(gaussian_blur_output));




  // Non-max Suppression
  cout << "Launching non max suppression kernel..." << endl;
  non_max_suppression(gradient_magnitude_h, gradient_direction_h, height, width,
                      nms_output);
  cout << "Launching non max suppression kernel finished." << endl;

  checkCudaErrors(cudaFree(gradient_magnitude));
  checkCudaErrors(cudaFree(gradient_direction));



  // Thresholding
  thresholding(nms_output, height, width, high_threshold, low_threshold,
               double_thresh_output);

  // Hystresis
  hysteresis(double_thresh_output, height, width, output_image);

  delete[] nms_output;
  delete[] double_thresh_output;
}

__global__ void gaussian_blur(const uint8_t *input_image, int height, int width,
                              uint8_t *output_image) {
  int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;

  // if pixel is out of bounds don't do anything
  if (pixel_id < 0 || pixel_id >= height * width)
    return;

  // calculate the row and col of this pixel in the image
  int row = pixel_id / width;
  int col = pixel_id % width;
  // if pixel is outside the given offset don't do anything
  if (col < OFFSET || col >= width - OFFSET || row < OFFSET ||
      row >= height - OFFSET)
    return;

  double output_intensity = 0;
  int kernel_index = 0;
  size_t pixel_index = col + (row * width);
  for (int krow = -OFFSET; krow <= OFFSET; krow++) {
    for (int kcol = -OFFSET; kcol <= OFFSET; kcol++) {
      output_intensity += input_image[pixel_index + (kcol + (krow * width))] *
                          gaussian_kernel[kernel_index];
      kernel_index++;
    }
  }
  output_image[pixel_id] = (uint8_t)(output_intensity / gaussian_kernel_sum);
}

__global__ void gradient_magnitude_direction(const uint8_t *input_image,
                                             int height, int width,
                                             double *magnitude,
                                             uint8_t *direction) {
  int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;

  // if pixel is out of bounds don't do anything
  if (pixel_id < 0 || pixel_id >= height * width)
    return;

  // calculate the row and col of this pixel in the image
  int row = pixel_id / width;
  int col = pixel_id % width;
  // if pixel is outside the given offset don't do anything
  if (col < OFFSET || col >= width - OFFSET || row < OFFSET ||
      row >= height - OFFSET)
    return;

  double grad_x_sum = 0.0;
  double grad_y_sum = 0.0;
  int kernel_index = 0;
  int pixel_index = col + (row * width);

  for (int krow = -OFFSET; krow <= OFFSET; krow++) {
    for (int kcol = -OFFSET; kcol <= OFFSET; kcol++) {
      grad_x_sum +=
          input_image[pixel_index + (kcol + (krow * width))] * Gx[kernel_index];
      grad_y_sum +=
          input_image[pixel_index + (kcol + (krow * width))] * Gy[kernel_index];
      kernel_index++;
    }
  }

  int pixel_direction = 0;

  if (grad_x_sum == 0.0 || grad_y_sum == 0.0) {
    magnitude[pixel_index] = 0;
  } else {
    magnitude[pixel_index] =
        ((std::sqrt((grad_x_sum * grad_x_sum) + (grad_y_sum * grad_y_sum))));
    double theta = std::atan2(grad_y_sum, grad_x_sum);
    theta = theta * (360.0 / (2.0 * M_PI));

    if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) ||
        (theta >= 157.5))
      pixel_direction = 1; // horizontal -
    else if ((theta > 22.5 && theta <= 67.5) ||
             (theta > -157.5 && theta <= -112.5))
      pixel_direction = 2; // north-east -> south-west/
    else if ((theta > 67.5 && theta <= 112.5) ||
             (theta >= -112.5 && theta < -67.5))
      pixel_direction = 3; // vertical |
    else if ((theta >= -67.5 && theta < -22.5) ||
             (theta > 112.5 && theta < 157.5))
      pixel_direction = 4; // north-west -> south-east \'
  }
  direction[pixel_index] = (uint8_t)pixel_direction;
}
void non_max_suppression(double *gradient_magnitude,
                         uint8_t *gradient_direction, int height, int width,
                         double *output_image) {
  memcpy(output_image, gradient_magnitude, width * height * sizeof(double));
  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      int pixel_index = col + (row * width);

      // unconditionally suppress border pixels
      if (row == OFFSET || col == OFFSET || col == width - OFFSET - 1 ||
          row == height - OFFSET - 1) {
        output_image[pixel_index] = 0;
        continue;
      }

      switch (gradient_direction[pixel_index]) {
      case 1:
        if (gradient_magnitude[pixel_index - 1] >=
                gradient_magnitude[pixel_index] ||
            gradient_magnitude[pixel_index + 1] >
                gradient_magnitude[pixel_index])
          output_image[pixel_index] = 0;
        break;
      case 2:
        if (gradient_magnitude[pixel_index - (width - 1)] >=
                gradient_magnitude[pixel_index] ||
            gradient_magnitude[pixel_index + (width - 1)] >
                gradient_magnitude[pixel_index])
          output_image[pixel_index] = 0;
        break;
      case 3:
        if (gradient_magnitude[pixel_index - (width)] >=
                gradient_magnitude[pixel_index] ||
            gradient_magnitude[pixel_index + (width)] >
                gradient_magnitude[pixel_index])
          output_image[pixel_index] = 0;
        break;
      case 4:
        if (gradient_magnitude[pixel_index - (width + 1)] >=
                gradient_magnitude[pixel_index] ||
            gradient_magnitude[pixel_index + (width + 1)] >
                gradient_magnitude[pixel_index])
          output_image[pixel_index] = 0;
        break;
      default:
        output_image[pixel_index] = 0;
        break;
      }
    }
  }
}
void thresholding(double *suppressed_image, int height, int width,
                  int high_threshold, int low_threshold,
                  uint8_t *output_image) {
  for (int col = 0; col < width; col++) {
    for (int row = 0; row < height; row++) {
      int pixel_index = col + (row * width);
      if (suppressed_image[pixel_index] > high_threshold)
        output_image[pixel_index] = 255; // Strong edge
      else if (suppressed_image[pixel_index] > low_threshold)
        output_image[pixel_index] = 100; // Weak edge
      else
        output_image[pixel_index] = 0; // Not an edge
    }
  }
}

void hysteresis(uint8_t *input_image, int height, int width,
                uint8_t *output_image) {
  memcpy(output_image, input_image, width * height * sizeof(uint8_t));
  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      int pixel_index = col + (row * width);
      if (output_image[pixel_index] == 100) {
        if (output_image[pixel_index - 1] == 255 ||
            output_image[pixel_index + 1] == 255 ||
            output_image[pixel_index - width] == 255 ||
            output_image[pixel_index + width] == 255 ||
            output_image[pixel_index - width - 1] == 255 ||
            output_image[pixel_index - width + 1] == 255 ||
            output_image[pixel_index + width - 1] == 255 ||
            output_image[pixel_index + width + 1] == 255)
          output_image[pixel_index] = 255;
        else
          output_image[pixel_index] = 0;
      }
    }
  }
}
