#include "canny.h"
#include "omp.h"
#include "opencv4/opencv2/core/core.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;

void canny_edge_detect_wrapper(string, string, int, int);

int main(int argc, char **argv) {
  string input_image_path = argv[1];
  string output_image_path = argv[2];
  int low_threshold = stoi(argv[3]);
  int high_threshold = stoi(argv[4]);
  omp_set_dynamic(0);
  // cout << "dynamic: " << omp_get_dynamic() << " n_threads: " << omp_get_max_threads() << endl;
  canny_edge_detect_wrapper(input_image_path, output_image_path, high_threshold,
                            low_threshold);
  return 0;
}

void canny_edge_detect_wrapper(string input_image_path,
                               string output_image_path, int high_threshold,
                               int low_threshold) {

  cv::Mat input_image =
      cv::imread(input_image_path, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat gray_image;

  cv::cvtColor(input_image, gray_image, cv::COLOR_BGR2GRAY);

  int width = gray_image.cols;
  int height = gray_image.rows;
  cv::Mat canny_image(height, width, CV_8UC1, cv::Scalar::all(0));
  double start_time = omp_get_wtime();
  canny_edge_detect(gray_image.data, height, width, high_threshold,
                    low_threshold, canny_image.data);
  double run_time = omp_get_wtime() - start_time;
  cout << run_time * 1000 <<  endl;

  cv::imwrite(output_image_path, canny_image);
}
