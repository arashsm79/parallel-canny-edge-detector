# Parallel Canny Edge Detector
> Parallel implementation of canny edge detection algorithm using OpenMP and CUDA

* [Introduction](#Introduction)
* [Serial](#Serial)
* [OpenMP](#OpenMP)
* [CUDA](#CUDA)
* [Building](#Building)
* [Usage](#Usage)
* [Results](#Results)
* [References](#References)

# Introduction
Canny edge detection is a technique to extract useful structural information from different vision objects and dramatically reduce the amount of data to be processed. It has been widely applied in various computer vision systems. [1]

The process of Canny edge detection algorithm can be broken down to five different steps:

- Apply Gaussian filter to smooth the image in order to remove the noise
- Find the magnitude and direction of gradients of the image
- Apply non-max suppression: suppress edges that or not local maximum in the direction of gradient
- Apply double threshold to determine potential edges
- Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

you can read more about the algorithm in [this wikipedia article](https://en.wikipedia.org/wiki/Canny_edge_detector).


# Serial
The serial implementation is available in the [serial](https://github.com/arashsm79/parallel-canny-edge-detector/tree/serial) branch.

# OpenMP
There are four variations for the OpenMP implementation that utilizes different parallelization methods:
- `parallel for` avaiable in the [`main`](https://github.com/arashsm79/parallel-canny-edge-detector/tree/main) branch.
- `parallel for collapse` avaiable in the [`parallel for collapse`](https://github.com/arashsm79/parallel-canny-edge-detector/tree/parallel-for-collapse) branch.
- `parallel for dynamic schedule` avaiable in the [`parallel for collapse`](https://github.com/arashsm79/parallel-canny-edge-detector/tree/parallel-for-dynamic-schedule) branch.
- `taskloop` avaiable in the [`taskloop`](https://github.com/arashsm79/parallel-canny-edge-detector/tree/taskloop) branch.

# CUDA
The CUDA implementation is available in the [CUDA](https://github.com/arashsm79/parallel-canny-edge-detector/tree/cuda) branch.


# Building
The project uses CMAKE as its build system generator. Create a folder called `build` for example, in the project directory and `cd` into it. Then run
```
build ..
```
and finally:
```
make
```
The resulting binary should be available under `build/src`


# Usage
```
/path/to/executable /path/to/input/image /path/to/output/image low_threshold  high_threshold 
```
Example:
```
./src/Main ../input/1280x720.jpg ../output/1280x720.jpg 30 90
```
# Results
Tables and charts are available in [this PDF file](https://github.com/arashsm79/parallel-canny-edge-detector/blob/main/results/arashsm79-parallel-canny-edge-detector-results.pdf).

Using OpenMP and its directives, we achieved speedups up to 4. This speed was gained without much tinkering with the code. All we did was add some directives to the serial implementation.

On the specified hardware, 8 threads on average, seem to be the optimal number of threads. The number 8 corresponds to the number of CPU cores on the specified hardware.

`parallel for` seems to be the way to go for the simple task of iterating over all pixels of the image. For huge input sizes however, `dynamic schedule` yields better results.

Using CUDA we were able to achieved speedups up to 10. In contrast to OpenMP, we had to change the code quite a bit to be able to run it efficiently on a GPU.
The small VRAM on the GPU caused some problems when trying to allocate memory for huge inputs sizes.

![example input](/input/1920x1080.jpg "Example Input")
![example output](/output/1920x1080.jpg "Example Output")

---
# References
* [1] https://en.wikipedia.org/wiki/Canny_edge_detector
