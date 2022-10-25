# SYCL Mandelbrots

Mandelbrot viewer making use of parallel processing optimisations using SYCL. Graphics are rendered via SDL. 
Tested on Ubuntu 20.04.4 compiled with DPC++ and ComputeCPP.

## Demo

https://user-images.githubusercontent.com/88731772/170325579-85e81c80-8ce0-423f-a376-830ab430f04a.mp4


## Build Instructions
```
$ git clone https://github.com/sami-hatna66/sycl-mandelbrot.git
$ cd sycl-mandelbrot
$ mkdir build && cd build
$ cmake -DCMAKE_CXX_COMPILER=path/to/llvm/build/bin/clang++ ..
$ cmake --build .
$ ./mandelbrot
```
