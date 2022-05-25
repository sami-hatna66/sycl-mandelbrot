# SYCL Mandelbrots

Mandelbrot viewer making use of parallel processing optimisations using SYCL. Graphics are rendered via SDL. 

## Build Instructions
```
$ git clone https://github.com/sami-hatna66/sycl-mandelbrot.git
$ cd sycl-mandelbrot
$ mkdir build && cd build
$ cmake -DCMAKE_CXX_COMPILER=path/to/llvm/build/bin/clang++ ..
$ cmake --build .
$ ./mandelbrot
```