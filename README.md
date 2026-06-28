# Parallel Programming Final Project

This project is a parallel image processing pipeline written in C with OpenMP.

The program generates a synthetic RGB image and processes it through these steps:

1. RGB to grayscale conversion
2. 3x3 blur filter
3. Sobel edge detection
4. Thresholding

Both sequential and parallel versions are included. The results are compared by execution time, speedup, efficiency, checksum values, and mismatched pixel count.

## Technologies

- C
- OpenMP
- MSYS2 / GCC

## Compile

```bash
gcc -O2 -fopenmp image_pipeline_openmp.c -o image_pipeline_openmp.exe
