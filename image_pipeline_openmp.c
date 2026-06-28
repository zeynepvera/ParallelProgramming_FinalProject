#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define THRESHOLD 100
#define CHUNK_SIZE 128

static unsigned char clamp_int(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (unsigned char)value;
}

static unsigned char *allocate_image(int width, int height) {
    unsigned char *img = (unsigned char *)malloc((size_t)width * height * sizeof(unsigned char));

    if (img == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    return img;
}

static void generate_rgb_image(unsigned char *r, unsigned char *g, unsigned char *b,
                               int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

            r[idx] = (unsigned char)((x * 255) / width);
            g[idx] = (unsigned char)((y * 255) / height);
            b[idx] = (unsigned char)((((x / 32) + (y / 32)) % 2) * 255);
        }
    }
}

static int write_pgm(const char *filename, unsigned char *image, int width, int height) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        printf("Could not open output file: %s\n", filename);
        return 0;
    }

    fprintf(file, "P5\n%d %d\n255\n", width, height);
    fwrite(image, sizeof(unsigned char), (size_t)width * height, file);
    fclose(file);

    return 1;
}

static void set_schedule_policy(const char *schedule_name) {
    if (strcmp(schedule_name, "static") == 0) {
        omp_set_schedule(omp_sched_static, CHUNK_SIZE);
    } else if (strcmp(schedule_name, "dynamic") == 0) {
        omp_set_schedule(omp_sched_dynamic, CHUNK_SIZE);
    } else if (strcmp(schedule_name, "guided") == 0) {
        omp_set_schedule(omp_sched_guided, CHUNK_SIZE);
    } else {
        printf("Unknown schedule type. Use: static, dynamic, or guided.\n");
        exit(1);
    }
}

static long long sequential_pipeline(unsigned char *r, unsigned char *g, unsigned char *b,
                                     unsigned char *gray, unsigned char *blur,
                                     unsigned char *edge, unsigned char *out,
                                     int width, int height, int threshold) {
    long long checksum = 0;

    /*
        Stage 1: RGB to grayscale
        Formula:
        gray = 0.299R + 0.587G + 0.114B

        Integer version:
        gray = (77R + 150G + 29B) / 256
    */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            gray[idx] = (unsigned char)((77 * r[idx] + 150 * g[idx] + 29 * b[idx]) / 256);
        }
    }

    /*
        Stage 2: 3x3 blur
        Each pixel is replaced by the average of its 3x3 neighborhood.
    */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

            if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                blur[idx] = gray[idx];
            } else {
                int sum = 0;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        sum += gray[(y + ky) * width + (x + kx)];
                    }
                }

                blur[idx] = (unsigned char)(sum / 9);
            }
        }
    }

    /*
        Stage 3: Sobel edge detection
        Sobel filter estimates horizontal and vertical intensity changes.
    */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

            if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                edge[idx] = 0;
            } else {
                int gx = 0;
                int gy = 0;

                gx += -1 * blur[(y - 1) * width + (x - 1)];
                gx +=  1 * blur[(y - 1) * width + (x + 1)];
                gx += -2 * blur[y * width + (x - 1)];
                gx +=  2 * blur[y * width + (x + 1)];
                gx += -1 * blur[(y + 1) * width + (x - 1)];
                gx +=  1 * blur[(y + 1) * width + (x + 1)];

                gy += -1 * blur[(y - 1) * width + (x - 1)];
                gy += -2 * blur[(y - 1) * width + x];
                gy += -1 * blur[(y - 1) * width + (x + 1)];
                gy +=  1 * blur[(y + 1) * width + (x - 1)];
                gy +=  2 * blur[(y + 1) * width + x];
                gy +=  1 * blur[(y + 1) * width + (x + 1)];

                edge[idx] = clamp_int(abs(gx) + abs(gy));
            }
        }
    }

    /*
        Stage 4: Thresholding
        If edge value is larger than threshold, output pixel becomes white.
        Otherwise, it becomes black.

        checksum is used to verify correctness.
    */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

            out[idx] = (edge[idx] > threshold) ? 255 : 0;
            checksum += out[idx];
        }
    }

    return checksum;
}

static long long parallel_pipeline(unsigned char *r, unsigned char *g, unsigned char *b,
                                   unsigned char *gray, unsigned char *blur,
                                   unsigned char *edge, unsigned char *out,
                                   int width, int height, int threshold) {
    long long checksum = 0;
    int x, y;

    /*
        One parallel region is used for the whole image processing pipeline.

        default(none):
        Forces us to explicitly define variable scope.

        shared:
        Image arrays are shared because all threads read/write different pixels.

        firstprivate:
        width, height and threshold are copied to each thread.

        private:
        x and y loop variables are private for each thread.

        reduction:
        checksum is safely accumulated by all threads.
    */
    #pragma omp parallel default(none) shared(r, g, b, gray, blur, edge, out) firstprivate(width, height, threshold) private(x, y) reduction(+:checksum)
    {
        /*
            Stage 1: RGB to grayscale

            collapse(2) combines the y and x loops into one larger iteration space.
            schedule(runtime) allows the schedule type to be selected at runtime.
        */
        #pragma omp for collapse(2) schedule(runtime)
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                int idx = y * width + x;
                gray[idx] = (unsigned char)((77 * r[idx] + 150 * g[idx] + 29 * b[idx]) / 256);
            }
        }

        /*
            There is an implicit barrier here.
            Blur must not start before grayscale is completely finished.
        */

        /*
            Stage 2: 3x3 blur
        */
        #pragma omp for collapse(2) schedule(runtime)
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                int idx = y * width + x;

                if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                    blur[idx] = gray[idx];
                } else {
                    int sum = 0;

                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            sum += gray[(y + ky) * width + (x + kx)];
                        }
                    }

                    blur[idx] = (unsigned char)(sum / 9);
                }
            }
        }

        /*
            There is an implicit barrier here.
            Sobel must not start before blur is completely finished.
        */

        /*
            Stage 3: Sobel edge detection
        */
        #pragma omp for collapse(2) schedule(runtime)
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                int idx = y * width + x;

                if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                    edge[idx] = 0;
                } else {
                    int gx = 0;
                    int gy = 0;

                    gx += -1 * blur[(y - 1) * width + (x - 1)];
                    gx +=  1 * blur[(y - 1) * width + (x + 1)];
                    gx += -2 * blur[y * width + (x - 1)];
                    gx +=  2 * blur[y * width + (x + 1)];
                    gx += -1 * blur[(y + 1) * width + (x - 1)];
                    gx +=  1 * blur[(y + 1) * width + (x + 1)];

                    gy += -1 * blur[(y - 1) * width + (x - 1)];
                    gy += -2 * blur[(y - 1) * width + x];
                    gy += -1 * blur[(y - 1) * width + (x + 1)];
                    gy +=  1 * blur[(y + 1) * width + (x - 1)];
                    gy +=  2 * blur[(y + 1) * width + x];
                    gy +=  1 * blur[(y + 1) * width + (x + 1)];

                    edge[idx] = clamp_int(abs(gx) + abs(gy));
                }
            }
        }

        /*
            There is an implicit barrier here.
            Thresholding must not start before Sobel is completely finished.
        */

        /*
            Stage 4: Thresholding and checksum calculation
        */
        #pragma omp for collapse(2) schedule(runtime)
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                int idx = y * width + x;

                out[idx] = (edge[idx] > threshold) ? 255 : 0;
                checksum += out[idx];
            }
        }
    }

    return checksum;
}

static long long count_differences(unsigned char *a, unsigned char *b,
                                   int width, int height) {
    long long diff_count = 0;
    size_t total = (size_t)width * height;

    for (size_t i = 0; i < total; i++) {
        if (a[i] != b[i]) {
            diff_count++;
        }
    }

    return diff_count;
}

int main(int argc, char *argv[]) {
    int width = 2048;
    int height = 2048;
    int threads = 4;
    int runs = 5;
    const char *schedule_name = "static";

    /*
        Usage:
        program.exe [width] [height] [threads] [static|dynamic|guided] [runs]

        Example:
        image_pipeline_openmp.exe 2048 2048 8 static 5
    */
    if (argc >= 3) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }

    if (argc >= 4) {
        threads = atoi(argv[3]);
    }

    if (argc >= 5) {
        schedule_name = argv[4];
    }

    if (argc >= 6) {
        runs = atoi(argv[5]);
    }

    if (width < 3 || height < 3 || threads < 1 || runs < 1) {
        printf("Usage: %s [width] [height] [threads] [static|dynamic|guided] [runs]\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(threads);
    set_schedule_policy(schedule_name);

    unsigned char *r = allocate_image(width, height);
    unsigned char *g = allocate_image(width, height);
    unsigned char *b = allocate_image(width, height);

    unsigned char *gray_seq = allocate_image(width, height);
    unsigned char *blur_seq = allocate_image(width, height);
    unsigned char *edge_seq = allocate_image(width, height);
    unsigned char *out_seq = allocate_image(width, height);

    unsigned char *gray_par = allocate_image(width, height);
    unsigned char *blur_par = allocate_image(width, height);
    unsigned char *edge_par = allocate_image(width, height);
    unsigned char *out_par = allocate_image(width, height);

    generate_rgb_image(r, g, b, width, height);

    double seq_total_time = 0.0;
    double par_total_time = 0.0;

    long long seq_checksum = 0;
    long long par_checksum = 0;

    /*
        Sequential version is executed multiple times.
        The average execution time is used.
    */
    for (int i = 0; i < runs; i++) {
        double start = omp_get_wtime();

        seq_checksum = sequential_pipeline(r, g, b,
                                           gray_seq, blur_seq, edge_seq, out_seq,
                                           width, height, THRESHOLD);

        double end = omp_get_wtime();
        seq_total_time += end - start;
    }

    /*
        Parallel version is executed multiple times.
        The average execution time is used.
    */
    for (int i = 0; i < runs; i++) {
        double start = omp_get_wtime();

        par_checksum = parallel_pipeline(r, g, b,
                                         gray_par, blur_par, edge_par, out_par,
                                         width, height, THRESHOLD);

        double end = omp_get_wtime();
        par_total_time += end - start;
    }

    double seq_avg = seq_total_time / runs;
    double par_avg = par_total_time / runs;
    double speedup = seq_avg / par_avg;
    double efficiency = speedup / threads;

    long long differences = count_differences(out_seq, out_par, width, height);

    write_pgm("output_sequential.pgm", out_seq, width, height);
    write_pgm("output_parallel.pgm", out_par, width, height);

    printf("\nOpenMP Image Processing Pipeline\n");
    printf("Image size              : %d x %d\n", width, height);
    printf("Threads                 : %d\n", threads);
    printf("Schedule                : %s, chunk=%d\n", schedule_name, CHUNK_SIZE);
    printf("Runs                    : %d\n", runs);
    printf("Sequential avg time (s) : %.6f\n", seq_avg);
    printf("Parallel avg time (s)   : %.6f\n", par_avg);
    printf("Speedup                 : %.4f\n", speedup);
    printf("Efficiency              : %.4f\n", efficiency);
    printf("Sequential checksum     : %lld\n", seq_checksum);
    printf("Parallel checksum       : %lld\n", par_checksum);
    printf("Mismatched pixels       : %lld\n", differences);
    printf("Output files            : output_sequential.pgm, output_parallel.pgm\n");

    /*
        CSV line for easy copy-paste into Excel.
    */
    printf("CSV,%d,%d,%d,%s,%d,%.6f,%.6f,%.4f,%.4f,%lld,%lld,%lld\n",
           width, height, threads, schedule_name, runs,
           seq_avg, par_avg, speedup, efficiency,
           seq_checksum, par_checksum, differences);

    free(r);
    free(g);
    free(b);

    free(gray_seq);
    free(blur_seq);
    free(edge_seq);
    free(out_seq);

    free(gray_par);
    free(blur_par);
    free(edge_par);
    free(out_par);

    return 0;
}