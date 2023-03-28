#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#define NUM_ROWS 51813503
#define NUM_VALS 103565681

// Macro definition for nvidia error checking
#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, float **matrixDiagonal, const char *filename, int *num_rows, int *num_cols, int *num_vals)
{
    //int err;
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    // Get number of rows, columns, and non-zero values
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");

    int *row_ptr_t = (int *)malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *)malloc(*num_vals * sizeof(int));
    float *values_t = (float *)malloc(*num_vals * sizeof(float));
    float *matrixDiagonal_t = (float *)malloc(*num_rows * sizeof(float));
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *)malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++)
    {
        row_occurances[i] = 0;
    }

    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        row_occurances[row]++;
    }

    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++)
    {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);

    // Set the file position to the beginning of the file
    rewind(file);

    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++)
    {
        col_ind_t[i] = -1;
    }

    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int i = 0, j = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        row--;
        column--;

        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1)
        {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        if (row == column)
        {
            matrixDiagonal_t[j] = value;
            j++;
        }
        i = 0;
    }
    fclose(file);
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
    *matrixDiagonal = matrixDiagonal_t;
}

// CPU implementation of SYMGS using CSR, DO NOT CHANGE THIS
void symgs_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal)
{

    // forward sweep
    for (int i = 0; i < num_rows; i++)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }

    // backward sweep
    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }
        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }
}

__global__
void parallel_symgs(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal){
    // Each row has approximatly 2 values
    // Each row is stored coalesced
    // The upper triangle of the matrix can be computed in parallel
    // The diagonal does not modify x so we can ignore it
}

int main(int argc, const char *argv[])
{

    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }

    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    float *matrixDiagonal;

    const char *filename = argv[1];

    double start_total, end_total;
    double start_input, end_input;
    double start_cpu, end_cpu;
    double start_gpu, end_gpu;

    start_total = get_time();
    start_input = get_time();
    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename, &num_rows, &num_cols, &num_vals);
    end_input = get_time();

    float *x = (float *)malloc(num_rows * sizeof(float));
    float *x_gpu = (float *)malloc(num_rows * sizeof(float));

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (float)(rand() % 100) / (rand() % 100 + 1); // the number we use to divide cannot be 0, that's the reason of the +1
        x_gpu[i] = x[i];
    }

    // CPU
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    // GPU
    int *d_row_ptr, *d_col_ind;
    float *d_values, *d_matrixDiagonal, *d_x, *d_x_computed;
    int done = 0;
    
    CHECK(cudaMallocManaged(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK(cudaMallocManaged(&d_col_ind, num_vals * sizeof(int)));
    CHECK(cudaMallocManaged(&d_values, num_vals * sizeof(float)));
    CHECK(cudaMallocManaged(&d_matrixDiagonal, num_rows * sizeof(float)));
    CHECK(cudaMallocManaged(&d_x, num_rows * sizeof(float)));
    CHECK(cudaMallocManaged(&d_x_computed, num_vals * sizeof(int)));


    CHECK(cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_matrixDiagonal, matrixDiagonal, num_rows * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, x_gpu, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    int threads_per_block = 1;
    int rows_per_thread = 1;
    int num_blocks = num_rows / (threads_per_block * rows_per_thread) + 1;

    start_gpu = get_time();
    parallel_symgs<<<threads_per_block, num_blocks>>>(d_row_ptr, d_col_ind, d_values, num_rows, d_x, d_matrixDiagonal);
    //CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    end_gpu = get_time();

    CHECK(cudaMemcpy(x_gpu, d_x, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    //Accuracy check
    float acc;
    int num;
    for(int i = 0; i < num_rows; i++){
        if(x_gpu[i] - x[i] > 0.001 || x_gpu[i] - x[i] < -0.001){
			num++;
        }
    }
    acc = ((float)num / num_rows) * 100;
    
    end_total = get_time();

    // Print timings
    printf("SYMGS Time INPUT: %.10lf\n", end_input - start_input);
    printf("SYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("SYMGS Time GPU: %.10lf\n", end_gpu - start_gpu);
    printf("%d errors on %d righe (delta > 10^-3).\nAccuracy: %.3f%%\n", num, num_rows, 100.0 - acc);
    printf("Total execution time: %.10lf\n", end_total - start_total);

    // Free
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(x);
    free(x_gpu);

    // CUDA Free
    CHECK(cudaFree(d_row_ptr));
    CHECK(cudaFree(d_col_ind));
    CHECK(cudaFree(d_matrixDiagonal));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_x));

    return 0;

}
