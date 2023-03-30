#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>

#define CUDA_THREADS 1408

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

// Gets the accuracy of gpu against cpu
float get_accuracy(float *x_gpu, float *x, int num_rows){
    float acc;
    float err;
    int num;
    for(int i = 0; i < num_rows; i++){
        if((err = fabs(x_gpu[i] - x[i]) ) > 0.1 && err / x[i] > 0.1){
			num++;
        }
    }
    acc = 100.0 - ((float) num / num_rows) * 100;
    return acc;
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
void parallel_symgs_fw(
                        const int *row_ptr, 
                        const int *col_ind, 
                        const float *values, 
                        const int num_rows, 
                        float *matrixDiagonal,
                        float *old_x,
                        float *new_x,
                        bool *updated)
{
    int row_index = threadIdx.x + blockDim.x * blockIdx.x;
    if(row_index >= num_rows || updated[row_index]) return;

    float sum = old_x[row_index];
    int row_start = row_ptr[row_index];
    int row_stop = row_ptr[row_index + 1];
    float currDiag = matrixDiagonal[row_index];
    bool row_ready = true;

    for(int i = row_start; 
        i < row_stop;   
        i++)
    {
        if(!row_ready) break;                                   // Row isn't ready, abort all row calcs
        if(col_ind[i] < 0) continue;                            // Out of bound value to be handled
        if(col_ind[i] >= row_index)                             // If the value is above the main diag, it has no dep
            sum -= values[i] * old_x[col_ind[i]];
        else if (updated[col_ind[i]])                           // If dep has already been updated, use it
            sum -= values[i] * new_x[col_ind[i]];
        else 
            row_ready = false;                                  // Otherwise lower the row ready flag
    }
    if (row_ready)                                              // If row ready is raised after whole row, row has finished
    {
        sum += old_x[row_index] * currDiag;                     // Remove diagonal contribution
        new_x[row_index] = sum / currDiag;                      // Update x value
        updated[row_index] = true;                              // Raise the updated value of the row
    }
    else updated[num_rows] = false;                             // Otherwise set done to false (new iteration is needed)
}

__global__
void parallel_symgs_bw(
                        const int *row_ptr, 
                        const int *col_ind, 
                        const float *values, 
                        const int num_rows, 
                        float *matrixDiagonal,
                        float *old_x,
                        float *new_x,
                        bool *updated)
{
    int row_index = threadIdx.x + blockDim.x * blockIdx.x;
    if(row_index >= num_rows || updated[row_index]) return;

    float sum = old_x[row_index];
    int row_start = row_ptr[row_index];
    int row_stop = row_ptr[row_index + 1];
    float currDiag = matrixDiagonal[row_index];
    bool row_ready = true;

    for(int i = row_start; 
        i < row_stop;   
        i++)
    {
        if(!row_ready) break;                                   // Row isn't ready, abort all row calcs
        if(col_ind[i] < 0) continue;                            // Out of bound value to be handled
        if(col_ind[i] <= row_index)                             // If the value is below the main diag, it has no dep
            sum -= values[i] * old_x[col_ind[i]];
        else if (updated[col_ind[i]])                           // If dep has already been updated, use it
            sum -= values[i] * new_x[col_ind[i]];
        else 
            row_ready = false;                                  // Otherwise lower the row ready flag
    }
    if (row_ready)                                              // If row ready is raised after whole row, row has finished
    {
        sum += old_x[row_index] * currDiag;
        new_x[row_index] = sum / currDiag;                      // Update x value
        updated[row_index] = true;                              // Raise the updated value of the row
    }
    else updated[num_rows] = false;                             // Otherwise set done to false (new iteration is needed)
}

int main(int argc, const char *argv[])
{

    if (argc != 3)
    {
        printf("Usage: ./exec matrix_file threads_per_block");
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
        x[i] = (float) (rand() % 100) / (float) (rand() % 100 + 1); // the number we use to divide cannot be 0, that's the reason of the +1
        x_gpu[i] = x[i];
    }

    // CPU
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    // GPU
    int *d_row_ptr, *d_col_ind;
    float *d_matrixDiagonal, *d_new_x, *d_old_x, *d_values;
    bool *d_updated;
    bool done = false;
    
    // Device mem allocation
    CHECK(cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_col_ind, num_vals * sizeof(int)));
    CHECK(cudaMalloc(&d_values, num_vals * sizeof(float)));
    CHECK(cudaMalloc(&d_matrixDiagonal, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&d_new_x, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&d_old_x, num_rows * sizeof(float)))
    CHECK(cudaMalloc(&d_updated, num_rows * sizeof(bool) + 1));

    // Array init for the updated flag
    // Last pos of array is a flag that gets lowered by the device
    // when the sweep has not been completed and needs to be reiterated
    CHECK(cudaMemset(d_updated, 0, num_rows * sizeof(bool) + 1));

    // Matrix data copy from host memory to device memory
    CHECK(cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(d_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyDefault));
    CHECK(cudaMemcpy(d_matrixDiagonal, matrixDiagonal, num_rows * sizeof(float), cudaMemcpyDefault));
    CHECK(cudaMemcpy(d_old_x, x_gpu, num_rows * sizeof(float), cudaMemcpyDefault));

    int threads_per_block = atoi(argv[2]);
    int num_blocks = num_rows / threads_per_block;

    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(threads_per_block, 1, 1);

    start_gpu = get_time();
    
    while(!done){
        CHECK(cudaMemset(d_updated + num_rows, 1, sizeof(bool)));
        parallel_symgs_fw<<<dimGrid, dimBlock>>>(
            d_row_ptr, 
            d_col_ind, 
            d_values, 
            num_rows,
            d_matrixDiagonal,
            d_old_x, 
            d_new_x,
            d_updated);
        CHECK(cudaDeviceSynchronize());
        CHECK_KERNELCALL();
        CHECK(cudaMemcpy(&done, d_updated + num_rows, sizeof(bool), cudaMemcpyDefault));
    }

    done = false;
    CHECK(cudaMemset(d_updated, 0, num_rows * sizeof(bool) + 1));

    while(!done){
        CHECK(cudaMemset(d_updated + num_rows, 1, sizeof(bool)));
        parallel_symgs_bw<<<dimGrid, dimBlock>>>(
            d_row_ptr, 
            d_col_ind, 
            d_values, 
            num_rows,
            d_matrixDiagonal,
            d_old_x, 
            d_new_x,
            d_updated);
        CHECK(cudaDeviceSynchronize());
        CHECK_KERNELCALL();
        CHECK(cudaMemcpy(&done, d_updated + num_rows, sizeof(bool), cudaMemcpyDefault));
    }
    
    end_gpu = get_time();

    CHECK(cudaMemcpy(x_gpu, d_new_x, num_rows * sizeof(float), cudaMemcpyDefault));

    end_total = get_time();

    // Print timings
    printf("SYMGS Time INPUT: %.10lf\n", end_input - start_input);
    printf("SYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("SYMGS Time GPU: %.10lf\n", end_gpu - start_gpu);
    printf("Accuracy: %.3f%%\n", get_accuracy(x_gpu, x, num_rows));
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
    CHECK(cudaFree(d_new_x));
    CHECK(cudaFree(d_old_x));
    CHECK(cudaFree(d_updated));

    return 0;
}