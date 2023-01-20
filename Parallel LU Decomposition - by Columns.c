#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 3

void printMatrix(double mat[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%lf ", mat[i][j]);
        printf("\n");
    }
}

void LU_decomposition(double mat[N][N], double lower[N][N],
    double upper[N][N], int rank, int size) {
    int i, j, k;
    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process - 1;

    // Initialize lower and upper matrices
    for (i = start_row; i <= end_row; i++) {
        for (j = 0; j < N; j++) {
            if (i >= j) {
                upper[i][j] = mat[i][j];
                for (k = 0; k < i; k++)
                    upper[i][j] -= lower[i][k] * upper[k][j];
            }
            if (i <= j) {
                lower[i][j] = mat[i][j];
                for (k = 0; k < i; k++)
                    lower[i][j] -= lower[i][k] * upper[k][j];
                if (i == j)
                    lower[i][j] = 1;
                else
                    lower[i][j] /= upper[i][i];
            }
        }
    }

    // Send data to other processes
    for (i = 0; i < size; i++) {
        if (i != rank) {
            int start_row = i * rows_per_process;
            int end_row = (i + 1) * rows_per_process - 1;
            MPI_Send(&upper[start_row][0], (end_row - start_row + 1) * N,
                MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&lower[start_row][0], (end_row - start_row + 1) * N,
                MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }

    // Receive data from other processes
    for (i = 0; i < size; i++) {
        if (i != rank) {
            int start_row = i * rows_per_process;
            int end_row = (i + 1) * rows_per_process - 1;
            MPI_Recv(&upper[start_row][0], (end_row - start_row + 1) * N,
                MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&lower[start_row][0], (end_row - start_row + 1) * N,
                MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

int main(int argc, char* argv[]) {
    double mat[N][N] = {{2, -1, -2}, {-4, 6, 3}, {-4, -2, 8}};
    double lower[N][N], upper[N][N];

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    LU_decomposition(mat, lower, upper, rank, size);

    // Print the result on process 0
    if (rank == 0) {
        printf("Lower triangular matrix:\n");
        printMatrix(lower);
        printf("Upper triangular matrix:\n");
        printMatrix(upper);
    }

    MPI_Finalize();
    return 0;
}


