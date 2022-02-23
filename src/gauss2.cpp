/* File:      gauss2.cpp
 *
 * Compile:   g++ -g -Wall -fopenmp -o gauss2 gauss2.cpp
 *
 * Run:       ./gauss2 <n> <num_threads> <ToPrint>
 * Note:      N = the num of formulas
 *            n should be evenly divided by num_proc
 *
 * Input:     None
 * Output:    The elapsed times for gauss using OpenMP
 *
 */

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stack>
using namespace std;

void GenMat(double* A, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = rand() % 10 + 1;
        }
    }
}

void GenVec(double* A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = rand() % 5;
    }
}

void PrintMat(double* A, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << A[i * size + j] << " ";
        }
        cout << endl;
    }
}

void PrintVec(double* A, int size) {
    for (int i = 0; i < size; i++) {
        cout << A[i] << endl;
    }
}

int main(int argc, char* argv[]) {
    int n, num_threads, ToPrint;
    if (argc > 3) {
        n = atoi(argv[1]);
        num_threads = atoi(argv[2]);
        ToPrint = atoi(argv[3]);
    }
    else {
        n = 4;
        num_threads = 2;
        ToPrint = 1;
    }

    // allocate
    double* A = new double[n * n];
    double* b = new double[n];
    double* x = new double[n];
    double* check = new double[n * n];
    int* calculated = new int[n];
    stack<int> elim;
    for (int i = 0; i < n; i++) {
        calculated[i] = -1;
    }

    // generate matrix A and vector b in random
    srand(time(NULL));
    GenMat(A, n);
    GenVec(b, n);

    // print matrix and vector (if n is small)
    if (ToPrint) {
        cout << "Matrix A:" << endl;
        PrintMat(A, n);
        cout << "Vector b:" << endl;
        PrintVec(b, n);
        for (int i = 0; i < n * n; i++) {
            check[i] = A[i];
        }
    }

    // begin
    double start_time = omp_get_wtime();

    //  Upper triangulation
    for (int j = 0; j < n; j++) {
        double max_coff = 0;
        int max_index;
        for (int i = 0; i < n; i++) {
            if (calculated[i] == -1 && abs(A[i * n + j]) > abs(max_coff)) {
                max_coff = A[i * n + j];
                max_index = i;
            }
        }
        calculated[max_index] = j;
        elim.push(max_index);
#pragma omp parallel for num_threads(num_threads) shared(A, b) schedule(dynamic)
        for (int i = 0; i < n; i++) {
            if (calculated[i] == -1) {
                double tmp_coff = A[i * n + j] / A[max_index * n + j];
                A[i * n + j] = 0;
                for (int k = j + 1; k < n; k++) {
                    A[i * n + k] -= tmp_coff * A[max_index * n + k];
                }
                b[i] -= tmp_coff * b[max_index];
            }
        }
    }

    // elimination element and get a solution
    for (int j = n - 1; j >= 0; j--) {
        int cur_index = elim.top();
        elim.pop();
        calculated[cur_index] = -1;
#pragma omp parallel for num_threads(num_threads) shared(A, b) schedule(dynamic)
        for (int i = 0; i < n; i++) {
            if (calculated[i] != -1) {
                double tmp_coff = A[i * n + j] / A[cur_index * n + j];
                A[i * n + j] = 0;
                b[i] -= tmp_coff * b[cur_index];
            }
        }
        b[cur_index] /= A[cur_index * n + j];
        A[cur_index * n + j] = 1;
        x[j] = b[cur_index];
    }

    // end
    double end_time = omp_get_wtime();

    // print answer(if n is small)
    if (ToPrint) {
        cout << "Answer x:" << endl;
        PrintVec(x, n);

        cout << "Check x:" << endl;
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sum += check[i * n + j] * x[j];
            }
            cout << sum << endl;
        }
    }

    // print elapsed time
    cout << "Time: " << end_time - start_time << endl;

    // free
    delete[]A;
    delete[]b;
    delete[]x;
    delete[]check;
    delete[]calculated;

    return 0;
}