#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <vector>
using namespace std;

#define CSC(call)  									\
do {											\
	cudaError_t res = call;							\
	if (res != cudaSuccess) {							\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								\
	}										\
} while(0)

struct comparator {												
	__host__ __device__ bool operator()(double a, double b) {		// Функция которая сравнивает объекты на "<"
		return fabs(a) < fabs(b); 									// operator() - переопределение оператора "()" для экземпляра этой структуры
	}
};

__global__ void down(double* matrix, int n, int m, int k, int row, int column) {
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    for (int j = column + 1 + idy; j < m + k; j += offsety) {
        for (int i = row + 1 + idx; i < n; i += offsetx) {
            matrix[i + n * j] -= matrix[n * column + i] / matrix[n * column + row] * matrix[n * j + row];
        }
    }
}

__global__ void up(double* matrix, int n, int m, int k, int row, int column) {
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    for (int j = m + idy; j < m + k; j += offsety) {
        for (int i = idx; i < row; i += offsetx) {
            matrix[i + n * j] -= matrix[n * column + i] / matrix[n * column + row] * matrix[n * j + row];
        }
    }    
}

__global__ void swap(double* matrix, int n, int m, int k, int i, int i_max) {
    int offsetx = blockDim.x * gridDim.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int j = idx; j < m + k; j += offsetx){
        double tmp = matrix[n * j + i];
        matrix[n * j + i] = matrix[n * j + i_max];
        matrix[n * j + i_max] = tmp;
    }
}

int main() {
    int n, m, k;
    ios_base::sync_with_stdio(0);
    cin.tie(0);

    cin >> n >> m >> k;

    double* host_matrix = (double*) malloc(sizeof(double) * (n * m + n * k));
    double* X = (double*) malloc(sizeof(double) * (m * k));

    for(int i = 0; i < n; ++i)
        for(int j = 0; j < m; ++j)
            cin >> host_matrix[j * n + i];
    

    for(int i = 0; i < n; ++i)
        for(int j = 0; j < k; ++j)
            cin >> host_matrix[(j + m) * n + i];
    
    for(int i = 0; i < m * k; ++i)
        X[i] = 0;

    double* dev_matrix;
    CSC(cudaMalloc(&dev_matrix, sizeof(double) * (n * m + n * k)));
    CSC(cudaMemcpy(dev_matrix, host_matrix, sizeof(double) * (n * m + n * k), cudaMemcpyHostToDevice));

    vector< pair<int, int> > pos;
    comparator comp;
    thrust::device_ptr<double> p_matrix, p_max;
    int i = 0, j = 0;
    int p_index;
    double max;

    while(i < n && j < m){
        p_matrix = thrust::device_pointer_cast(dev_matrix + j * n);
        p_max = thrust::max_element(p_matrix + i, p_matrix + n, comp);
        p_index = p_max - p_matrix;     
        
        CSC(cudaMemcpy(&max, thrust::raw_pointer_cast(p_max), sizeof(double), cudaMemcpyDeviceToHost));

        // max is zero
        if(abs(max) < 1e-7) {
            j += 1;
            continue;
        }
        pos.push_back(make_pair(i, j));

        if(i >= n - 1)
            break;
        
        if(p_index != i)
            swap<<<32, 32>>>(dev_matrix, n, m, k, i, p_index);            

        down<<<dim3(32, 32), dim3(32, 32)>>>(dev_matrix, n, m, k, i, j);
        
        i += 1;
        j += 1;
    }

    for (int i = pos.size() - 1; i >= 0; --i)
        up<<<dim3(32, 32), dim3(32, 32)>>>(dev_matrix, n, m, k, pos[i].first, pos[i].second);

    CSC(cudaMemcpy(host_matrix, dev_matrix, sizeof(double) * (n * m + n * k), cudaMemcpyDeviceToHost));
    
    // normalization
    for(int j = 0; j < k; ++j){
        for(int i = pos.size() - 1; i >= 0; --i)
            X[j * m + pos[i].second] = host_matrix[n * (m + j) + pos[i].first] / host_matrix[n * pos[i].second + pos[i].first];
    }
    
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < k; ++j)
            printf("%.10e ", X[j * m + i]);
        printf("\n");
    }
    
    CSC(cudaFree(dev_matrix));
    free(X);
    free(host_matrix);
    return 0;

}
