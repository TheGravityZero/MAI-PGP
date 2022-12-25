#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void kernel(float *arr, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
//	printf("%d %d %d\n", blockIdx.x, threadIdx.x, idx);
	while (idx < n) {
		arr[idx] = abs(arr[idx]);
		idx += offset;
	}
}

int main() {
	int i, n;
	std::cin >> n;
	float *arr = (float *)malloc(sizeof(float) * n);
	for(i = 0; i < n; i++)
		std::cin >> arr[i];
	float *dev_arr;
	cudaMalloc(&dev_arr, sizeof(float) * n);
	cudaMemcpy(dev_arr, arr, sizeof(float) * n, cudaMemcpyHostToDevice);

	kernel<<<32, 32>>>(dev_arr, n);

	cudaMemcpy(arr, dev_arr, sizeof(float) * n, cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		printf("%.10f ", arr[i]);
	printf("\n");

	cudaFree(dev_arr);
	free(arr);
	return 0;
}

