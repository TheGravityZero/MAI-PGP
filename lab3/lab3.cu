#include <stdio.h>
#include <stdlib.h>
#include <iostream>
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


__constant__ double AVG[32][3];

__global__ void kernel(uchar4* dev_data, int w, int h, int num_classes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	while (idx < w * h){
        double4 tmp;
        double res, maxval;
        uchar4 p = dev_data[idx];
        for(int i = 0; i < num_classes; ++i){
            tmp.x = p.x - AVG[i][0];
            tmp.y = p.y - AVG[i][1];  
            tmp.z = p.z - AVG[i][2];
            res = tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z;
            res = -res;
            if(i == 0){
                maxval = res;
                dev_data[idx].w = 0;
            }
            if(res > maxval){
                maxval = res;
                dev_data[idx].w =  i;
            }
        }
        idx += offset;
	}
}

int main(){
	int w, h;
	string input, output;
	cin >> input;
	cin >> output;
        
    int num_classes, num_pixels;
    cin >> num_classes;
    vector<vector<int2>> classes(num_classes); 
    for(int i = 0; i < num_classes; ++i){
        cin >> num_pixels;
        int2 cord;
        for(int j = 0; j < num_pixels; ++j){
            cin >> cord.x >> cord.y;
            classes[i].push_back(cord);
        }
    }

	FILE* fp = fopen(input.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4* data = (uchar4*) malloc(sizeof(uchar4) * w * h);
	//printf("%d %d", h, w);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	
    // Инициализация avg
    double avg[32][3];
    for (int i = 0; i < 32; ++i){
        avg[i][0] = 0.0;
        avg[i][1] = 0.0;
        avg[i][2] = 0.0;
    }

    // Вычисление avg
    for(int i = 0; i < num_classes; ++i){
        for(int j = 0; j < classes[i].size(); ++j){
            int x = classes[i][j].x;
            int y = classes[i][j].y;
            uchar4 p = data[y * w + x];
            avg[i][0] += p.x;
            avg[i][1] += p.y;
            avg[i][2] += p.z;
        }
        avg[i][0] /= classes[i].size();
        avg[i][1] /= classes[i].size();
        avg[i][2] /= classes[i].size();
    }

    CSC(cudaMemcpyToSymbol(AVG, avg, sizeof(double) * 32 * 3));

	uchar4* dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
	CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
	
    cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start));

	kernel<<<128, 64>>>(dev_data, w, h, num_classes);
	CSC(cudaGetLastError());
	CSC(cudaDeviceSynchronize());

    CSC(cudaEventRecord(stop));
	CSC(cudaEventSynchronize(stop));
	float t;
	CSC(cudaEventElapsedTime(&t, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));
    printf("time = %f ms\n", t);
    
    CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_data));
    //for(int i = 0; i < w * h; ++i){
    //    printf("%d", (int) data[i].w);
    //}
    fp = fopen(output.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);
	free(data);
	return 0;
}
