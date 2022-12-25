#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#define CSC(call)  									\
do {											\
	cudaError_t res = call;							\
	if (res != cudaSuccess) {							\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								\
	}										\
} while(0)


texture<uchar4, 2, cudaReadModeElementType> tex;
__constant__ double coefficients[1025];

__host__ double calculate_coefficients(int r){
	const double PI = acos(-1.0);
    double sum = 0.0, tmp;
	double constant = 1.0 / (sqrt(2.0 * PI) * r);
    double coeffs[1025];
    for (int i = 0; i <= r; ++i){
        tmp = constant * exp((-(double)(i * i)) / (double)(2 * r * r));
        sum += 2 * tmp;   
        coeffs[i] = tmp;
    }
	sum -= constant * exp((-(0.0 * 0.0)) / (2 * r * r)); // Так как два раза учитываем центральный элемент
    for (int i = 0; i <= r; ++i){
        coeffs[i] /= sum;
    }
	CSC(cudaMemcpyToSymbol(coefficients, coeffs, (1025) * sizeof(double)));
	return sum;
}


__global__ void horizontal(uchar4 *out, int w, int h, int r, double sum){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	uchar4 pixel;
	for(y = idy; y < h; y += offsety){
		for(x = idx; x < w; x += offsetx){
			double coeff;
            double4 out_p;
			out_p.x = 0;
			out_p.y = 0;
			out_p.z = 0;
			for (int i = -r; i <= r; ++i) {
                coeff = coefficients[abs(i)];
                int y_ = max(0, min(y, h));
                int x_ = max(0, min(x + i, w));
                
                pixel = tex2D(tex, x_, y_);

                out_p.x += pixel.x * coeff;
                out_p.y += pixel.y * coeff;
                out_p.z += pixel.z * coeff;
            }
			out[y * w + x] = make_uchar4(out_p.x, out_p.y, out_p.z, 255);
		}
	}
}

__global__ void vertical(uchar4 *out, int w, int h, int r, double sum){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	uchar4 pixel;
	for(y = idy; y < h; y += offsety){
		for(x = idx; x < w; x += offsetx) {
			double coeff;
            double4 out_p;
			out_p.x = 0;
			out_p.y = 0;
			out_p.z = 0;
			for (int i = -r; i <= r; ++i) {
                coeff = coefficients[abs(i)];
                int y_ = max(0, min(y + i, h));
                int x_ = max(0, min(x, w));
                
                pixel = tex2D(tex, x_, y_);

                out_p.x += pixel.x * coeff;
                out_p.y += pixel.y * coeff;
                out_p.z += pixel.z * coeff;
            }
			out[y * w + x] = make_uchar4(out_p.x, out_p.y, out_p.z, 255);

		}
	}
}

using namespace std;
int main() {
	int w, h, r;
	string input, output;
	cin >> input;
	cin >> output;
	cin >> r;
	
	FILE *fp = fopen(input.c_str(), "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	//printf("%d %d", h, w);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);
	//printf("%d ", data[500].w);
	if (r > 0){
		cudaArray *arr;
		cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
		CSC(cudaMallocArray(&arr, &ch, w, h));
		CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

		tex.normalized = false;
		tex.filterMode = cudaFilterModePoint;	
		tex.channelDesc = ch;
		tex.addressMode[0] = cudaAddressModeClamp;
		tex.addressMode[1] = cudaAddressModeClamp;
		double sum = calculate_coefficients(r);
		CSC(cudaBindTextureToArray(tex, arr, ch));

		uchar4 *dev_out;
		CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

		horizontal<<< dim3(16, 16), dim3(32, 32) >>>(dev_out, w, h, r, sum);
		CSC(cudaGetLastError());
		CSC(cudaDeviceSynchronize());
		CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
		
		CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));
        vertical<<<dim3(16, 16), dim3(32, 32)>>>(dev_out, w, h, r, sum);
		CSC(cudaGetLastError());
		CSC(cudaDeviceSynchronize());
        CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
		
		CSC(cudaUnbindTexture(tex));
		CSC(cudaFreeArray(arr));
		CSC(cudaFree(dev_out));
	}
	
	fp = fopen(output.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);
	free(data);
	return 0;
}
