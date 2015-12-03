#include "vector.h"

__global__ void kernal1(int *A, int *B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
		B[i] = A[i]+B[i];
}

__global__ void kernal2(int *A, int k, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
		A[i] = A[i]*k;
}

__host__ int my_vector::get_size()
{
	return len;
}

__host__ my_vector my_vector::summa(my_vector a, my_vector b)
{
	if (a.len!=b.len)
	{
		printf("error summa\n");
		this->len=0;
		return *this;
	}
	size_t size = a.len * sizeof(int);
	this->len=a.len;
	int i;
	//iniziliaze host array
	int *h_A = (int *)malloc(size);
	int *h_B = (int *)malloc(size);
	for (i=0; i<a.len; ++i) h_A[i]=a.X[i];
	for (i=0; i<a.len; ++i) h_B[i]=b.X[i];
	
	//inizialiaze global aray
	int *d_A = NULL;
	int *d_B = NULL;
	cudaError err = cudaSuccess;
	err = cudaMalloc((void **)&d_A, size);
	if (err != cudaSuccess){printf("malloc A error\n");exit(0);}
	err = cudaMalloc((void **)&d_B, size);
	if (err != cudaSuccess){printf("malloc B error\n");exit(0);}
	
	//Copy vector A and vector B
	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){printf("copy error\n"); exit(0);}
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){printf("copy error\n"); exit(0);}
	
	kernal1<<<a.len,1>>>(d_A, d_B, a.len);
	err = cudaGetLastError();

	if (err != cudaSuccess){printf("kernal1 error\n"); exit(0);}

	err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){printf("copy error\n"); exit(0);}

	// Free device global memory
	err = cudaFree(d_A);
	err = cudaFree(d_B);

	// Free host memory
	for(i=0; i<a.len; ++i) this->X[i]=h_B[i];

	free(h_A);
	free(h_B);
	
	return *this;
}

__host__ my_vector my_vector::mult (int k, my_vector a)
{
	this->len=a.len;
	size_t size = a.len * sizeof(int);
	this->len=a.len;
	int i;
	//iniziliaze host array
	int *h_A = (int *)malloc(size);
	for (i=0; i<a.len; ++i) h_A[i]=a.X[i];
	
	//inizialiaze global aray
	int *d_A = NULL;
	cudaError err = cudaSuccess;
	err = cudaMalloc((void **)&d_A, size);
	if (err != cudaSuccess){printf("malloc A error\n");exit(0);}
	
	//Copy vector A and vector B
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){printf("copy error\n"); exit(0);}
	
	kernal2<<<a.len,1>>>(d_A, k, a.len);
	err = cudaGetLastError();

	if (err != cudaSuccess){printf("kernal2 error\n"); exit(0);}

	err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){printf("copy error\n"); exit(0);}

	// Free device global memory
	err = cudaFree(d_A);

	// Free host memory
	for(i=0; i<a.len; ++i) this->X[i]=h_A[i];

	free(h_A);
	return *this;
}

__host__ void my_vector::write ()
{
    int i;
	for (i=0; i<len; ++i) printf("%d ",X[i]);
	printf("\n");
}
