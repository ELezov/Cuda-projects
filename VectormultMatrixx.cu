#include <iostream>
#include <stdio.h>
#include <cuda.h>
#define N 15000

using namespace std;
__global__ void MatrVectMul(int *d_c, int *d_a, int *d_b)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
    		d_c[i]=0;
    		for (int k=0;k<N;k++)
    			d_c[i]+=d_a[i+k*N]*d_b[k];
	}
}
//Здесь: threadIdx.x – идентификатор потока в блоке по координате x,
//blockIdx.x – идентификатор блока в гриде по координате x,
//blockDim.x – количество потоков в одном блоке.

int main()
{
	cudaEvent_t start, stop;
	float gpuTime=0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// обычные массивы в оперативной памяти
	int *h_a,*h_b,*h_c;
	h_a = new int[N*N];
	h_b = new int[N];
	h_c = new int[N];

	for (int i=0;i<N;i++)	 // инициализация массивов a и b
	{ 
		for (int k=0;k<N;k++)
    		{
          		h_a[i*N+k]=1;
    		}
	  	h_b[i]=2;
	}

	// указатели на массивы в видеопамяти
	int *d_a,*d_b,*d_c;

	// выделение видеопамяти
	cudaMalloc((void **)&d_a, sizeof(int)*N*N); 
	cudaMalloc((void **)&d_b, sizeof(int)*N); 
	cudaMalloc((void **)&d_c, sizeof(int)*N); 

	// копирование из оперативной памяти в видеопамять
	cudaMemcpy(d_a, h_a, sizeof(int)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(int)*N, cudaMemcpyHostToDevice);

	// установка количества блоков
	dim3 grid((N+255)/256, 1, 1);
	// установка количества потоков в блоке
	dim3 threads(256, 1, 1);
	//Начать отсчета времени
	cudaEventRecord(start,0);
	cudaEventSynchronize(start);
	// вызов функции
	MatrVectMul <<< grid, threads >>> (d_c, d_a, d_b);

	 //Окончание работы ядра, остановка времени
	cudaEventRecord(stop,0);
	//Синхронизируем с моментом окончания расчетов
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime,start,stop);
	printf("Time: %.9f msec.\n",gpuTime);

	// копирование из видеопамяти в оперативную память 
	cudaMemcpy(h_c, d_c, sizeof(int)*N, cudaMemcpyDeviceToHost);
	
	//for (int i=0;i<N;i++) cout<<h_c[i]<<' ';
	// освобождение памяти
	cudaFree(d_a); 
	cudaFree(d_b); 
	cudaFree(d_c);
}
