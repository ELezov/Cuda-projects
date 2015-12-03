#ifdef _WIN32
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#endif
#include <stdio.h>
#include <iostream>
#include <fstream>
#define threadsPerBlock 512
#define blocksPerGrid 10
using namespace std;

__global__ void cuda_multi_matrix_on_vector(int *matrix, int *vector, int *new_vector, int numElements){
	__shared__ int cache[threadsPerBlock];
	const int idx = blockDim.x*blockIdx.x + threadIdx.x;//глобальный индекс
	const int tIdx = threadIdx.x;//индекс нити
	const int k = (numElements - 1 + threadsPerBlock) / threadsPerBlock;//всего кол-во блоков
	
	for (int i = 0; i < k; i++){//в блок влезает threadsPerBlock нитей. Чтобы посчитать всю строку на нужно читать кусок вектора k раз
		if (tIdx+threadsPerBlock*i < numElements){//если индекс нити плюс потоковое смещение меньше n то копируем в память shared
			cache[tIdx] = vector[tIdx + threadsPerBlock * i];
		}
		__syncthreads();

		int min = numElements - i*threadsPerBlock;//определяем хвост
		if (min > threadsPerBlock)min = threadsPerBlock;//если хвост слишком длинный то берём по нитям
		if (idx < numElements){
			for (int j= 0; j < min; j++){
				new_vector[idx] += cache[j]*matrix[(i*threadsPerBlock + j)*numElements + idx];//каждая нить считает свой вектор умножая кусок вектора на сообверствующий кусок матрицы
			}
		}
		__syncthreads();
	}
}

int main(int argc, char *argv[]){
	ifstream in("input.txt");
	int numElements=5120;
	//in >> numElements;
	int *mas_matrix = new int[numElements*numElements];
	for (int i = 0; i < numElements; i++){
		for (int j = 0; j < numElements; j++){
			//in >> mas_matrix[j*numElements + i];
			if(i==j) mas_matrix[j*numElements + i]=1;
		}
	}
	int *mas_vector = new int[numElements];
	for (int i = 0; i < numElements; i++){
		mas_vector[i]=i;
		//in >> mas_vector[i];
	}

	//cudaDeviceSynchronize();

	int *d_mas_matrix=NULL;
	cudaError err = cudaSuccess; 

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	//Выделяем память для векторов на видеокарте
	err = cudaMalloc((void**)&d_mas_matrix, numElements*numElements*sizeof(int));
	//&d_mas_matrix — указатель, в который записывается адрес выделенной памяти,
	//numElements*numElements*sizeof(int) — размер выделяемой памяти в байтах.
	//Возвращает: 
	//cudaSuccess — при удачном выделении памяти
	//cudaErrorMemoryAllocation — при ошибке выделения памяти


	if(err!= cudaSuccess){
		cout << "cudaMallocError";
		return 0;
	}
	 //Копируем данные в память видеокарты 
	err = cudaMemcpy(d_mas_matrix, mas_matrix, numElements*numElements*sizeof(int), cudaMemcpyHostToDevice);
	//cudaError_t cudaMemcpy(void* dst, const void* src ,size_t count, enum cudaMemcpyKind kind), где
	//dst — указатель, содержащий адрес места-назначения копирования,
	//src — указатель, содержащий адрес источника копирования,
	//count — размер копируемого ресурса в байтах,
	//cudaMemcpyKind — перечисление, указывающее направление копирования (может быть cudaMemcpyHostToDevice, cudaMemcpyDevice ToHost, cudaMemcpyHostToHost, cudaMemcpyDeviceToDevice).
	//Возвращает:
	//cudaSuccess – при удачном копировании
	//cudaErrorInvalidValue – неверные параметры аргумента (например, размер копирования отрицателен)
	//cudaErrorInvalidDevicePointer – неверный указатель памяти в видеокарте
	//cudaErrorInvalidMemcpyDirection – неверное направление (например, перепутан источник и место-назначение копирования)
	
	if(err != cudaSuccess){
		cout << "cudaMemcpyError";
		return 0;
	}

	int *d_mas_vector;
	//Выделяем память для векторов на видеокарте
	err = cudaMalloc((void**)&d_mas_vector, numElements*sizeof(int));
	if(err != cudaSuccess){
		cout << "cudaMallocError";
		return 0;
	}
	 //Копируем данные в память видеокарты 
	err = cudaMemcpy(d_mas_vector, mas_vector, numElements*sizeof(int), cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		cout << "cudaMemcpyError";
		return 0;
	}
	
	int device_count;
	cudaDeviceProp deviceProp;
	err=cudaGetDeviceCount(&device_count);
	if (err != cudaSuccess){
		cout << "cudaGetDeviceCountError";
	}

	err=cudaGetDeviceProperties(&deviceProp, 0);
	if (err != cudaSuccess){
		cout << "cudaGetDevicePropertiesError";
		return 0;
	}
	if (device_count < 1){
		cout << "cudaDevice<1Error";
		return 0;
	}

	int* d_new_vector;
	//Выделяем память для векторов на видеокарте
	err = cudaMalloc((void**)&d_new_vector, numElements*sizeof(int));
	if (err != cudaSuccess){
		cout << "cudaMallocError";
		return 0;
	}

	
	cudaMemset(d_new_vector, 0, sizeof(int)*numElements);

	dim3 block_dim(threadsPerBlock);
	dim3 grid_dim((numElements + block_dim.x - 1) / block_dim.x);
	
	cudaEventRecord(start,0);

	grid_dim=blocksPerGrid;
	block_dim=threadsPerBlock;
	cuda_multi_matrix_on_vector << <grid_dim,block_dim >> >(d_mas_matrix, d_mas_vector, d_new_vector, numElements);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	
	ofstream out("output.txt");
	
	err = cudaMemcpy(mas_vector, d_new_vector, numElements*sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		cout << "cudaMemcpyDTHError";
		return 0;
	}

	for (int i = 0; i < numElements; i++){
		//out << mas_vector[i] << " ";
		printf("%d ",mas_vector[i]);
		printf("\n");
	}

	float Time=0.0f;
	cudaEventElapsedTime(&Time,start,stop);
	
	printf("Time: %.2f msec.\n",Time);

	err = cudaFree(d_mas_matrix);

	free(mas_vector);
	
	system("pause");
	return 0;//возвращает управление
}
