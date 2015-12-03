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
//�����: threadIdx.x � ������������� ������ � ����� �� ���������� x,
//blockIdx.x � ������������� ����� � ����� �� ���������� x,
//blockDim.x � ���������� ������� � ����� �����.

int main()
{
	cudaEvent_t start, stop;
	float gpuTime=0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// ������� ������� � ����������� ������
	int *h_a,*h_b,*h_c;
	h_a = new int[N*N];
	h_b = new int[N];
	h_c = new int[N];

	for (int i=0;i<N;i++)	 // ������������� �������� a � b
	{ 
		for (int k=0;k<N;k++)
    		{
          		h_a[i*N+k]=1;
    		}
	  	h_b[i]=2;
	}

	// ��������� �� ������� � �����������
	int *d_a,*d_b,*d_c;

	// ��������� �����������
	cudaMalloc((void **)&d_a, sizeof(int)*N*N); 
	cudaMalloc((void **)&d_b, sizeof(int)*N); 
	cudaMalloc((void **)&d_c, sizeof(int)*N); 

	// ����������� �� ����������� ������ � �����������
	cudaMemcpy(d_a, h_a, sizeof(int)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(int)*N, cudaMemcpyHostToDevice);

	// ��������� ���������� ������
	dim3 grid((N+255)/256, 1, 1);
	// ��������� ���������� ������� � �����
	dim3 threads(256, 1, 1);
	//������ ������� �������
	cudaEventRecord(start,0);
	cudaEventSynchronize(start);
	// ����� �������
	MatrVectMul <<< grid, threads >>> (d_c, d_a, d_b);

	 //��������� ������ ����, ��������� �������
	cudaEventRecord(stop,0);
	//�������������� � �������� ��������� ��������
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime,start,stop);
	printf("Time: %.9f msec.\n",gpuTime);

	// ����������� �� ����������� � ����������� ������ 
	cudaMemcpy(h_c, d_c, sizeof(int)*N, cudaMemcpyDeviceToHost);
	
	//for (int i=0;i<N;i++) cout<<h_c[i]<<' ';
	// ������������ ������
	cudaFree(d_a); 
	cudaFree(d_b); 
	cudaFree(d_c);
}
