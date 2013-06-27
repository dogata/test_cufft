/*
Test script using cuFFT library for 2D convolution
-- Adapted from sample cuFFT script for complex-to-real transform
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>  // these are in /usr/local/cuda/samples/common/inc
#include <helper_cuda.h>

// macro
#define NELEMS(x)  (sizeof(x) / sizeof(x[0]))

// Types
typedef double2 Complex;
typedef double Real;
typedef struct kval
{
  int kx;
  int ky;
}Kval;

// Global variables
static const int KXMAX = 40, KYMAX = KXMAX;
static const int NX = 3*KXMAX+1, NY = NX,
  NREAL = NX*NY;
static const int NXCMPLX = NX, NYCMPLX = NY/2+1,
  NCMPLX = NXCMPLX*NYCMPLX;
static const int mem_csize = sizeof(Complex)*NCMPLX;
static const int mem_rsize = sizeof(Real)*NREAL;
Kval *kindex;
cufftHandle planb, planf;
static const int blockSize = 32;
static const int nBlocks = NREAL/blockSize + (NREAL%blockSize == 0?0:1);

// timing variables
int msec;
clock_t start, diff;

// Function declaration
void runTest(int argc, char **argv);
void initFFT(int argc, char **argv);
void finalFFT(int argc, char **argv);
void convolveFFT(int argc, char **argv);
void makeIndex(Kval *kindex_out);
int getNindex(const int kx, const int ky);
int getKXindex(const int n);
int getKYindex(const int n);
void makeData(Complex *h_data, const Kval *kindex);
void complexScale(Complex *h_data, const Real scale);
Complex complexConjg(const Complex a);
Complex complexMul(const Complex a, const Complex b);
void convolveIter(const Complex *a, const Complex *b, Complex *c);
static __global__ void RealPointwiseMul(Real *a, const Real *b, const int size, const Real scale);


// --> MAIN <--
int main(int argc, char **argv)
{


  
  printf("\n Initalize FFT \n");
  initFFT(argc, argv);

  start = clock();
  printf("\n Run FFT test \n");
  runTest(argc, argv);

  diff = clock() - start;
  msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

  printf("\n Run Convolution FFT test \n");
  convolveFFT(argc, argv);
  

}

// initFFT -- initialize index arrays
void initFFT(int argc, char **argv)
{
  FILE *ptr_file, *ptr_file1;
  unsigned int n;

  findCudaDevice(argc, (const char **)argv);

  // Test complex type
  printf("KXMAX = %i, KYMAX = %i\n",KXMAX,KYMAX);
  printf("NX = %i, NY = %i, NREAL = %i\n", NX, NY, NREAL);
  printf("NXCMPLX = %i, NYCMPLX = %i, NCMPLX = %i\n", NXCMPLX, NYCMPLX, NCMPLX);
  kindex = (Kval *)malloc(sizeof(Kval) * NCMPLX);

  // Initialize index array
  makeIndex(kindex);

  // Output k-index:
  ptr_file = fopen("kindex.out","w");
  ptr_file1 = fopen("index.out","w");
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*int(NYCMPLX);
	  fprintf(ptr_file,"kindex[%i].kx = %i, kindex[%i].ky = %i\n", n,kindex[n].kx,n,kindex[n].ky);
	  fprintf(ptr_file1,"kindex[%i].kx = %i, kindex[%i].ky = %i, n = %i\n",
		  n,getKXindex(n),n,getKYindex(n),getNindex(getKXindex(n),getKYindex(n)));
	}
    }
  fclose(ptr_file);
  fclose(ptr_file1);

  // Create FFT plans
  checkCudaErrors(cufftPlan2d(&planb, NX, NY, CUFFT_Z2D));
  checkCudaErrors(cufftPlan2d(&planf, NX, NY, CUFFT_D2Z));
  checkCudaErrors(cufftSetCompatibilityMode(planb, CUFFT_COMPATIBILITY_NATIVE));
  checkCudaErrors(cufftSetCompatibilityMode(planf, CUFFT_COMPATIBILITY_NATIVE));

}

// runTest
void runTest(int argc, char **argv)
{
  unsigned int n;
  FILE *ptr_file;

  // Allocate host arrays
  Complex *h_data = (Complex *)malloc(mem_csize);
  Complex *h_data1 = (Complex *)malloc(mem_csize);
  Real *h_rdata = (Real *)malloc(mem_rsize);
  
  // Initalize the memory for the signal, making Hermitian data
  makeData(h_data, kindex);

  //printf("\nInitial data before FFT:\n");
  ptr_file = fopen("data_init.out","w");
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*int(NYCMPLX);
	  /*printf("kindex[%i].kx = %i, kindex[%i].ky = %i, x = %g, y = %g\n", 
	    n,kindex[n].kx,n,kindex[n].ky,h_data[n].x,h_data[n].y);*/
	  fprintf(ptr_file,"(%g, %g) ",h_data[n].x,h_data[n].y);
	}
      fprintf(ptr_file,"\n");
    }
  fclose(ptr_file);
  //printf("Initial data written.\n");

  // Allocate device data arrays
  cufftDoubleComplex *d_data, *d_data1;
  cufftDoubleReal *d_rdata;
  checkCudaErrors(cudaMalloc((void**)&d_data,mem_csize));
  checkCudaErrors(cudaMalloc((void**)&d_data1,mem_csize));
  checkCudaErrors(cudaMalloc((void**)&d_rdata,mem_rsize));

  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(d_data, h_data, mem_csize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(h_data1, d_data, mem_csize, cudaMemcpyDeviceToHost));

  //printf("\nCheck memory copy to GPU before FFT:\n");
  ptr_file = fopen("data_init_check.out","w");
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*int(NYCMPLX);
	  fprintf(ptr_file,"(%g, %g) ",h_data1[n].x,h_data1[n].y);
	}
      fprintf(ptr_file,"\n");
    }
  fclose(ptr_file);

  //printf("Initial data check written.\n");

  // Perform backward FFT
  checkCudaErrors(cufftExecZ2D(planb, (cufftDoubleComplex *)d_data, (cufftDoubleReal *)d_rdata));

  // Perform forward FFT
  checkCudaErrors(cufftExecD2Z(planf, (cufftDoubleReal *)d_rdata, (cufftDoubleComplex *)d_data));
  
  // Copy device memory to host
  checkCudaErrors(cudaMemcpy(h_data1, d_data, mem_csize, cudaMemcpyDeviceToHost));

  //printf("\nAfter FFT:\n");
  ptr_file = fopen("data_final.out","w");
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*int(NYCMPLX);
	  h_data1[n].x = h_data1[n].x/double(NREAL);
	  h_data1[n].y = h_data1[n].y/double(NREAL);
	  fprintf(ptr_file,"(%g, %g) ",h_data1[n].x,h_data1[n].y);
	}
      fprintf(ptr_file,"\n");
    }


  //printf("\n Errors: \n");
  ptr_file = fopen("data_diff.out","w");
    for (unsigned int i = 0; i < NXCMPLX; ++i)
      {
	for (unsigned int j = 0; j < NYCMPLX; ++j)
	  {
	    n = j + i*int(NYCMPLX);
	    fprintf(ptr_file,"(%g, %g) ",abs(h_data[n].x-h_data1[n].x),abs(h_data[n].y-h_data1[n].y));
	  }
	fprintf(ptr_file,"\n");
      }
  fclose(ptr_file);

  // Deallocate memory
  free((void *)h_data); free((void *)h_data1); free((void *)h_rdata);

  cudaFree(d_data); cudaFree(d_data1);
  cudaFree(d_rdata);

}


// convolveFFT -- convolution using FFT
void convolveFFT(int argc, char **argv)
{
  unsigned int n;
  FILE *ptr_file;

  // Allocate host arrays
  Complex *h_data_a = (Complex *)malloc(mem_csize);
  Complex *h_data_b = (Complex *)malloc(mem_csize);
  
  // Initalize the memory for the signal, making Hermitian data
  makeData(h_data_a, kindex);
  makeData(h_data_b, kindex);

  ptr_file = fopen("conv_init.out","w");
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*int(NYCMPLX);
	  /*printf("kindex[%i].kx = %i, kindex[%i].ky = %i, x = %g, y = %g\n", 
	    n,kindex[n].kx,n,kindex[n].ky,h_data[n].x,h_data[n].y);*/
	  fprintf(ptr_file,"(%g, %g) ",h_data_a[n].x,h_data_a[n].y);
	}
      fprintf(ptr_file,"\n");
    }
  fclose(ptr_file);

  start = clock();
  // Calculates convolution on host without FFT
  Complex *h_data1 = (Complex *)malloc(mem_csize);
  printf("\nPerform convolution without FFT\n");
  convolveIter(h_data_a,h_data_b,h_data1);

  diff = clock() - start;
  msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

  // output convolution data
  ptr_file = fopen("conv_iter.out","w");
    for (unsigned int i = 0; i < NXCMPLX; ++i)
      {
	for (unsigned int j = 0; j < NYCMPLX; ++j)
	  {
	    n = j + i*int(NYCMPLX);
	    fprintf(ptr_file,"(%g, %g) ",h_data1[n].x,h_data1[n].y);
	  }
	fprintf(ptr_file,"\n");
      }
  fclose(ptr_file);

  // Allocate device data arrays
  cufftDoubleComplex *d_data_a, *d_data_b;
  cufftDoubleReal *d_rdata_a, *d_rdata_b;
  checkCudaErrors(cudaMalloc((void**)&d_data_a,mem_csize));
  checkCudaErrors(cudaMalloc((void**)&d_data_b,mem_csize));
  checkCudaErrors(cudaMalloc((void**)&d_rdata_a,mem_rsize));
  checkCudaErrors(cudaMalloc((void**)&d_rdata_b,mem_rsize));

  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(d_data_a, h_data_a, mem_csize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_data_b, h_data_b, mem_csize, cudaMemcpyHostToDevice));

  /*
  checkCudaErrors(cudaMemcpy(h_data1, d_data_a, mem_csize, cudaMemcpyDeviceToHost));
  ptr_file = fopen("conv_init_check.out","w");
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*int(NYCMPLX);
	  fprintf(ptr_file,"(%g, %g) ",h_data1[n].x,h_data1[n].y);
	}
      fprintf(ptr_file,"\n");
    }
  fclose(ptr_file);
  */
  

  // Perform backward FFT
  checkCudaErrors(cufftExecZ2D(planb, (cufftDoubleComplex *)d_data_a, (cufftDoubleReal *)d_rdata_a));
  checkCudaErrors(cufftExecZ2D(planb, (cufftDoubleComplex *)d_data_b, (cufftDoubleReal *)d_rdata_b));

  start = clock();
  // Convolution on device
  printf("\nPerform convolution with FFT\n");
  printf("blockSize = %i, nBlocks = %i\n",blockSize,nBlocks);
  RealPointwiseMul<<<blockSize, nBlocks>>>(d_rdata_a, d_rdata_b, NREAL, double(1.0/NREAL));
  getLastCudaError("Kernel execution failed [ RealPointwiseMul ]");

  // Perform forward FFT
  cudaMemset(d_data_a, 0.0, mem_csize);
  checkCudaErrors(cufftExecD2Z(planf, (cufftDoubleReal *)d_rdata_a, (cufftDoubleComplex *)d_data_a));
  
  // Copy device memory to host
  memset(h_data_a, 0.0, sizeof(h_data_a[0]) * NCMPLX);
  checkCudaErrors(cudaMemcpy(h_data_a, d_data_a, mem_csize, cudaMemcpyDeviceToHost));

  diff = clock() - start;
  msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

  // Zero out the values at higher modes
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*int(NYCMPLX);
	  if ( (abs(getKXindex(n)) > KXMAX) || (abs(getKYindex(n)) > KYMAX) )
	    {
	      h_data_a[n].x = 0.0; h_data_a[n].y = 0.0;
	    }
	}
    }

  ptr_file = fopen("conv.out","w");
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*int(NYCMPLX);
	  fprintf(ptr_file,"(%g, %g) ",h_data_a[n].x,h_data_a[n].y);
	}
      fprintf(ptr_file,"\n");
    }
  fclose(ptr_file);

 

  // check differences
  Real xdiff, ydiff;
  Complex sumdiff;
  sumdiff.x = 0.0; sumdiff.y = 0.0;
  ptr_file = fopen("conv_diff.out","w");
    for (unsigned int i = 0; i < NXCMPLX; ++i)
      {
	for (unsigned int j = 0; j < NYCMPLX; ++j)
	  {
	    n = j + i*int(NYCMPLX);
	    xdiff = abs(h_data_a[n].x-h_data1[n].x);
	    ydiff = abs(h_data_a[n].y-h_data1[n].y);
	    sumdiff.x = sumdiff.x + xdiff; sumdiff.y = sumdiff.y + ydiff;
	    fprintf(ptr_file,"(%g, %g) ",xdiff,ydiff);
	  }
	fprintf(ptr_file,"\n");
      }
  fclose(ptr_file);

  sumdiff.x = sumdiff.x/double(NXCMPLX); sumdiff.y = sumdiff.y/double(NYCMPLX);
  printf("\nAverage differences: (%g, %g)\n\n",sumdiff.x,sumdiff.y);

  // Deallocate memory
  free((void *)h_data_a); free((void *)h_data_b); free((void *)h_data1);

  cudaFree(d_data_a); cudaFree(d_data_b);
  cudaFree(d_rdata_a); cudaFree(d_rdata_b);
}



/* Routines to initialize index arrays
   kindex.kx, kindex.ky: mapping n --> (kx,ky)
   kindex.n: mapping (kx,ky) --> n
*/
void makeIndex(Kval *kindex_out)
{

  unsigned int n;
  memset(kindex_out, 0, sizeof(kindex_out[0]) * NCMPLX);

  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*int(NYCMPLX);
	  if (j <= KYMAX)
	    {
	      if (i <= KXMAX)
		{
		  kindex_out[n].kx = i;
		  kindex_out[n].ky = j;
		}
	      else
		{
		  if (i >= (NXCMPLX-KXMAX))
		    {
		      kindex_out[n].kx = i - NXCMPLX;
		      kindex_out[n].ky = j;
		    }
		}
	    }
	}
    }


}

// Create a radom set of 2D Hermitian data
void makeData(Complex *h_data, const Kval *kindex_in)
{
  int kx;
  unsigned int n,m;
  memset(h_data, 0.0, sizeof(h_data[0]) * NCMPLX);

  srand(time(NULL));  // generate seed
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*NYCMPLX;
	  if (j <= KYMAX)
	    {
	      if (i <= KXMAX)
		{
		  h_data[n].x = rand() / float(RAND_MAX);
		  h_data[n].y = rand() / float(RAND_MAX);
		}
	      else 
		{
		  if (i >= (NXCMPLX-KXMAX) )
		    {
		      kx = getKXindex(n);
		      if ((kx < 0) && (j == 0))
			{
			  //m = NYCMPLX*(-kindex[n].kx);  // reference to conjugate modes
			  m = getNindex(-kx,0);
			  h_data[n].x = h_data[m].x;  // this needs to be conjugate modes
			  h_data[n].y = -h_data[m].y;
			}
		      else
			{
			  h_data[n].x = rand() / float(RAND_MAX);
			  h_data[n].y = rand() / float(RAND_MAX);
			}
		    }
		}
	    }
	}
    }
  h_data[0].x = 0; h_data[0].y = 0; // enforces the (0,0) element
}


// finalFFT -- clean up, destroy cuFFT plans
void finalFFT(int argc, char **argv)
{
    cufftDestroy(planf); cufftDestroy(planb);
}


// Real pointwise multiplication -- multiplies arrays element-wise on device
//   and scale according to the normalization for FFT
static __global__ void RealPointwiseMul(Real *a, const Real *b, const int size, const Real scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
      a[i] = a[i]*b[i]*scale;
      //a[i] = a[i]*scale;
    }
}


// complexScale -- multiplies each element in the cmplex array
void complexScale(Complex *h_data, const Real scale)
{
  unsigned int n;
  for (unsigned int i = 0; i < NXCMPLX; ++i)
    {
      for (unsigned int j = 0; j < NYCMPLX; ++j)
	{
	  n = j + i*NYCMPLX;
	  h_data[n].x = h_data[n].x/scale;
	  h_data[n].y = h_data[n].y/scale;
	}
    }
}

// convolveIter -- convolution by iterations
void convolveIter(const Complex *a, const Complex *b, Complex *c)
{
  int kxd, kyd, m, p;
  Complex var1temp, var2temp;

  memset(c, 0.0, sizeof(c[0]) * NCMPLX);

  for (int i = -KXMAX; i <= KXMAX; ++i)
    {
      for (int j = 0; j <= KYMAX; ++j)
	{
	  p = getNindex(i,j);
	  for (int ix = -KXMAX; ix <= KXMAX; ++ix)
	    {
	      for (int iy = -KYMAX; iy <= KYMAX; ++iy)
		{
		  kxd = i - ix; kyd = j - iy;

		  // var1temp
		  var1temp.x = 0.0; var1temp.y = 0.0;
		  if ((abs(kxd) > KXMAX) || (abs(kyd) > KYMAX))
		    {
		      var1temp.x = 0; var1temp.y = 0;
		    }
		  else
		    {
		      if (kyd < 0)
			{
			  m = getNindex(-kxd,-kyd);
			  //var1temp = complexConjg(a[m]);
			  var1temp.x = a[m].x; var1temp.y = -a[m].y;
			}
		      else
			{
			  m = getNindex(kxd,kyd);
			  var1temp.x = a[m].x; var1temp.y = a[m].y;
			}
		    }

		  // var2temp
		  var2temp.x = 0.0; var2temp.y = 0.0;
		  if (iy < 0)
		    {
		      m = getNindex(-ix,-iy);
		      //var2temp = complexConjg(a[m]);
		      var2temp.x = b[m].x; var2temp.y = -b[m].y;
		    }
		  else
		    {
		      m = getNindex(ix,iy);
		      var2temp.x = b[m].x; var2temp.y = b[m].y;
		    }

		  // sum up the terms
		  c[p].x = c[p].x + (var1temp.x*var2temp.x - var1temp.y*var2temp.y);
		  c[p].y = c[p].y + (var1temp.x*var2temp.y + var1temp.y*var2temp.x);

		}
	    }
	  
	}
    }
}

// getNindex -- find the number of element by inputting the modes
int getNindex(const int kx, const int ky)
{
  int i,j,n;
  if (kx >= 0)
    {
      i = kx;
    }
  else
    {
      i = NXCMPLX+kx;
    }
  j = ky;
  n = j + i*NYCMPLX;
  return n;
}


// getKXindex -- get kx mode by specifying element number
int getKXindex(const int n)
{
  int kx = 0,i;
  i = n/NYCMPLX;
  if (i <= NXCMPLX/2)
    {
      kx = i;
    }
  else
    {
      kx = i - NXCMPLX;
    }
  return kx;
}


// getKYindex -- get ky mode by specifying element number
int getKYindex(const int n)
{
  int ky = 0;
  ky = int(n%NYCMPLX);
  return ky;
}


// complexConjg -- compute the complex conjugate of a number
Complex complexConjg(const Complex a)
{
  Complex b;
  b.x = a.x;
  b.y = -a.y;
  return b;
}

// complexMul -- compute complex multiplication
Complex complexMul(const Complex a, const Complex b)
{
  Complex c;
  c.x = a.x*b.x - a.y*b.y;
  c.y = a.x*b.y + a.y*b.x;
  return c;
}
