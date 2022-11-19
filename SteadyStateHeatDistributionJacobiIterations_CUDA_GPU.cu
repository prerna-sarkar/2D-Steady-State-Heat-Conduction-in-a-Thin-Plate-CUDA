/*
CUDA program to determine the steady state heat distribution in a thin metal plate using synchronous iteration on a GPU
*/

#include <stdio.h>
#include<fstream>
#include<iomanip> //precision
#include <unistd.h> //getopt
#include <stdlib.h>  //atoi

void TempDistribution(double*, double*, int, int);

inline cudaError_t HANDLE_ERROR(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

int main(int argc, char *argv[])
{
    int numInteriorPoints=0;
    int numIterations=0;
    int opt;
    while((opt = getopt(argc,argv,"n:I:")) != -1)     // take inputs from command line arguments
    {
        switch(opt)
        {
            case 'n':
                numInteriorPoints = atoi(optarg);
                break;
            case 'I':
                numIterations = atoi(optarg);
                break;
            default:
                printf("ERROR: Usage: [-n numInteriorPoints -I numIterations"); 
                return 1; //exit program
        }
    }
    
    if(numInteriorPoints<=0 || numIterations<=0)
    {
        printf("ERROR: numInteriorPoints and numIterations cannot be <= 0\n");
        return 1;
    }

    int width = numInteriorPoints+2;
    
    int size = width * width * sizeof(double); //allocate num of bytes for the 1D array representing points on our 2D plate
    
    // declare unified memory pointers
    double* H; 
    double* G;
    
    // allocate unified memory – accessible from CPU or GPU
    cudaMallocManaged(&H,size);
    cudaMallocManaged(&G,size);


    // initialise temp at boundaries to values given in the problem statement
    for (int c=0; c<width; c++) //top 
    {
        if(c<round(width*0.3)||c>=round(width*0.7))
        {
            H[0 * width + c] = 20.0;
            G[0 * width + c] = 20.0;
        }
        else
        {
            H[0 * width + c] = 100.0;
            G[0 * width + c] = 100.0;
        }
    }
    
    for (int c=0; c<width; c++) //bottom
    {
        H[((width-1) * width) + c] = 20.0;
        G[((width-1) * width) + c] = 20.0;
    }
    
    for (int r=0; r< width; r++) //left
    {
        H[((r) * width) + 0] = 20.0;
        G[((r) * width) + 0] = 20.0;
    }
    
    for (int r=1; r< width; r++) //right
    {
        H[((r) * width) + (width-1)] = 20.0;
        G[((r) * width) + (width-1)] = 20.0;
    }

    for (int r = 1; r < width-1; r++)   // initialise temp at interior points to zero
    {
        for (int c = 1; c < width-1; c++)
        {
            H[r * width + c] = 0;
            G[r * width + c] = 0;
        }
    }
    
    TempDistribution(H, G, width, numIterations);

    // Free memory
    cudaFree(H);
    cudaFree(G);
    
    return 0;
}


__global__ void GvalueCalculate(double* H, double* G, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x ;  // Calculate the column index of the G element, denote by x
    int y = blockIdx.y * blockDim.y + threadIdx.y ; // Calculate the row index of the G element, denote by y
    
    // each thread computes one element
    if(x>0 && y>0 && x<(width-1) && y<(width-1)) // check if thread lies within the plate's interior points region
    {
        int index = y*width + x; //(row number*length of row) + column number
        int left = y*width + (x-1);
        int right = y*width + (x+1);
        int up = (y-1)*width + x;
        int down = (y+1)*width + x;
        
        G[index] = 0.25*(H[up] + H[down] + H[left] + H[right]);
    }

}


void TempDistribution(double* H, double* G, int width, int numIterations)
{
    // capture start time
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));;

    // kernel invocation code
    dim3 dimBlock(32, 32); // 32*32 = 1024 threads per block
    dim3 dimGrid((width-2)/32 + 1, (width-2)/32 + 1); // blocks per grid
   
   
    for (int i=1; i<(numIterations/2)+1; i++)
    {
        GvalueCalculate << <dimGrid, dimBlock>> > (H, G, width);
        cudaDeviceSynchronize();
        GvalueCalculate << <dimGrid, dimBlock>> > (G, H, width); //swapping G and H in kernel call to avoid copying within kernel
        cudaDeviceSynchronize();
    }

    std::ofstream myOutFile("finalTemperatures.csv"); //output to .csv file

    for (int r = 0; r < width; r++)
    {
        for (int c = 0; c < width; c++)
        {
            myOutFile << std::setprecision(5) <<H[r * width + c];
            if (c<width-1)
                myOutFile << ",";
        }
        myOutFile << "\n";
    }

    // get stop time, and display the timing results
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float   elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Thin plate calculation took:  %3.1f milliseconds\n", elapsedTime);

    // destroy events to free memory
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
}
