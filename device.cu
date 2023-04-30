#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RGB.h"
#include <cmath>

/**
* Helper function to calculate the greyscale value based on R, G, and B (Dr. Mao)
*/
__device__ int greyscale(BYTE red, BYTE green, BYTE blue)
{
	int grey = 0.3 * red + 0.59 * green + 0 * 11 * blue; // calculate grey scale
	return min(grey, 255);
}

/**
* Kernel for executing on GPY (Dr. Mao)
*/
__global__ void greyscaleKernel(RGB* d_pixels, int height, int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (y >= height || y >= width)
		return;

	int index = y * width + x; // index == (x,y)

	int grey = greyscale(d_pixels[index].red, d_pixels[index].green, d_pixels[index].blue); // calculate grey scale

	d_pixels[index].red = grey;
	d_pixels[index].green = grey;
	d_pixels[index].blue = grey;
}


/**
* Averaging function kernel on GPU (Kaelan)
*/
__global__ void averageKernel(RGB* d_pixels, int height, int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; //width
	int y = blockIdx.y * blockDim.y + threadIdx.y; //height

	if (y >= height || y >= width)
		return;

	double total_red = 0, total_green = 0, total_blue = 0;
	int w = y + 1;							// w, a, s, d are all variables that represent coordinates of pixels offset from the index, they are defined to keep the program from exiting the bounds
	if (w > height || w < 0) {				// of the image. This will affect the average Ex. 
		w = y;								// Normal: 1 + 2 + 3 + 4 + 5 = 15 / 5 = 3	// somewhere in middle of image if 1-5 were assigned from top to bottom, left to right
	}										// New: 3 + 3 + 3 + 4 + 5 = 18 / 5 = 3.6 //Example of top left if 1-5 were assigned from top to bottom, left to right
	int a = y - 1;
	if (a > height || a < 0) {				// Current situation: Lena works, Sierra and marbles throw an exception. Lena is a pure white image but is a new result file.
		a = y;								// When a .red, .green, or .blue is taken away, the image is visible without those colors
	}
	int s = x - 1;
	if (s > width || s < 0) {
		s = x;
	}
	int d = x + 1;
	if (d > width || d < 0) {
		d = x;
	}

	int index = y * width + x; // compute index position of (y,x) coordinate, middle of star
	int component2 = (w)* width + x;				//top middle of star
	int component3 = y * width + (s);			//left of star		
	int component4 = y * width + (d);			//right of star
	int component5 = (a)* width + x;

	total_red = d_pixels[index].red + d_pixels[component2].red + d_pixels[component3].red + d_pixels[component4].red + d_pixels[component5].red; // add the red value at this pixel to total
	total_green = d_pixels[index].green + d_pixels[component2].green + d_pixels[component3].green + d_pixels[component4].green + d_pixels[component5].green; // add the green value at this pixel to total
	total_blue = d_pixels[index].blue + d_pixels[component2].blue + d_pixels[component3].blue + d_pixels[component4].blue + d_pixels[component5].blue; // add the blue value at this pixel to total
	total_red = total_red / 5;
	total_green = total_green / 5;
	total_blue = total_blue / 5;

	d_pixels[index].red = min(total_red, 255);
	d_pixels[index].green = min(total_green, 255);
	d_pixels[index].blue = min(total_blue, 255);
}

/**
*	Sobel kernel (Kaelan and Griffith)
*/
__global__ void GXY(RGB* d_pixels, RGB* d_newimage, int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (y >= height || y >= width)
		return;

	int w = y + 1;							// w, a, s, d are all variables that represent coordinates of pixels offset from the index, they are defined to keep the program from exiting the bounds
	if (w > height || w < 0) {				// of the image. This will affect the average Ex. 
		w = y;								// Normal: 1 + 2 + 3 + 4 + 5 = 15 / 5 = 3	// somewhere in middle of image if 1-5 were assigned from top to bottom, left to right
	}										// New: 3 + 3 + 3 + 4 + 5 = 18 / 5 = 3.6 //Example of top left if 1-5 were assigned from top to bottom, left to right
	int a = y - 1;
	if (a > height || a < 0) {
		a = y;							
	}
	int s = x - 1;
	if (s > width || s < 0) {
		s = x;
	}
	int d = x + 1;
	if (d > width || d < 0) {
		d = x;
	}

	int component1 = w * width + s;			// Set components for the stencil, components are in order reading from top left of stencil to top right and then moving down a row
	int component2 = w * width + x;			// index is the pixel that is being manipulated, and named aptly to put the stencil into perspective
	int component3 = w * width + d;
	int component4 = y * width + s;
	int index = y * width + x;
	int component5 = y * width + d;
	int component6 = a * width + s;
	int component7 = a * width + x;
	int component8 = a * width + d;

	//Gx
	float redx = d_pixels[component1].red + d_pixels[component6].red + (2 * d_pixels[component4].red) + (-1 * (d_pixels[component3].red + d_pixels[component8].red + (2 * d_pixels[component5].red)));
	float greenx = d_pixels[component1].green + d_pixels[component6].green + (2 * d_pixels[component4].green) + (-1 * (d_pixels[component3].green + d_pixels[component8].green + (2 * d_pixels[component5].green)));
	float bluex = d_pixels[component1].blue + d_pixels[component6].blue + (2 * d_pixels[component4].blue) + (-1 * (d_pixels[component3].blue + d_pixels[component8].blue + (2 * d_pixels[component5].blue)));
	//Gy
	float redy = d_pixels[component6].red + d_pixels[component8].red + (2 * d_pixels[component7].red) + (-1 * (d_pixels[component1].red + d_pixels[component3].red + (2 * d_pixels[component2].red)));
	float greeny = d_pixels[component6].green + d_pixels[component8].green + (2 * d_pixels[component7].green) + (-1 * (d_pixels[component1].green + d_pixels[component3].green + (2 * d_pixels[component2].green)));
	float bluey = d_pixels[component6].blue + d_pixels[component8].blue + (2 * d_pixels[component7].blue) + (-1 * (d_pixels[component1].blue + d_pixels[component3].blue + (2 * d_pixels[component2].blue)));
	//Gx && Gy
	float red = sqrt((redx * redx) + (redy * redy));
	float green = sqrt((greenx * greenx) + (greeny * greeny));
	float blue = sqrt((bluex * bluex) + (bluey * bluey));
	//brighter
	red = 4 * red;
	green = 4 * green;
	blue = 4 * blue;
	//set
	d_newimage[index] = { red, green, blue };
}

/**
*	Laplacian kernel (Kaelan)
*/
__global__ void laplacianX(RGB* d_pixels, RGB* d_newimage, int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x; // width
	int y = blockIdx.y * blockDim.y + threadIdx.y; // height

	if (y >= height || y >= width)
		return;

	int w = y + 1;							// w, a, s, d are all variables that represent coordinates of pixels offset from the index, they are defined to keep the program from exiting the bounds
	if (w > height || w < 0) {				// of the image. This will affect the average Ex. 
		w = y;								// Normal: 1 + 2 + 3 + 4 + 5 = 15 / 5 = 3	// somewhere in middle of image if 1-5 were assigned from top to bottom, left to right
	}										// New: 3 + 3 + 3 + 4 + 5 = 18 / 5 = 3.6 //Example of top left if 1-5 were assigned from top to bottom, left to right
	int a = y - 1;
	if (a > height || a < 0) {				// Current situation: Lena works, Sierra and marbles throw an exception. Lena is a pure white image but is a new result file.
		a = y;								// When a .red, .green, or .blue is taken away, the image is visible without those colors
	}
	int s = x - 1;
	if (s > width || s < 0) {
		s = x;
	}
	int d = x + 1;
	if (d > width || d < 0) {
		d = x;
	}

	int component1 = w * width + s;			// Set components for the stencil, components are in order reading from top left of stencil to top right and then moving down a row
	int component2 = w * width + x;			// index is the pixel that is being manipulated, and named aptly to put the stencil into perspective
	int component3 = w * width + d;
	int component4 = y * width + s;
	int index = y * width + x;
	int component5 = y * width + d;
	int component6 = a * width + s;
	int component7 = a * width + x;
	int component8 = a * width + d;

	// 
	float red = (-1 * (d_pixels[component1].red + d_pixels[component2].red + d_pixels[component3].red + d_pixels[component4].red + d_pixels[component5].red + d_pixels[component6].red + d_pixels[component7].red + d_pixels[component8].red)) + (8 * d_pixels[index].red);
	float green = (-1 * (d_pixels[component1].green + d_pixels[component2].green + d_pixels[component3].green + d_pixels[component4].green + d_pixels[component5].green + d_pixels[component6].green + d_pixels[component7].green + d_pixels[component8].green)) + (8 * d_pixels[index].green);
	float blue = (-1 * (d_pixels[component1].blue + d_pixels[component2].blue + d_pixels[component3].blue + d_pixels[component4].blue + d_pixels[component5].blue + d_pixels[component6].blue + d_pixels[component7].blue + d_pixels[component8].blue)) + (8 * d_pixels[index].blue);
	//brighter
	red = 8 * red;
	green = 8 * green;
	blue = 8 * blue;
	//set
	d_newimage[index] = { red, green, blue };
}

/**
*	Helper function to calculate the number of blocks on an axis based on the total grid size and number of threads in that axis
*/
__host__ int calcBlockDim(int total, int num_threads)
{
	int r = total / num_threads;
	if (total % num_threads != 0) // add one to cover all the threads per block
		++r;
	return r;
}

/**
*	Host function for launching greyscale kernel (Dr. Mao)
*/
__host__ void d_convert_greyscale(RGB* pixel, int height, int width)
{
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	greyscaleKernel << <grid, block >> > (d_pixel, height, width);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

/**
*	Host function for launching blurry/averaging kernel (Kaelan)
*/

__host__ void d_compute_component_average(RGB* pixel, int height, int width)
{
	RGB* d_pixel;

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	averageKernel << <grid, block >> > (d_pixel, height, width);

	cudaMemcpy(pixel, d_pixel, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

/**
*	Host function for launching sobel function (Kaelan and Griffith)
*/

__host__ void d_xy(RGB* pixel, RGB* newimage, int height, int width) {
	RGB* d_pixel;
	RGB* d_newimage;

	cudaMalloc(&d_newimage, height * width * sizeof(RGB));
	cudaMemcpy(d_newimage, newimage, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	GXY << <grid, block >> > (d_pixel, d_newimage, height, width);

	cudaMemcpy(pixel, d_newimage, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}

/**
*	Host function for launching laplacian function (Kaelan)
*/

__host__ void d_laplacian(RGB* pixel, RGB* newimage, int height, int width) {
	RGB* d_pixel;
	RGB* d_newimage;

	cudaMalloc(&d_newimage, height * width * sizeof(RGB));
	cudaMemcpy(d_newimage, newimage, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	cudaMalloc(&d_pixel, height * width * sizeof(RGB));
	cudaMemcpy(d_pixel, pixel, height * width * sizeof(RGB), cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = calcBlockDim(width, block.x);
	grid.y = calcBlockDim(height, block.y);

	laplacianX << <grid, block >> > (d_pixel, d_newimage, height, width);

	cudaMemcpy(pixel, d_newimage, height * width * sizeof(RGB), cudaMemcpyDeviceToHost);
}