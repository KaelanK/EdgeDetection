/**
* CS-315 (Distributed Scalable Computing) Converting to greyscale
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <chrono>
#include <vector>
using namespace std;

#include "bmp.h"

#define NUM_IMAGES	3

// Resource: https://stackoverflow.com/questions/36042637/how-to-calculate-execution-time-in-millisecondsl // calculate functions time to complete
// Resource: https://www.cs.auckland.ac.nz/compsci373s1c/PatricesLectures/Edge%20detection-Sobel_2up.pdf //Major source of logic for sobel method
// Resource: https://aishack.in/tutorials/sobel-laplacian-edge-detectors/ // laplacian reference, really only needed grid formula
void d_convert_greyscale(RGB* pixels, int height, int width); // Function using device to convert to greyscale
void d_compute_component_average(RGB* pixels, int height, int width); // Function using device to blur the image
void d_laplacian(RGB* pixels, RGB* newimage, int height, int width);   // Function using device to edge detect using laplacian method
void d_xy(RGB* pixels, RGB* newimage, int height, int width);	//Function using device to edge detect using sobel method
void d_ConvertToBitmap(string filename); //Converts jpg to a bitmap file

void d_ConvertToBitmap(string filename) {}


/**
*  Computes the average of the red, green, and blue components of an image
*
* @param pixels  The array of RGB (Red, Green, Blue) components of each pixel in the image
* @param height  The height of the image
* @param width   The width of the image
*/

int main()
{
	do {
		string image_archive[NUM_IMAGES] = { "lena.bmp", "marbles.bmp", "sierra_02.bmp" };
		cout << "Select an image: \n";
		for (int i = 0; i < NUM_IMAGES; ++i)
			cout << i << ": " << image_archive[i] << endl;
		cout << NUM_IMAGES << ": exit\n";

		int choice;
		do {
			cout << "Please choice: ";
			cin >> choice;
			if (choice == NUM_IMAGES) {
				cout << "Goodbye!\n";
				exit(0);
			}
		} while (choice < 0 || choice > NUM_IMAGES);

		BitMap d_image(image_archive[choice]); // Need to reset the "image" variable to be the original

		// Display some of the image's properties
		cout << "Image properties\n";
		cout << setw(15) << left << "Dimensions: " << d_image.getHeight() << " by " << d_image.getWidth() << endl;
		cout << setw(15) << left << "Size: " << d_image.getImageSize() << " bytes\n";
		cout << setw(15) << left << "Bit encoding: " << d_image.getBitCount() << " bits\n\n";

		// Get the image array of file, two copies one to evaluate, one to manipulate
		RGB* pixels = d_image.getRGBImageArray();
		RGB* newimage = d_image.getRGBImageArray();

		//Convert the image to black and white using greyscale function (Dr. Mao's code)
		d_convert_greyscale(pixels, d_image.getHeight(), d_image.getWidth());

		//Blur the image 20 times (blur using average technique from exam 2)
		for (int i = 0; i < 20; i++) {
			d_compute_component_average(pixels, d_image.getHeight(), d_image.getWidth()); //Device runs average, works with all images
		}

		//Choose which edge detection you would like to do
		cout << "Please choose laplacian or sobel: [1/2] " << endl;
		string method;
		cin >> method;

		//Timer for functions, comment out the one not testing
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);


		cudaEventRecord(start);
		// Sobel function uses two graphs of x and y and compares them
		//d_xy(pixels, newimage, d_image.getHeight(), d_image.getWidth());

		//Laplacian uses a singular graph with arithmetic more obliged to running through a single graph

		if (method == "1")
			d_laplacian(pixels, newimage, d_image.getHeight(), d_image.getWidth());
		else if (method == "2")
			d_xy(pixels, newimage, d_image.getHeight(), d_image.getWidth());
		else
			cout << "Task failed, no method ran.";

		cudaEventRecord(stop);

		// Output collected data
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "The function took " << milliseconds << " milliseconds\n\n";

		// Assign the modified pixels back to the image
		d_image.setImageFromRGB(pixels);

		// Save this image in "result.bmp"
		d_image.saveBMP("result.bmp");


		//reset message
		cout << "Check out result.bmp (click on it) to see image processing result\n\n";
		char response = 'y';
		cout << "Do you wish to repeat? [y/n] ";
		cin >> response;
		if (response != 'y') {
			cout << "Sorry to see you go ...\n";
			exit(0);
		}
	} while (true);

}