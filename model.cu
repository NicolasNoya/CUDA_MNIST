#include <iostream>
#include "cuda.h"
#include <fstream>
#include <string>
#include "cuda_utils.h"
#include "weights.h"

const int width = 28;  
const int height = 28; 


class Model {
private:
    const float *Dense0;
    const float *Bias1;
    const float *Dense2;
    const float *Bias3;
    const float *Dense4;
    const float *Bias5;

    void load_image(std::string path, float *image) {
        
        // Allocate a 2D array to store the image pixels
        unsigned char imageArray[height*width];

        // Open the raw image file (binary mode)
        std::ifstream file("../dataset/mnist_images/image_3.jpg", std::ios::binary);

        // Read the image data into the 2D array
        file.read(reinterpret_cast<char*>(imageArray), width * height);

        //Make the array a float array
        float imageArrayFloat[height*width];
        for (int i = 0; i < height*width; i++) {
            imageArrayFloat[i] = (float)imageArray[i];
        }

        // Close the file
        file.close();

        image = imageArrayFloat;

    }


public:
    // Constructor
    Model() {
        // Dense0 = layer_dense_0;
        // Bias1 = layer_bias_1;
        // Dense2 = layer_dense_2;
        // Bias3 = layer_bias_3;
        // Dense4 = layer_dense_4;
        // Bias5 = layer_bias_5;

        cudaMalloc((void **)&this->Dense0, 32*784*sizeof(float));
        cudaMalloc((void **)&this->Bias1, 32*sizeof(float));
        cudaMalloc((void **)&this->Dense2, 32*32*sizeof(float));
        cudaMalloc((void **)&this->Bias3, 32*sizeof(float));
        cudaMalloc((void **)&this->Dense4, 10*32*sizeof(float));
        cudaMalloc((void **)&this->Bias5, 10*sizeof(float));

        cudaMalloc((void **)&Dense0T, 32*784*sizeof(float));
        cudaMalloc((void **)&Dense2T, 32*32*sizeof(float));
        cudaMalloc((void **)&Dense4T, 10*32*sizeof(float));

        cudaMemcpy(Dense0T, layer_dense_0, 32*784*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Bias1, layer_bias_1, 32*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Dense2T, layer_dense_2, 32*32*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Bias3, layer_bias_3, 32*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Dense4T, layer_dense_4, 10*32*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Bias5, layer_bias_5, 10*sizeof(float), cudaMemcpyHostToDevice);

        dim3 gridSize(2,25,1);
        dim3 blockSize(32,32,1);

        transposeMatrix<<<gridSize, blockSize>>>(this->Dense0T, Dense0, 32, 784);
        transposeMatrix<<<gridSize, blockSize>>>(this->Dense2T, Dense2, 32, 32);
        transposeMatrix<<<gridSize, blockSize>>>(this->Dense4T, Dense4, 10, 32);
        cudaFree(Dense0T);
        cudaFree(Dense2T);
        cudaFree(Dense4T);
        cudaDeviceSynchronize();
    }

    //
    int predict(std::string path){
        float *image;
        image = (float *)malloc(height*width*sizeof(float));
        this->load_image(path, image);
        cudaMalloc((void **)&d_image, height*width*sizeof(float));
        cudaMemcpy(d_image, image, height*width*sizeof(float), cudaMemcpyHostToDevice);

        // First Layer
        dim3 blockSizeLayer1(784, 1, 1);
        dim3 gridSizeLayer1(1, 32, 1);
        cudaMalloc((void **)&layer1, 32*sizeof(float));
        easyvectmult<<<blockSizeLayer1, gridSizeLayer1>>>(d_image, this->Dense0, layer_1, 32, 784);
        cudaDeviceSynchronize();

        dim3 blockSizeBias1(32, 1, 1);
        dim3 gridSizeBias1(1, 1, 1);
        easyvectsum<<<blockSizeBias1, gridSizeBias1>>>(layer_1, this->Bias1, 32, 1);
        cudaDeviceSynchronize();

        RELU<<<blockSizeBias1, gridSizeBias1>>>(layer_1, 32);
        cudaDeviceSynchronize();

        // Second Layer
        dim3 blockSizeLayer2(32, 1, 1);
        dim3 gridSizeLayer2(1, 32, 1);
        cudaMalloc((void **)&layer2, 32*sizeof(float));
        easyvectmult<<<blockSizeLayer2, gridSizeLayer2>>>(layer_1, this->Dense2, layer_2, 32, 32);
        cudaDeviceSynchronize();

        dim3 blockSizeBias2(32, 1, 1);
        dim3 gridSizeBias2(1, 1, 1);
        easyvectsum<<<blockSizeBias2, gridSizeBias2>>>(layer_2, this->Bias3, 32, 1);
        cudaDeviceSynchronize();

        RELU<<<blockSizeBias2, gridSizeBias2>>>(layer_2, 32);
        cudaDeviceSynchronize();

        // Third Layer
        dim3 blockSizeLayer3(32, 1, 1);
        dim3 gridSizeLayer3(1, 10, 1);
        cudaMalloc((void **)&layer3, 10*sizeof(float));
        easyvectmult<<<blockSizeLayer3, gridSizeLayer3>>>(layer_2, this->Dense4, layer_3, 10, 32);
        cudaDeviceSynchronize();

        dim3 blockSizeBias3(10, 1, 1);
        dim3 gridSizeBias3(1, 1, 1);
        easyvectsum<<<blockSizeBias3, gridSizeBias3>>>(layer_3, this->Bias5, 10, 1);
        cudaDeviceSynchronize();

        int max_index = 0;
        float max_value = 0;
        gpuoutput = (float *)malloc(10*sizeof(float));
        cudaMemcpy(gpuoutput, layer_3, 10*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++) {
            if (gpuoutput[i] > max_value) {
                max_value = gpuoutput[i];
                max_index = i;
            }
        }
        return max_index;
    }
};