#include <iostream>
#include <fstream>

int main() {
    int width = 28;  
    int height = 28; 
    
    // Allocate a 2D array to store the image pixels
    unsigned char imageArray[height*width];

    // Open the raw image file (binary mode)
    std::ifstream file("../dataset/mnist_images/image_3.jpg", std::ios::binary);

    // Read the image data into the 2D array
    file.read(reinterpret_cast<char*>(imageArray), width * height);


    //Test the image data
    std::cout << "Pixel at (0, 0): " << (int)imageArray[0] << std::endl;
    std::cout << "Pixel at (10, 10): " << (int)imageArray[15] << std::endl;
    std::cout << "Pixel at (27, 27): " << (float)imageArray[27] << std::endl;
    std::cout << "Pixel at (27, 27): " << (float)imageArray[29] << std::endl;

    // Close the file
    file.close();

    return 0;
}