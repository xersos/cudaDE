/* Copyright 2017 Ian Rankin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 * to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//
//  testMain.cpp
//
// This is a test code to show an example usage of Differential Evolution

#include <stdio.h>

#include "DifferentialEvolution.hpp"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <ctime>

using namespace std;
int main(int argc, char **argv)
{
    int popsize = stoi(argv[1]);
    int dimensions = stoi(argv[2]);

    float F1 = strtof(argv[3], NULL);
    float F2 = strtof(argv[4], NULL);
    float F3 = strtof(argv[5], NULL);
    float F4 = strtof(argv[6], NULL);

    float maxB = strtof(argv[7], NULL);
    float minB = strtof(argv[8], NULL);
    
    bool isCrash = strtof(argv[9], NULL);
    
    int generations = 0;
    
    generations = (dimensions * 10000) / popsize;
     std::cout << "generations:  " << generations << std::endl;
    // create the min and max bounds for the search space.
    float minBounds[2] = {minB, minB};
    float maxBounds[2] = {maxB, maxB};
    
    // 2D Array that contains results
    float res[25][2] = {0};
    std::time_t resultSS = std::time(nullptr);
    for(int i = 0; i < 25; i++){
        // a random array or data that gets passed to the cost function.
        float arr[3] = {2.5, 2.6, 2.7};
        if(isCrash){
            std::cout << "iteration " << i + 1 << std::endl;
            
        }
        
        // data that is created in host, then copied to a device version for use with the cost function.
        struct data x;
        struct data *d_x;
        gpuErrorCheck(cudaMalloc(&x.bias, sizeof(float) * 3));
        unsigned long size = sizeof(struct data);
        gpuErrorCheck(cudaMalloc((void **)&d_x, size));
        x.v = 3;
        x.dim = 2;
        gpuErrorCheck(cudaMemcpy(x.bias, (void *)&arr, sizeof(float) * 3, cudaMemcpyHostToDevice));
        std::time_t resultS = std::time(nullptr);
        // Create the minimizer with a popsize of 192, 50 generations, Dimensions = 2, CR = 0.9, F = 2
        DifferentialEvolution minimizer(popsize,generations,dimensions, 0.9, 0.5, minBounds, maxBounds, F1, F2, F3, F4);
        
        gpuErrorCheck(cudaMemcpy(d_x, (void *)&x, sizeof(struct data), cudaMemcpyHostToDevice));
        
        // get the result from the minimizer
        std::vector<float> result = minimizer.fmin(d_x);
        //std::cout << "Result = " << result[0] << ", " << result[1] << std::endl;
        std::time_t resultF = std::time(nullptr);
        std::time_t resultFF = std::time(nullptr);
        
        if(isCrash){
            std::cout << resultF - resultS << " milliseconds for iter\n" << std::endl;
        }
        
        cudaFree(x.bias);
        cudaFree(d_x);
    }
    std::time_t resultFF = std::time(nullptr);
    std::cout << std::endl;
    std::cout << resultFF - resultSS << " milliseconds since start\n" << std::endl;
    std::cout << res << std::endl;
    std::cout << "Finished main function." << std::endl;
    return 1;
}