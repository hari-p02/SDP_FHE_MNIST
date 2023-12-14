#ifndef lenet5inference_h
#define lenet5inference_h

#include <vector>
#include <cmath>
#include <array>
#include <string>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <functional>
#include <iostream>
#include <sstream>
#include <chrono>

class Lenet5Inference {
    public:
        Lenet5Inference(const std::vector<std::string> paths);

        std::vector<float> readVector(std::string path);

        int find_flattened_index(int i, int j, int k, int h, std::vector<int> shape, int type);

        void Save(std::vector<float>& input, std::string filename);

        std::vector<float> Convolve(const std::vector<float> &feature_maps, const std::vector<int> feature_maps_SHAPE, const std::vector<float> filters, const std::vector<int> filters_SHAPE, const std::vector<float> biases, int stride, const std::string padding_type); 

        std::vector<float> AveragePooling(const std::vector<float>& input_feature_map, int pool_size, int stride, int num); 

        float Tanh(float x);

        void  ApplyActivationToConvolution(std::vector<float> &conv_output);

        std::vector<float> DenseLayer(const std::vector<float> &input_vector, const std::vector<float> weights, const std::vector<float> biases, bool activation_function, int num); 
        
        std::vector<float> Softmax(const std::vector<float>& logits);
        
        std::vector<float> Flatten3DVector(const std::vector<std::vector<std::vector<float>>>& vec3D);

        int Forward();

    private:
        std::vector<std::string> paths;
        
        std::vector<float> filters_layer_0;
        std::vector<int> filters_layer_0_SHAPE = {5, 5, 1, 6};

        std::vector<float> biases_layer_0;

        std::vector<float> filters_layer_2;
        std::vector<int> filters_layer_2_SHAPE = {5, 5, 6, 16};

        std::vector<float> biases_layer_2;

        std::vector<float> filters_layer_4;
        std::vector<int> filters_layer_4_SHAPE = {5, 5, 16, 120};

        std::vector<float> biases_layer_4;

        std::vector<float> dense_layer_6;
        std::vector<int> dense_layer_6_SHAPE = {84, 480};

        std::vector<float> dense_biases_layer_6;

        std::vector<float> dense_layer_7;
        std::vector<int> dense_layer_7_SHAPE = {10, 84};

        std::vector<float> dense_biases_layer_7;

        std::vector<float> test;
        std::vector<int> test_SHAPE = {32, 32, 1};

        int label;
};


#endif 