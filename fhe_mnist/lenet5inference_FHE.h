#ifndef lenet5inferenceFHE_h
#define lenet5inferenceFHE_h

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
#include "openfhe.h"
#include <chrono>

using namespace lbcrypto;

class Lenet5Inference {
    public:
        Lenet5Inference(const std::vector<std::string> paths);

        std::vector<Ciphertext<DCRTPoly>> readVector(std::string path);

        int find_flattened_index(int i, int j, int k, int h, std::vector<int> shape, int type);

        std::vector<Ciphertext<DCRTPoly>> Convolve(
    const std::vector<Ciphertext<DCRTPoly>> &feature_maps,
    const std::vector<int> feature_maps_SHAPE,
    const std::vector<Ciphertext<DCRTPoly>> filters, 
    const std::vector<int> filters_SHAPE, 
    const std::vector<Ciphertext<DCRTPoly>> biases, 
    int stride, 
    const std::string padding_type);

    void Decrypt_and_Save(std::vector<Ciphertext<DCRTPoly>>& input, std::string filename);

        
std::vector<Ciphertext<DCRTPoly>> AveragePooling(
    const std::vector<Ciphertext<DCRTPoly>>& input_feature_map,
    int pool_size, 
    int stride,
    int num);
        // float Tanh(float x);

        // void  ApplyActivationToConvolution(std::vector<float> &conv_output);

        std::vector<Ciphertext<DCRTPoly>> DenseLayer(
    const std::vector<Ciphertext<DCRTPoly>> &input_vector, 
    const std::vector<Ciphertext<DCRTPoly>> weights, 
    const std::vector<Ciphertext<DCRTPoly>> biases, 
    bool activation_function,
    int num); 
        // std::vector<float> Softmax(const std::vector<float>& logits);
        
        std::vector<Ciphertext<DCRTPoly>> Flatten3DVector(const std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& vec3D);

        int Forward();

    private:
        std::vector<std::string> paths;

        std::vector<Ciphertext<DCRTPoly>> filters_layer_0;
        std::vector<int> filters_layer_0_SHAPE = {5, 5, 1, 6};

        std::vector<Ciphertext<DCRTPoly>> biases_layer_0;

        std::vector<Ciphertext<DCRTPoly>> filters_layer_2;
        std::vector<int> filters_layer_2_SHAPE = {5, 5, 6, 16};

        std::vector<Ciphertext<DCRTPoly>> biases_layer_2;

        std::vector<Ciphertext<DCRTPoly>> filters_layer_4;
        std::vector<int> filters_layer_4_SHAPE = {5, 5, 16, 120};

        std::vector<Ciphertext<DCRTPoly>> biases_layer_4;

        std::vector<Ciphertext<DCRTPoly>> dense_layer_6;
        std::vector<int> dense_layer_6_SHAPE = {84, 480};

        std::vector<Ciphertext<DCRTPoly>> dense_biases_layer_6;

        std::vector<Ciphertext<DCRTPoly>> dense_layer_7;
        std::vector<int> dense_layer_7_SHAPE = {10, 84};

        std::vector<Ciphertext<DCRTPoly>> dense_biases_layer_7;

        std::vector<Ciphertext<DCRTPoly>> test;
        std::vector<int> test_SHAPE = {32, 32, 1};

        int label;

        uint32_t multDepth = 1;
        uint32_t scaleModSize = 50;
        uint32_t batchSize = 1;

        CCParams<CryptoContextCKKSRNS> parameters;

        CryptoContext<DCRTPoly> cc;

        lbcrypto::KeyPair<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<long unsigned int> > > > keys; 
};


#endif 
