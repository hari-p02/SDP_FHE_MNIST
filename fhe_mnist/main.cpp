#include "lenet5inference_FHE.h"
// #include <cstdlib>

using namespace lbcrypto;

Lenet5Inference :: Lenet5Inference(const std::vector<std::string> input_paths) {
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    keys = cc->KeyGen();

    cc->EvalMultKeyGen(keys.secretKey);

    cc->EvalRotateKeyGen(keys.secretKey, {1, -2});
    
    label = 7;

    for (auto path: input_paths) {
        paths.push_back(path);
    }
}

std::vector<Ciphertext<DCRTPoly>> Lenet5Inference :: readVector(std::string path) {
    std::cout << path << std::endl;
    std::ifstream file(path);
    std::vector<Ciphertext<DCRTPoly>> vec;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double number;
        while (ss >> number) {
            std::vector<double> temp = {number};
            Plaintext ptxt = cc->MakeCKKSPackedPlaintext(temp);
            vec.push_back(cc->Encrypt(keys.publicKey, ptxt));
            if (ss.peek() == ',') ss.ignore();
        }
    }

    return vec;
}

void Lenet5Inference :: Decrypt_and_Save(std::vector<Ciphertext<DCRTPoly>>& input, std::string filename) {
        std::ofstream outFile(filename);

        for (auto& c1 : input) {
            Plaintext result;
            cc->Decrypt(keys.secretKey, c1, &result);
            result->SetLength(batchSize);
            auto decryptedValues = result->GetCKKSPackedValue();
            outFile << decryptedValues[0] << std::endl; 
        }

        outFile.close(); 
}

int Lenet5Inference :: find_flattened_index(int i, int j, int k, int h, std::vector<int> shape, int type) {
    if (type == 4) {
        return i * (shape[1] * shape[2] * shape[3]) + j * (shape[2] * shape[3]) + k * shape[3] + h;
    }
    else if (type == 2) {
        return i * shape[1] + j;
    }
    else if (type == 3) {
        return i * (shape[1] * shape[2]) + j * shape[2] + k;
    }
    else {
        return i;
    }
}

std::vector<Ciphertext<DCRTPoly>> Lenet5Inference :: Convolve(
    const std::vector<Ciphertext<DCRTPoly>> &feature_maps,
    const std::vector<int> feature_maps_SHAPE,
    const std::vector<Ciphertext<DCRTPoly>> filters, 
    const std::vector<int> filters_SHAPE, 
    const std::vector<Ciphertext<DCRTPoly>> biases, 
    int stride, 
    const std::string padding_type) 
{
    int num_filters = filters_SHAPE[3];
    int filter_height = filters_SHAPE[0];
    int filter_width = filters_SHAPE[1];
    int input_depth = feature_maps_SHAPE[2];
    int input_height = feature_maps_SHAPE[0];
    int input_width = feature_maps_SHAPE[1];

    int pad_h, pad_w;
    int output_height, output_width;
    if (padding_type == "same") {
        pad_h = (filter_height - 1) / 2;
        pad_w = (filter_width - 1) / 2;
        output_height = (input_height + pad_h * 2 - filter_height) / stride + 1;
        output_width = (input_width + pad_w * 2 - filter_width) / stride + 1;
    } else { 
        pad_h = pad_w = 0;
        output_height = (input_height - filter_height) / stride + 1;
        output_width = (input_width - filter_width) / stride + 1;
    }

    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>> output(
        output_height,
        std::vector<std::vector<Ciphertext<DCRTPoly>>>(
            output_width,
            std::vector<Ciphertext<DCRTPoly>>(num_filters, 0)
        )
    );

    for (int f = 0; f < num_filters; ++f) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                std::vector<double> temp = {0};
                Plaintext ptxt = cc->MakeCKKSPackedPlaintext(temp); 
                auto total_sum = cc->Encrypt(keys.publicKey, ptxt);
                // Plaintext result0;
                // cc->Decrypt(keys.secretKey, total_sum, &result0);
                // result0->SetLength(batchSize);
                // auto decryptedValues0 = result0->GetCKKSPackedValue();
                // std::cout << decryptedValues0[0] << std::endl;
                for (int dy = 0; dy < filter_height; ++dy) {
                    for (int dx = 0; dx < filter_width; ++dx) {
                        for (int d = 0; d < input_depth; ++d) {
                            int sum_y = y * stride + dy - pad_h;
                            int sum_x = x * stride + dx - pad_w;
                            if (0 <= sum_y && sum_y < input_height && 0 <= sum_x && sum_x < input_width) {
                                // std::cout << "Here in Convolve" << std::endl;
                                auto filter_val = filters[Lenet5Inference::find_flattened_index(dy, dx, d, f, filters_SHAPE, 4)];
                                // std::cout << "Here in Convolve" << std::endl;
                                auto feature_map_val = feature_maps[Lenet5Inference::find_flattened_index(sum_y, sum_x, d, -1, feature_maps_SHAPE, 3)];
                                // std::cout << "Here in Convolve" << std::endl;
                                total_sum = cc->EvalAdd(total_sum, cc->EvalMult(filter_val, feature_map_val));
                                // Plaintext result;
                                // cc->Decrypt(keys.secretKey, total_sum, &result);
                                // result->SetLength(batchSize);
                                // auto decryptedValues = result->GetCKKSPackedValue();
                                // Plaintext result1;
                                // cc->Decrypt(keys.secretKey, filter_val, &result1);
                                // result1->SetLength(batchSize);
                                // auto decryptedValues1 = result1->GetCKKSPackedValue();
                                // Plaintext result2;
                                // cc->Decrypt(keys.secretKey, feature_map_val, &result2);
                                // result2->SetLength(batchSize);
                                // auto decryptedValues2 = result2->GetCKKSPackedValue();
                                // std::cout << decryptedValues[0] << " " << decryptedValues1[0] << " " << decryptedValues2[0] << std::endl;
                                // exit(0);
                                // std::cout << result->GetLogPrecision() << std::endl;
                            }
                        }
                    }
                }
                output[y][x][f] = cc->EvalAdd(total_sum, biases[f]);
		/// size_t e = sizeof(output[y][x][f]);
		// std::cout << "Bytes of One element is: " << e << std::endl;
            }
        }
    }

    return Lenet5Inference::Flatten3DVector(output);
}


std::vector<Ciphertext<DCRTPoly>> Lenet5Inference :: AveragePooling(
    const std::vector<Ciphertext<DCRTPoly>>& input_feature_map,
    int pool_size, 
    int stride,
    int num) 
{
    std::vector<int> cur_SHAPE;
    int input_height = 0;
    int input_width = 0;
    int input_depth = 0;
    if (num == 1) {
        input_height = 32;
        input_width = 32;
        input_depth = 6;
        cur_SHAPE = {32, 32, 6};
    }
    else if (num == 2) {
        input_height = 12;
        input_width = 12;
        input_depth = 16;
        cur_SHAPE = {12, 12, 16};
    }
    

    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>> pooled_feature_map(
        output_height,
        std::vector<std::vector<Ciphertext<DCRTPoly>>>(
            output_width,
            std::vector<Ciphertext<DCRTPoly>>(input_depth, 0)
        )
    );

    for (int d = 0; d < input_depth; ++d) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                std::vector<double> temp = {0};
                Plaintext ptxt = cc->MakeCKKSPackedPlaintext(temp); 
                auto pool_sum = cc->Encrypt(keys.publicKey, ptxt);
                for (int dy = 0; dy < pool_size; ++dy) {
                    for (int dx = 0; dx < pool_size; ++dx) {
                        int idx_y = y * stride + dy;
                        int idx_x = x * stride + dx;
                        if (idx_y < input_height && idx_x < input_width) {
                            pool_sum = cc->EvalAdd(pool_sum, input_feature_map[Lenet5Inference::find_flattened_index(idx_y, idx_x, d, -1, cur_SHAPE, 3)]);
                        }
                    }
                }
                pooled_feature_map[y][x][d] = pool_sum; // / static_cast<float>(pool_size * pool_size);  <-- cant do division
            }
        }
    }

    return Lenet5Inference::Flatten3DVector(pooled_feature_map);
}

// void Lenet5Inference :: ApplyActivationToConvolution(std::vector<float> &conv_output) {  <-- for now cant use tanh (no poly linear approx to it that I can find)
//     for (size_t i = 0; i < conv_output.size(); ++i) {
//         conv_output[i] = Lenet5Inference::Tanh(conv_output[i]);
//     }
// }



// float Lenet5Inference :: Tanh(float x) {
//     return std::tanh(x);
// }

std::vector<Ciphertext<DCRTPoly>> Lenet5Inference::DenseLayer(
    const std::vector<Ciphertext<DCRTPoly>> &input_vector, 
    const std::vector<Ciphertext<DCRTPoly>> weights, 
    const std::vector<Ciphertext<DCRTPoly>> biases, 
    bool activation_function,
    int num) 
{
    int input_size = 0;
    if (num == 1) {
        input_size = 480;
    }
    else if (num == 2) {
        input_size = 84;
    }

    size_t num_neurons = biases.size();
    std::vector<Ciphertext<DCRTPoly>> output_vector;
    output_vector.reserve(num_neurons);

    for (size_t i = 0; i < num_neurons; ++i) {
        std::vector<double> temp = {0.0};
        Plaintext ptxt = cc->MakeCKKSPackedPlaintext(temp); 
        auto weighted_sum = cc->Encrypt(keys.publicKey, ptxt);
        for (int j = 0; j < input_size; ++j) {
            weighted_sum += cc->EvalAdd(weighted_sum, cc->EvalMult(input_vector[j],  weights[Lenet5Inference::find_flattened_index(i, j, -1, -1, {-1, input_size}, 2)]));
        }
        weighted_sum = cc->EvalAdd(weighted_sum, biases[i]);

        if (activation_function) {
            output_vector.push_back(weighted_sum); // <-- getting rid of tanh since it have division
        } else {
            output_vector.push_back(weighted_sum);
        }
    }

    return output_vector;
}


// std::vector<float> Lenet5Inference :: Softmax(const std::vector<float>& logits) { <-- unable to do division due to softmax
//         std::vector<float> softmax_output(logits.size());
//         float max_logit = *std::max_element(logits.begin(), logits.end());
        
//         float sum_of_exps = 0.0;
//         for (size_t i = 0; i < logits.size(); ++i) {
//             softmax_output[i] = std::exp(logits[i] - max_logit);
//             sum_of_exps += softmax_output[i];
//         }

//         for (float& value : softmax_output) {
//             value /= sum_of_exps;
//         }

//         return softmax_output;
// }

std::vector<Ciphertext<DCRTPoly>> Lenet5Inference :: Flatten3DVector(const std::vector<std::vector<std::vector<Ciphertext<DCRTPoly>>>>& vec3D) {
    std::vector<Ciphertext<DCRTPoly>> flattened;

    for (const auto& matrix : vec3D) {        
        for (const auto& row : matrix) {      
            for (auto val : row) {          
                flattened.push_back(val);
            }
        }
    }

    return flattened;
}

int Lenet5Inference :: Forward() { 
    /* ----------------- Layer 1 ----------------- */
    auto layer1_data = std::chrono::high_resolution_clock::now();
    test = Lenet5Inference :: readVector(paths[10]);
    filters_layer_0 = Lenet5Inference :: readVector(paths[0]);
    biases_layer_0 = Lenet5Inference :: readVector(paths[1]);
    auto layer1_data_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dataprocessingtime_layer1 =  layer1_data_end - layer1_data;

    std::cout << "Starting output1" << std::endl;
    auto layer1_start = std::chrono::high_resolution_clock::now();
    auto output1 = Lenet5Inference::Convolve(test, test_SHAPE, filters_layer_0, filters_layer_0_SHAPE, biases_layer_0, 1, "same");
    // Lenet5Inference::ApplyActivationToConvolution(output1);
    auto layer1_end = std::chrono::high_resolution_clock::now();

    try {
        Decrypt_and_Save(output1, std::string("output1.txt"));
    } catch (const std::exception& e) {
        std::cout << "Error (D & S): " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Could not decrypt and save output1" << std::endl;
    }
    

    filters_layer_0.clear();
    filters_layer_0.shrink_to_fit();
    biases_layer_0.clear();
    biases_layer_0.shrink_to_fit();
    test.clear();
    test.shrink_to_fit();

    std::cout << "Layer 1 (Data Processing): " << dataprocessingtime_layer1.count() << " seconds" << std::endl;
    std::cout << "Layer 1 (Conv) Time Taken: " << (std::chrono::duration<double>(layer1_end - layer1_start)).count() << " seconds" << std::endl;
    
    
    /* ----------------- Layer 2 ----------------- */
    std::cout << "Starting poo1" << std::endl;
    auto layer2_start = std::chrono::high_resolution_clock::now();
    auto pool1 = Lenet5Inference::AveragePooling(output1, 2, 2, 1);
    auto layer2_end = std::chrono::high_resolution_clock::now();

    try {
        Decrypt_and_Save(pool1, std::string("pool1.txt"));
    } catch (const std::exception& e) {
        std::cout << "Error (D & S): " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Could not decrypt and save pool1" << std::endl;
    }

    output1.clear();
    output1.shrink_to_fit();

    std::cout << "Layer 2 (Avg Pool) Time Taken: " << (std::chrono::duration<double>(layer2_end - layer2_start)).count() << " seconds" << std::endl;


    /* ----------------- Layer 3 ----------------- */
    auto layer3_data = std::chrono::high_resolution_clock::now();
    filters_layer_2 = Lenet5Inference :: readVector(paths[2]);
    biases_layer_2 = Lenet5Inference :: readVector(paths[3]);
    auto layer3_data_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dataprocessingtime_layer3 = layer3_data_end - layer3_data;

    std::cout << "Starting output2" << std::endl;
    auto layer3_start = std::chrono::high_resolution_clock::now();
    auto output2 = Lenet5Inference::Convolve(pool1, {16, 16, 6} ,filters_layer_2, filters_layer_2_SHAPE, biases_layer_2, 1, "valid");
    // Lenet5Inference::ApplyActivationToConvolution(output2);
    auto layer3_end = std::chrono::high_resolution_clock::now();

    try {
        Decrypt_and_Save(output2, std::string("output2.txt"));
    } catch (...) {
        std::cout << "Could not decrypt and save output2" << std::endl;
    }

    filters_layer_2.clear();
    filters_layer_2.shrink_to_fit();
    biases_layer_2.clear();
    biases_layer_2.shrink_to_fit();
    pool1.clear();
    pool1.shrink_to_fit();

    std::cout << "Layer 3 (Data Processing): " << dataprocessingtime_layer3.count() << " seconds" << std::endl;
    std::cout << "Layer 3 (Conv) Time Taken: " << (std::chrono::duration<double>(layer3_end - layer3_start)).count() << " seconds" << std::endl;
    

    /* ----------------- Layer 4 ----------------- */
    std::cout << "Starting pool2" << std::endl;
    auto layer4_start = std::chrono::high_resolution_clock::now();
    auto pool2 = Lenet5Inference::AveragePooling(output2, 2, 2, 2);
    auto layer4_end = std::chrono::high_resolution_clock::now();

    try {
        Decrypt_and_Save(pool2, std::string("pool2.txt"));
    } catch (const std::exception& e) {
        std::cout << "Error (D & S): " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Could not decrypt and save pool2" << std::endl;
    }

    output2.clear();
    output2.shrink_to_fit();

    std::cout << "Layer 4 (Avg Pool) Time Taken: " << (std::chrono::duration<double>(layer4_end - layer4_start)).count() << " seconds" << std::endl;


    /* ----------------- Layer 5 ----------------- */
    auto layer5_data = std::chrono::high_resolution_clock::now();
    filters_layer_4 = Lenet5Inference :: readVector(paths[4]);
    biases_layer_4 = Lenet5Inference :: readVector(paths[5]);
    auto layer5_data_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dataprocessingtime_layer5 = layer5_data_end - layer5_data;

    std::cout << "Starting output3" << std::endl;
    auto layer5_start = std::chrono::high_resolution_clock::now();
    auto output3 = Lenet5Inference::Convolve(pool2, {6, 6, 16}, filters_layer_4, filters_layer_4_SHAPE, biases_layer_4, 1, "valid");
    // Lenet5Inference::ApplyActivationToConvolution(output3); 
    auto layer5_end = std::chrono::high_resolution_clock::now();

    try {
        Decrypt_and_Save(output3, std::string("output3.txt"));
    } catch (const std::exception& e) {
        std::cout << "Error (D & S): " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Could not decrypt and save output3" << std::endl;
    }
    

    filters_layer_4.clear();
    filters_layer_4.shrink_to_fit();
    biases_layer_4.clear();
    biases_layer_4.shrink_to_fit();
    pool2.clear();
    pool2.shrink_to_fit();

    std::cout << "Layer 5 (Data Processing): " << dataprocessingtime_layer5.count() << " seconds" << std::endl;
    std::cout << "Layer 5 (Conv) Time Taken: " << (std::chrono::duration<double>(layer5_end - layer5_start)).count() << " seconds" << std::endl;
    

    /* ----------------- Layer 6 ----------------- */
    auto layer6_data = std::chrono::high_resolution_clock::now();
    dense_layer_6 = Lenet5Inference :: readVector(paths[6]);
    dense_biases_layer_6 = Lenet5Inference :: readVector(paths[7]);
    auto layer6_data_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dataprocessingtime_layer6 = layer6_data_end - layer6_data;

    
    std::cout << "Starting dense1" << std::endl;
    auto layer6_start = std::chrono::high_resolution_clock::now();
    auto dense1 = Lenet5Inference::DenseLayer(output3, dense_layer_6, dense_biases_layer_6, true, 1);
    auto layer6_end = std::chrono::high_resolution_clock::now();

    try {
        Decrypt_and_Save(dense1, std::string("dense1.txt"));
    } catch (const std::exception& e) {
        std::cout << "Error (D & S): " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Could not decrypt and save dense1" << std::endl;
    }

    dense_layer_6.clear();
    dense_layer_6.shrink_to_fit();
    dense_biases_layer_6.clear();
    dense_biases_layer_6.shrink_to_fit();
    output3.clear();
    output3.shrink_to_fit();

    std::cout << "Layer 6 (Data Processing): " << dataprocessingtime_layer6.count() << " seconds" << std::endl;
    std::cout << "Layer 6 (Dense) Time Taken: " << (std::chrono::duration<double>(layer6_end - layer6_start)).count() << " seconds" << std::endl;
    
    

    /* ----------------- Layer 7 ----------------- */
    auto layer7_data = std::chrono::high_resolution_clock::now();
    dense_layer_7 = Lenet5Inference :: readVector(paths[8]);
    dense_biases_layer_7 = Lenet5Inference :: readVector(paths[9]);
    auto layer7_data_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dataprocessingtime_layer7 = layer7_data_end - layer7_data;


    std::cout << "Starting output_layer_output" << std::endl;
    auto layer7_start = std::chrono::high_resolution_clock::now();
    auto output_layer_output = Lenet5Inference::DenseLayer(dense1, dense_layer_7, dense_biases_layer_7, false, 2);
    // auto output_probabilities = Lenet5Inference::Softmax(output_layer_output);
    auto layer7_end = std::chrono::high_resolution_clock::now();

    try {
        Decrypt_and_Save(output_layer_output, std::string("output_layer_output.txt"));
    } catch (const std::exception& e) {
        std::cout << "Error (D & S): " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Could not decrypt and save output_layer_output" << std::endl;
    }

    dense_layer_7.clear();
    dense_layer_7.shrink_to_fit();
    dense_biases_layer_7.clear();
    dense_biases_layer_7.shrink_to_fit();
    dense1.clear();
    dense1.shrink_to_fit();
    output_layer_output.clear();
    output_layer_output.shrink_to_fit();

    std::cout << "Layer 7 (Data Processing): " << dataprocessingtime_layer7.count() << " seconds" << std::endl;
    std::cout << "Layer 7 (Dense) Time Taken: " << (std::chrono::duration<double>(layer7_end - layer7_start)).count() << " seconds" << std::endl;
    

    auto totaldataprocesstime = dataprocessingtime_layer1.count() + dataprocessingtime_layer3.count() + dataprocessingtime_layer5.count() + dataprocessingtime_layer6.count() + dataprocessingtime_layer7.count();

    std::cout << "Total time taken to Read Data and Encrypt it: " << totaldataprocesstime << " seconds" << std::endl;

    std::cout << "*************** Data Encryptiong Time Per Layer: " << std::endl;
    std::cout << "Layer 1: " << dataprocessingtime_layer1.count() << " seconds" << std::endl;
    std::cout << "Layer 3: " << dataprocessingtime_layer3.count() << " seconds" << std::endl;
    std::cout << "Layer 5: " << dataprocessingtime_layer5.count() << " seconds" << std::endl;
    std::cout << "Layer 6: " << dataprocessingtime_layer6.count() << " seconds" << std::endl;
    std::cout << "Layer 7: " << dataprocessingtime_layer7.count() << " seconds" << std::endl;
    std::cout << "******************************" << std::endl;


    std::cout << "Layer 1 (Conv) Time Taken: " << (std::chrono::duration<double>(layer1_end - layer1_start)).count() << " seconds" << std::endl;
    std::cout << "Layer 2 (Ang Pool) Time Taken: " << (std::chrono::duration<double>(layer2_end - layer2_start)).count() << " seconds" << std::endl;
    std::cout << "Layer 3 (Conv) Time Taken: " << (std::chrono::duration<double>(layer3_end - layer3_start)).count() << " seconds" << std::endl;
    std::cout << "Layer 4 (Avg Pool) Time Taken: " << (std::chrono::duration<double>(layer4_end - layer4_start)).count() << " seconds" << std::endl;
    std::cout << "Layer 5 (Conv) Time Taken: " << (std::chrono::duration<double>(layer5_end - layer5_start)).count() << " seconds" << std::endl;
    std::cout << "Layer 6 (Dense) Time Taken: " << (std::chrono::duration<double>(layer6_end - layer6_start)).count() << " seconds" << std::endl;
    std::cout << "Layer 7 (Dense) Time Taken: " << (std::chrono::duration<double>(layer7_end - layer7_start)).count() << " seconds" << std::endl;
    // auto max_it = std::max_element(output_layer_output.begin(), output_layer_output.end());
    // int max_index = std::distance(output_layer_output.begin(), max_it);
    
    return 0;
}

int main() {
    // std::vector<std::string> paths = {"/home/harip/sdp/mnist/weights/filters_layer_0.txt",
    //  "/home/harip/sdp/mnist/weights/biases_layer_0.txt",
    //  "/home/harip/sdp/mnist/weights/filters_layer_2.txt",
    //  "/home/harip/sdp/mnist/weights/biases_layer_2.txt",
    //  "/home/harip/sdp/mnist/weights/filters_layer_4.txt",
    //  "/home/harip/sdp/mnist/weights/biases_layer_4.txt",
    //  "/home/harip/sdp/mnist/weights/dense_layer_6.txt",
    //  "/home/harip/sdp/mnist/weights/dense_biases_layer_6.txt",
    //  "/home/harip/sdp/mnist/weights/dense_layer_7.txt",
    //  "/home/harip/sdp/mnist/weights/dense_biases_layer_7.txt",
    //  "/home/harip/sdp/mnist/weights/test.txt"
    //  };
    std::vector<std::string> paths = {"/data/hkp18001/sdp27/weights/filters_layer_0.txt",
     "/data/hkp18001/sdp27/weights/biases_layer_0.txt",
     "/data/hkp18001/sdp27/weights/filters_layer_2.txt",
     "/data/hkp18001/sdp27/weights/biases_layer_2.txt",
     "/data/hkp18001/sdp27/weights/filters_layer_4.txt",
     "/data/hkp18001/sdp27/weights/biases_layer_4.txt",
     "/data/hkp18001/sdp27/weights/dense_layer_6.txt",
     "/data/hkp18001/sdp27/weights/dense_biases_layer_6.txt",
     "/data/hkp18001/sdp27/weights/dense_layer_7.txt",
     "/data/hkp18001/sdp27/weights/dense_biases_layer_7.txt",
     "/data/hkp18001/sdp27/weights/test.txt"
     };

    Lenet5Inference lenet(paths);

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << lenet.Forward() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Entire Inference Time Taken: " << (std::chrono::duration<double>(end - start)).count() << " seconds" << std::endl;
    
    return 0;
}
