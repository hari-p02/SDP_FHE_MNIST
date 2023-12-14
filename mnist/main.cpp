#include "lenet5inference.h"


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

    std::cout << lenet.Forward() << std::endl;

    return 0;
}