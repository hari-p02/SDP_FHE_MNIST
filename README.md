# UCONN Senior Design Semester 1

This repository contains the code for FHE and non-FHE implementations for LeNet-5 for the MNIST Dataset.

## Repository Structure

There are 4 main folders in this repository:

### 1. `train_model`

- Contains `train.py` for defining and training a LeNet-5 model.
- Saves weights and biases as flattened vectors in .txt files.
- Includes one test image.

### 2. `weights`

- Stores .txt files with weights trained using `train.py` from the `train_model` folder.

### 3. `mnist`

- Contains code for timing FHE computations on plaintext weights and data.
- Compile with `g++ main.cpp lenet5inference.cpp -o main`.
- Run with `./main`.

### 4. `fhe_mnist`

- A rewritten version of `mnist` code using OpenFHE for encryption.
- Overloads addition and multiplication with `EvalAdd()` and `EvalMult()` from OpenFHE.
- In order to run the executable there is a build folder in `fhe_mnist`, if you make changes to the `main.cpp` or `lenet5inference_FHE.h` make sure to run:
  - `cd ./build/`
  - `rm -R *`
  - `cmake ..`
  - `make`
  - `./main` // if this doesn't work run `LD_LIBRARY_PATH=/usr/local/lib ./main`
  - NOTE: This program take a while
- Note: Execution of the program may take a while.

### 5. `outs`

- Stores outputs of decrypted values.
- Includes several examples for `mulDepth` settings of 10, 5, and 2.

## Additional Information

- Ensure OpenFHE is set up for `fhe_mnist`. [Installation instructions](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html).
- For changes in `fhe_mnist`, follow the build instructions.
