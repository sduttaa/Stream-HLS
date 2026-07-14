# Stream-HLS

![Stream-HLS](framework.png)

Stream-HLS is a novel framework designed to simplify the development of customized hardware circuits for software applications using high-level synthesis (HLS). Traditional HLS approaches often require significant expertise in hardware design and frameworks, making the process non-trivial. Stream-HLS takes PyTorch or C/C++ code and automatically generates a streaming dataflow design in HLS optimized for a target FPGA.

## Key Features

Stream-HLS offers the following features:

- **Automatic Hardware Generation**: Converts C/C++ or PyTorch software code into optimized custom hardware designs and host code for FPGAs.
- **Streaming Dataflow Architectures**: Generates hardware designs consisting of modules that communicate through pure streaming channels when possible or through a mixture of shared buffers and streaming channels.
- **Global Scheduling and Performance Model**: Provides an accurate analytical performance model for global scheduling of streaming dataflow architectures.
- **High Performance**: Evaluated using various HLS benchmarks and real-world benchmarks from convolutional neural networks and transformer models. Stream-HLS designs outperform the designs of prior state-of-the-art automation frameworks and manually-optimized designs of abstraction frameworks by up to 79.43× and 10.62× geometric means respectively.
- **Built on MLIR**: Utilizes the multi-level intermediate representation (MLIR) framework, allowing for future reuse and extensions.

## Installation

### External Tools (Front-end)
#### 1- [Torch-MLIR](https://github.com/llvm/torch-mlir)
Create a Python virtual environment using `conda` or `pyenv` using `Python 3.11`. Now run:
```sh
pip install -r requirements.txt
```
Alternatively, you can install Torch and Torch-MLIR packages from this [link](https://github.com/llvm/torch-mlir/releases/tag/snapshot-20240127.1096), and install them using pip.

#### 2- [Polygeist](https://github.com/llvm/Polygeist) (Optional)
Can be installed if the C/C++ front-end is needed.

### Stream-HLS Prerequisites
#### 1- LLVM/MLIR
Clone and build [LLVM/MLIR](https://github.com/llvm/llvm-project/tree/llvmorg-18.1.2) version llvmorg-18.1.2 in extern using the following commands.

```sh
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON

cmake --build . --target check-mlir
```


#### 2- Ampl and Gurobi Solvers
Install the free version of [Ampl in academia](https://ampl.com/ampl-in-academia/) which is shipped with Gurobi.

> **Open-source DSE migration:** the repository now contains an experimental,
> license-free OR-Tools CP-SAT boundary for composing schedule candidates and
> communication modes. The existing end-to-end pipeline continues to use
> AMPL/Gurobi until compiler integration and benchmark parity are complete. See
> [`docs/open-source-dse.md`](./docs/open-source-dse.md) for the schema, status
> contract, and migration plan.

### Stream-HLS Installation

After building the MLIR/LLVM project, please run:
```sh
./build-streamhls.sh /path/to/llvm-project
``` 
Or open the build bash script [`build-streamhls.sh`](./build-streamhls.sh) and set path variable manually to point to the llvm-project installed in the previous step.


## How to Use

For simple usage, refer to the examples provided in the [`examples`](./examples/) directory. This directory includes sample scripts and configurations to help you get started with Stream-HLS.

If you encounter any problems, please feel free to open an [`issue`](https://github.com/UCLA-VAST/Stream-HLS/issues).

## Publications
Please refer to our FPGA'25 paper for more details. If you use Stream-HLS in your research, please use the following bibtex entry to cite us:
```bibtex
@inproceedings{streamhlsFPGA25,
    author = {Basalama, Suhail and Cong, Jason},
    title = {Stream-HLS: Towards Automatic Dataflow Acceleration},
    year={2025},
    isbn = {9798400713965},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3706628.3708878},
    doi = {10.1145/3706628.3708878},
    booktitle = {Proceedings of the 2025 ACM/SIGDA International Symposium on Field Programmable Gate Arrays},
    numpages = {11},
    keywords = {accelerator, sptrsv, stream-based architecture},
    location = {Monterey, CA, USA},
    series = {FPGA '25}
}
```


## Acknowledgement
Some parts of Stream-HLS were modified, extended, and/or borrowed from [ScaleHLS](https://github.com/UIUC-ChenLab/scalehls).
