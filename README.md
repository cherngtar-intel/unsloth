## 💾 Installation Instructions

## Setup for Intel XPU
Setup [Prerequisites](https://github.com/intel/intel-xpu-backend-for-triton?tab=readme-ov-file#prerequisites) of Intel XPU backend for Triton

Clone this repository:
```
git clone https://github.com/cherngtar-intel/unsloth.git -b unsloth_xpu_20241001
cd unsloth
```
To avoid potential conflicts with installed packages it is recommended to create and activate a new Python virtual environment:
```
python -m venv .venv --prompt unsloth
source .venv/bin/activate
```
Download and pip install [nightly wheel](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml) from Intel xpu backend for Triton (only require torch and triton):
```
pip install torch-*.whl triton-*.whl
```
Install all dependencies listed in the requirements text file:
```
pip install -r requirements.txt
```
Initialize the toolchain
```
# replace /opt/intel/oneapi with the actual location of Intel® Deep Learning Essentials
source /opt/intel/oneapi/setvars.sh
```

#### To run Unsloth inference with xpu
```
python unsloth_inference.py
```

#### To run Unsloth backend server with Intel AI Assistant Client Application
Install grpcio packages:
```
pip install grpcio==1.66.1
pip install grpcio-tools==1.66.1
```
To launch unsloth server:
```
python unsloth_server.py
```
Further steps from client side, refer to [fork repository Intel AI Assistant Client Application](https://github.com/kuanxian1/applications.ai.superbuilder/tree/kuanxian1/upmerge-v0.7.0.1102)
