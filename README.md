# UNet_Decoder_CRF

## Setup (Compatible with NERSC Perlmutter)
### 1. Create a conda environment 
Conda is a great tool to import tools and create environments for projects. To avoid any conflicts with existing work or other versions of required dependencies, use conda to create a virtual environment. Make sure to activate the environment before proceeding to the next steps.
- Python Version: 3.8 
`conda create --name CRF_CPU_Env python=3.8`
`conda activate CRF_CPU_Env`

### 2. Clone the repository
-`git clone --recurse-submodules https://github.com/mavaylon1/UNet_Decoder_CRF.git`
-`pip install -e .`
-`-pip install protobuf==3.20.*`

### 3. Install cudatoolkit and cudnn via conda 
In the current state, if users want to use GPU for training their models, users need to install the following cudatoolkit and cudnn versions.
- `conda install cudatoolkit==11.3.1 cudnn==8.2.1` 

### 4. Install tensorflow v. 2.6
In the current state, tensorflow 2.6.0 is confirmed as the latest version to work with the cuda and cudnn versions above on Perlmutter GPU.
- `pip install tensorflow==2.6`

### 5. Downgrade keras
As of August 2022, the keras installed with tensorflow v. 2.6 is not compatible. The fix we found was to downgrade keras to 2.6:
- `pip install keras==2.6.*`

### 4. CRF Setup
From the cloned directory go to the src folder. 
- `cd src/UNet_Decoder_CRF/crf/cpp`
- `make`

If you run into issues with the CRF, refer to https://github.com/sadeepj/crfasrnn_keras and follow the demo.
Depending on the configuration, one may need to specify the absolute path to `high_dim_filter.so` in `high_dim_filter_loader.py` for tensorflow module load.
