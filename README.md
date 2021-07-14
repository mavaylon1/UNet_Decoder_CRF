# LBNL_Segmentation_crf

## Setup
### 1. Create a conda environment 
Conda is a great tool to import tools and create environments for projects. To avoid any conflicts with existing work or other versions of required dependencies, use conda to create a virtual environment. Make sure to activate the environment before proceeding to the next steps.
- Python Version: 3.7 (Untested on different python versions)
`conda create --name CRF_CPU_Env python=3.7 `
`conda activate CRF_CPU_Env`

### 2. Clone the repository
`git clone --recurse-submodules https://github.com/mavaylon1/LBNL_Segmentation_crf.git`
- The main branch is allen.

### 3. Install dependencies
In the current state, if users want to use Conditional Random Fields (CRF) in their models, it must be trained on a CPU. 
- `conda install keras` 
- `pip install tensorflow==2.2.0`
- `conda install tqdm`
- `pip install numpy`
- `conda install imgaug`
- `conda install -c conda-forge opencv`
- `conda install jupyter notebook`

Note: Keras API Version: 2.3.1 is confirmed to work.

The CPU can be used to train CRF and non-CRF models, but if a GPU is available, then users can train non-CRF models with the following installations.
- conda install -c anaconda keras-gpu

### 4. Optional: CRF Setup
From the cloned directory go to the src folder.
- `cd src/cpp`
- `make`

If you run into issues with the CRF, refer to https://github.com/sadeepj/crfasrnn_keras and follow the demo.
