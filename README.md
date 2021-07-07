# LBNL_Segmentation_crf

## Setup
### 1. Create a conda environment 
Conda is a great tool to import tools and create environments for projects. To avoid any conflicts with existing work or other versions of required dependencies, use conda to create a virtual environment. Make sure to activate the environment before proceeding to the next steps.
- Python Version: 3.7 (Untested on different python versions)

### 2. Clone the repository
`git clone --recurse-submodules https://github.com/mavaylon1/LBNL_Segmentation_crf.git`
- The main branch is allen.

### 3. Install dependencies
In the current state, if users want to use Conditional Random Fields (CRF) in their models, it must be trained on a CPU. 
- Keras API Version: 2.3.1

- Tensorflow Version: 2.2.0

- Keras from Tensorflow Version: 2.3.0-tf

The CPU can be used to train CRF and non-CRF models, but if a GPU is available, then users can train non-CRF models with the following installations.
- conda install -c anaconda keras-gpu

