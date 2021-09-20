import sys

from setuptools import setup, find_packages

# Some Python installations don't add the current directory to path.
if '' not in sys.path:
    sys.path.insert(0, '')

setup(
      name="U-Dec",
      version="0.1.0",
      description="A package for semantic segmentation using CNNs and the CRF-RNN layer.",
      author="Matthew Avaylon",
      author_email='mavaylon@lbl.gov',
      platforms=["Mac OS and Linux"],
      license="",
      url="",
      packages=find_packages(),
      install_requires=[
            "h5py<=2.10.0",
            "Keras>-2.3.1",
            "Tensorflow==2.2.0"
            "imageio==2.5.0",
            "imgaug>=0.4.0",
            "opencv-python",
            "tqdm"],
      python_requires=">=3.6"
)
