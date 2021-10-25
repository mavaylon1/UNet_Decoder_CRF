import sys

from setuptools import setup, find_packages

# Some Python installations don't add the current directory to path.
if '' not in sys.path:
    sys.path.insert(0, '')
# print(find_packages('src'))
setup(
      name="UNet_Decoder_CRF",
      version="0.1.0",
      description="A package for semantic segmentation using CNNs and the CRF-RNN layer.",
      author="Matthew Avaylon",
      author_email='mavaylon@lbl.gov',
      platforms=["Mac OS and Linux"],
      license="",
      url="",
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
            "h5py<=2.10.0",
            "imageio",
            "imgaug",
            "opencv-python",
            "tqdm"],
      python_requires=">=3.6"
)
