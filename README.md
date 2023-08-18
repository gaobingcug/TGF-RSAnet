# Exploring the integration of remote sensing images for assisting DEM super-resolution through a terrain-guided content-edge fusion network
It is the pytorch implementation of the paper: Exploring the integration of remote sensing images for assisting DEM super-resolution through a terrain-guided content-edge fusion network. Here is the first commit, more illustrations of the implementation will be released soon.
# Our environments
python==3.8.11
GDAL==2.4.1
scikit_image==0.18.1
matplotlib==3.4.2
numpy==1.22.4
kornia==0.6.1
torch==1.9.1
torchvision==0.10.0
# Dataset
* Description: This dataset included a part of 3× test sets, which were obtained from TanDEM-X and Sentienl-2. The dataset contains a total of 500 144*144 data pairs. Each data pair contains one HR DEM, one LR DEM and one RS image. Complete DEM and remote sensing images can be obtained from https://www.intelligence-airbusds.com and https://earthexplorer.usgs.gov/, respectively.
* Download the dataset from the following link: [Baidu cloud disk](https://pan.baidu.com/s/12S4BwgRoJsZoD0uZrrkqGQ?pwd=lxum) (code:lxum)
# pre-train model
* Description: We offer pre-trained models for 2×, 3× and 4× super-resolution pre-trained models, which were located in the checkpoints folder.
# Acknowledgement
Thanks for the excellent work and codebase of TfaSR! 
