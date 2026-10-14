# CooperativePathways

Analysis code related to the manuscript: **"Cooperative processing of visual threat across collicular pathways"** *(submitted)*.

This repository contains the analysis scripts used to process data and generate the figures presented in the study. 

### Installation

The code has been tested on Windows 10 using Python 3.11. To set up the environment and install the required dependencies:
    ```bash
    conda create -n coop_pathways python=3.11 -y
    conda activate coop_pathways
    pip install -r requirements.txt
    ```

### Usage Instructions
Jupyter notebooks used for the paper's figures are located in the notebooks/ directory, organized into subfolders by figure.

* Data Pipeline: Raw data is managed via an internal DataJoint pipeline and exported to .npy files for analysis.

* Data Availability: To reconstruct the figures without access to the DataJoint database, use the exported .npy datasets, which will be made available on Zenodo.

* Running the Code: Open the desired notebook inside notebooks/, update the data and figure paths, and run the notebook cell by cell.

Expected output: Individual figure panels and statistical outputs as presented in the paper.

### Contact
Bram Nuttin bram.nuttin@kuleuven.be
