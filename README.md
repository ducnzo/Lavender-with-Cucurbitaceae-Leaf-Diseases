# Detect Cucurbitaceae Leaf Disease  

## Table of content
## General 
I'm in using Ubuntu 20.04 
This is tutorial for training model using EfficientNet on GPU with Tensorflow.

My system:

Ubuntu 20.04

Ram: 16GB

GPU: GTX 2080


[Ref](https://phoenixnap.com/blog/future-gpu-machine-learning-ai)

## Requirement of resources
1. CUDA toolkit v11.8 and cuDNN v8.6 
2. Anaconda 
3. Tensorflow


## Step 1: Install CUDA toolkit v11.8 and cuDNN v8.6
### General 
Version TF 2.13 use CUDA v11.8 and cuDNN v8.6. Follow [tested build configurations](https://www.tensorflow.org/install/source#gpu)

### 1.1. CUDA toolkit v11.8

Click [CUDA toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) to download CUDA toolkit v11.8.

Choose Linux x86_64 Ubuntu version 20.04

Use runfile(local) to auto update suitable Graphics Drivers for your device. 

or 

Run the below commands in a terminal:

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```

```bash
sudo sh cuda_11.8.0_520.61.05_linux.run
```
#### Addition
If your system existed NVIDIA driver Xrog, you need to disable to install new driver from CUDA toolkit installation. Follow [link](https://docs.nvidia.com/ai-enterprise/deployment-guide-vmware/0.1.0/nouveau.html)


> **Note**: To remove completely CUDA toolkit: 

``````bash 
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get remove --purge '^libnvidia-.*'
sudo apt-get remove --purge '^cuda-.*'
``````
then
``````bash
sudo apt-get install linux-headers-$(uname -r)
``````

### 1.2. cuDNN v8.6

Click [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) to download cuDNN v8.6. 

Then choose cuDNN v8.6.0 for CUDA 11.x and download "Local Installer for Linux x86_64 (Tar)"

Navigate to the location where the installation file was saved. My location is "Downloads"

``````bash
cd Downloads
``````

Run the below commands in a terminal:

``````bash
tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
``````
``````bash
sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn*.h /usr/local/cuda/include 
``````
``````bash
sudo cp -P cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64 
``````
``````bash
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
``````
> **Note**: You can modify the name of `filename.tar.xz` suitable with the name of installation file you dowloaded. 

### 1.3. Set PATH   

``````bash
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
``````
``````bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
``````
```bash
source ~/.bashrc
```
#### To check version 
``````bash
nvcc --version
``````

```bash 
nvidia-smi
```


<!-- ```c++
#include <iostream>

int main()
{
    return 0;
}
``` -->

<!-- Table

|something|something|
|--------|----------|
|100|1000| -->

<!-- ![Fig.1](99504f8bafca709429db.jpg) -->

<!-- <table>
  <tr>
    <td> <img src="8efaabbfb59675c82c87.jpg"  alt="1" width = 480px height = 640px ></td>
    <td> <img src="8efaabbfb59675c82c87.jpg"   alt="2" width = 480px height = 640px> </td>
   </tr> 
</table> -->

## Step 2: Install Anaconda
### General
Anaconda is an open-source distribution of Python and R programming languages used for data science, machine learning, and artificial intelligence projects. It comes with various pre-installed libraries and packages that are useful for scientific computing, data analysis, and data visualization. [ref](https://www.tutorialspoint.com/how-to-install-anaconda-on-ubuntu-18-04-and-20-04)
### 2.1. Download 

Download [Anaconda](https://www.anaconda.com/) and choose the right OS. 

### 2.2. Install Anaconda
Once download is complete, navigate to directory where you saved Anaconda installation file. My directory is "Downloads". 
Notice your name of downloaded installation file to replace the name of file in below commands if necessary. 

Run the below commands in a terminal:

``````bash
cd Downloads
``````
``````bash
chmod +x Anaconda3-2023.07-2-Linux-x86_64.sh
``````
``````bash
./Anaconda3-2023.07-2-Linux-x86_64.sh
``````
After two times confirm "yes", run:

``````bash
source ~/.bashrc
``````
Check installation

```bash
conda list
```
This will display a list of all packages and libraries installed by default with Anaconda.

Using Navigator in base environment, run: 

```bash
conda anaconda-navigator
```
Follow [Getting started with Navigator](https://docs.anaconda.com/free/navigator/getting-started/)
### 2.3. Create an environment

[ref](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)

Build an identical conda environment to easily apply in project with suitable version of package. Easy to manage environment for specific project.

1. To create an evironment. Run:
```bash
conda create -n name_envir
```
> **Note**: Replace `name_envir` with name you want to give to your environment. 

OR

To create an environment with a specific version of Python:

```bash
conda create -n name_envir python==3.8
```

2. When conda asks you to proceed, type `y`:

``````
proceed ([y]/n)
``````
This creates the `name_envir` environment in /envs/. No packages will be installed in this environment.


3. To activate environment, run: 
```bash
conda activate name_envir
```
> A base environment was automatically set up when you open terminal. It leads to difficulty in using other functions without Anaconda. Therefore, we can [Turn Off](https://bobbyhadz.com/blog/deactivate-and-disable-anaconda-base-environment) this feature.


### 2.4. Using Jupyter Notebook on Visual Studio Code 
Follow [link](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) to set up Jupyter Notebook on Visual Studio Code. 

if you don't have installed Visual Studio Code, let's open [Navigator](#22-install-anaconda) and install Visual Studio Code in Navigator or download from [website](https://code.visualstudio.com/download).

Install jupyter in anaconda, run in terminal:  
```bash
pip install jupyter
```

### 2.5. Using Conda Virtual Environments 

To activate virtual environment

```bash
conda activate name_envir
```
To deactivate virtual environment
```bash
conda deactivate 
```
To list all available virtual environments
```bash
conda env list
```
To delete a virtual environment −
```bash
conda env remove --name name_envir
```

To install packages with Pip or Conda
```bash
pip install package_name
```
OR

```bash
conda install package_name
```
### Uninstall Anaconda 

If you want to uninstall Anaconda from your system, you can do so by running following command

```bash
rm -rf ~/anaconda3
```

## Step 3: Install Tensorflow and necessary package
Tensorflow v2.13.0, tflite-model-maker

> After creating a new conda environment, you need to activate that environment where you want to install Tensorflow and necessary packages. This project use Python v3.8. 

1. Check pip version and update pip

TensorFlow requires a recent version of pip, so upgrade your pip installation to be sure you're running the latest version.

```bash
pip --version
pip install --upgrade pip
```

2. Install Tensorflow with pip

```bash 
pip install tensorflow==2.13.*
```
> **Note**: Do not install TensorFlow with conda. It may not have the latest stable version. pip is recommended since TensorFlow is only officially released to PyPI.

Since version 1.24 of numpy, np.object is deprecated so if exist error, please downgrade the version of nympy to 1.23.5.

```bash 
pip install numpy==1.23.5
```

Install tflite-model-maker:

```bash
pip install tflite-model-maker
```
