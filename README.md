# PPCAN
Adversarial Neural Network for Privacy Preserving Communications

## Repository guide
In order to have a copy of the repository, you need to:
- Install git from here: https://git-scm.com/download
- Clone the repository using <code>git clone https://github.com/thomasmarchioro3/PPCAN.git</code>
- Download the folder <code>data/</code> from here https://www.dropbox.com/sh/izxq4c1uuu866gt/AADQ3-fyHMYb1D6kyF_bd06Ma
- Put the folder inside the main folder of the repository (i.e., <code>PPCAN</code>)

## Tensorflow guide
Once you have successfully downloaded the repository, you need to install Tensorflow.
The simplest way is to install it through Anaconda: 
- Be sure of having Python3 installed
- Download Anaconda from https://www.anaconda.com/ and install it
- Open the Anaconda Prompt and get the tensorflow GPU environment, using the commands <code>conda create -n tensorflow_gpuenv tensorflow-gpu</code> and <code>conda activate tensorflow_gpuenv</code> 
- Close the Anaconda Prompt
- Open the Anaconda Navigator and switch from <code>root</code> to <code>tensorflow_gpuenv</code>
- Install Jupyter Lab
- Open Jupyter Lab, open a new terminal and navigate to the path of the repository

## Usage guide
To run a training simulation, use either <code>python ./train.py</code> or <code>python ./trainv2.py</code>.
You can also set some parameters:
- The tradeoff parameter alpha: for example, to set alpha equal to 0.2 <code>python ./train.py --alpha 0.2</code> (default is 0.95)
- SNR of the legitimate channel: for example, to set the SNR to 20dB <code>python ./train.py --legit_channel_snr 20</code> (default is 10)
- SNR of the adversary's channel: for example, to set the SNR to -10dB <code>python ./train.py --adv_channel_snr -10</code> (default is 5)
