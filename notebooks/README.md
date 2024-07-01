# QCD School 2024 ML Exercise: Anomaly detection in high energy physics

<img src="images/front.png" alt="AD" width="300" img align="right"/>

In this cool tutorial we will demonstrate how to design a tiny autoencoder (AE) that we will use for anomaly detection in particle physics. More specifically, we will demonstrate how we can use autoencoders to select potentially New Physics enhanced proton collision events in a more unbiased way!

We will train the autoencoder to learn to compress and decompress data, assuming that for highly anomalous events, the AE will fail.

## Dataset

As a dataset, we will use the CMS Open data that you have been made familiar with already.

We will train on a QCD MC dataset (we could also train directly on data), and evaluate the AE performance on a New Physics simulated sample: A Bulk graviton decaying to two vector bosons: G(M=2 TeV) → WW

We'll train using background data only and test using both background and the Graviton sample. Let's fetch them! The background data are available [here](https://opendata.cern.ch/record/63168) (recid = 63168) and the signal data [here](https://opendata.cern.ch/record/33703) (recid = 33703). The signal consists of 1,37M events and the background 19,279M events. We will use roughly 500K for each process.


## Model compression: Quantization aware training with QKeras
<img src="images/nmi_qkeras_hls4ml.jpeg" alt="hls4ml and qkeras" width="200" img align="left"/>

There are some cheap tricks you can perform to compress the model. These are pruning and quantization-aware-training and both are very easily implemented. Let's have a look.

To quantize the model during training, such that the network will get the opportunity to adapt to the narrower bitwidth we use the library [QKeras](https://www.nature.com/articles/s42256-021-00356-5.epdf?sharing_token=A6MQVmmncHNyCtDUXzrqtNRgN0jAjWel9jnR3ZoTv0N3uekY-CrHD1aJ9BTeJNRfQ1EhZ9jJIhgZjfrQxrmxMLMZ4eGzSeru7-ASFE-Xt3NVE6yorlffwUN0muAm1auU2I6-5ug4bOLCRYvA0mp-iT-OdPsrBYeH0IHRYx0t3wc%3D), developed in a joint effort between CERN and Google.

## Deployment on FPGA with

<img src="https://gitlab.cern.ch/fastmachinelearning/cms_mlatl1t_tutorial/-/raw/master/part2/images/hls4ml_logo.png?ref_type=heads" width="400"/>

We will translate this model into FPGA firmware using hls4ml (see ([here](https://github.com/fastmachinelearning/hls4ml-tutorial/tree/main)).
hls4ml seamlessly talks to QKeras, making our jobs way easier for us, but there is still some work for us to do to make sure we get good hardware model accuracy. 

## Getting started

We only assume that you have followed the previous QCD School tutorials, and especially set up the python container. You can start it again with
```
$ docker start -i qcd-school
cmsusr@40ffea690e46:/code$ jupyter-lab --ip=0.0.0.0 --no-browser
```
Then clone this repository:
```
git clone
```