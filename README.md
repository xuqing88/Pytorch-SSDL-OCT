# Semi-supervised-DL-method-for-Retinopathy-Automatic-Detection



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Package used](#installation)
* [Usage](#usage)
* [License](#license)




<!-- ABOUT THE PROJECT -->
## About The Project
The paper: **Automatic Detection of Retinopathy with Optical Coherence Tomography Images via a Semi-supervised Deep Learning Method**

A semi-supervised deep learning method for retinopathy detection with OCT images is proposed. 

<!-- =
***![BOE_result](https://github.com/xuqing88/Pytorch-Semi-supervised-DL-method-for-Retinopathy-Automatic-Detection/blob/master/result/BOE_result.JPG)
***![CELL_result](https://github.com/xuqing88/Pytorch-Semi-supervised-DL-method-for-Retinopathy-Automatic-Detection/blob/master/result/CELL_result.JPG)
-->


<!-- GETTING STARTED -->
## Getting Started

To get a local copy and runn following simple steps.

### Prerequisites

Download the dataset from below links and unzip the data files into folder `dataset`.
* [BOE](http://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm)
* [CELL](https://www.kaggle.com/paultimothymooney/kermany2018)


### Package used

```sh
 torch              1.6.0+cu101
 scikit-learn       0.21.3
 numpy              1.18.1
```

<!-- USAGE EXAMPLES -->
## Usage

1. Split the data into four subsets: train, validation, test and unlabel. The proposed method will use samples from train and unlabel folders for model training

```sh
python BOE_dataset_split.py
```

```sh
python CELL_dataset_split.py
```
2. Run the model training with the selected dataset

```sh
python VAT_semiDL_train.py -d 'BOE'
```



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.




