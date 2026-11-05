# Triton Ascend Ops

## News

## About
Since the micro architecture of AI chips are different from different vendors, so is Huawei Ascend, which is quite different from NV GPU. To make Triton ops perform well, we usually need to do some code changes on the code with NV GPU. The main purpose of this repo is:
* providing some typical ops, which have already been fine-tuned on Huawei Ascend
* providing some tutorials, which show the tips and Triton DSL extension on Huawei Ascend

### ops

### Tutorial
Three kinds of example are provided as following:
* basic examples
* best practice
* DSL extension

## How run it 

#### Prerequisite

* CANN
* Pytorch and Pytorch Ascend
* Triton Ascend

<p style="color: #6b7280;"><sub>About Pytorch Ascend: TorchNPU is a deep learning adapter plugin built by Ascend for PyTorch.</sub></p>

<p style="color: #6b7280;"><sub>About Triton Ascend: Triton language and compiler for Huawei Ascend NPU.</sub></p>


## License

Follow MIT license, check LICENSE file
