# AdaBits: Neural Network Quantization with Adaptive Bit-widths
# SAT: Neural Network Quantization with Scale-Adjusted Training

SAT [arXiv](https://arxiv.org/abs/1912.10207) [BMVC2020](https://www.bmvc2020-conference.com/assets/papers/0634.pdf) | AdaBits [arXiv](https://arxiv.org/abs/1912.09666) [CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_AdaBits_Neural_Network_Quantization_With_Adaptive_Bit-Widths_CVPR_2020_paper.html) | [Model Zoo](#model-zoo) | [BibTex](#citing)

<!--
<img src="https://user-images.githubusercontent.com/22609465/50390872-1b3fb600-0702-11e9-8034-d0f41825d775.png" width=95%/>
-->

Illustration of neural network quantization with scale-adjusted training and adaptive bit-widths. The same model can run at different bit-widths, permitting instant and adaptive accuracy-efficiency trade-offs.


## Run

0. Requirements:
    * python3, pytorch 1.0, torchvision 0.2.1, pyyaml 3.13.
    * Prepare ImageNet-1k data following pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet).
1. Training and Testing:
    * The codebase is a general ImageNet training framework using yaml config under `apps` dir, based on PyTorch.
    * To test, download pretrained models to `logs` dir and directly run command.
    * To train, comment `test_only` and `pretrained` in config file. You will need to manage [visible gpus](https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) by yourself.
    * Command: `python train.py app:{apps/***.yml}`. `{apps/***.yml}` is config file. Do not miss `app:` prefix.
2. Still have questions?
    * If you still have questions, please search closed issues first. If the problem is not solved, please open a new.


## Model Zoo

All models are available at /mnt/cephfs\_new\_wj/uslabcv/jinq/qnn/imagenet/checkpoints/

All settings are

fp\_pretrained=True

weight\_only=False

rescale=True

clamp=True

rescale\_conv=False

For all adaptive models, switchbn=True, switch\_alpha=True


## Technical Details

Implementing network quantization with adaptive bit-widths is straightforward:
  * Switchable batchnorm and adaptive bit-widths layers are implemented in [`models/quant_ops`](/models/quant_ops.py).
  * Quantization training with adaptive bit-widths is implemented in the [`run_one_epoch`] function in [`train.py`](/train.py).


## Acknowledgement
This repo is based on [slimmable\_networks](https://github.com/JiahuiYu/slimmable_networks) and benefits from the following projects
  * [Neural Network Distiller](https://github.com/NervanaSystems/distiller)


## License

CC 4.0 Attribution-NonCommercial International

The software is for educaitonal and academic research purpose only.


## Citing
```
@article{jin2019quantization,
  title={Towards efficient training for neural network quantization},
  author={Jin, Qing and Yang, Linjie and Liao, Zhenyu},
  journal={arXiv preprint arXiv:1912.10207},
  year={2019}
}
@article{jin2020sat,
  title={Neural Network Quantization with Scale-Adjusted Training},
  author={Jin, Qing and Yang, Linjie and Liao, Zhenyu and Qian, Xiaoning},
  booktitle={The British Machine Vision Conference},
  year={2020}
}
@article{jin2020adabits,
  title={AdaBits: Neural Network Quantization with Adaptive Bit-Widths},
  author={Jin, Qing and Yang, Linjie and Liao, Zhenyu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
