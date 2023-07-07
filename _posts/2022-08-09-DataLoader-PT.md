---
layout: post
title: PyTorch Dataset, DataLoaders and DataPipes
---
This article explains how PyTorch seamlessly handles data for us to be able to train our large Deep Learning models efficiently without wasting the GPUs.

![While the model trains in the main process, data loading happens in parallel via the sub-processes.](https://github.com/srishti-git1110/srishti-git1110.github.io/blob/master/images/dataloader-generator.png?raw=true)

Article orginially published on the Weights & Biases blog: [How To Eliminate the Data Processing Bottleneck With PyTorch](https://wandb.ai/srishti-gureja-wandb/posts/reports/How-To-Eliminate-the-Data-Processing-Bottleneck-With-PyTorch--VmlldzoyNDMxNzM1)
