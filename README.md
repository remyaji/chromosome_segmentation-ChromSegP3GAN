# Chromosome_segmentation-ChromSegP3GAN
## Overview
The details of dataset, trained models and python codes are available here for the segmentation of overlapping chromosome clusters with two chromosomes.
### Details of Dataset
Clusters with exactly two chromosomes are generated from the G banded metaphase images. The mask images are prepared by the labelling tool. The mask images are preprocessed with the python code to get label with 4 unique values (0,1,2,3) in which '0' indicates the background, '1' indicates the chromosome 1, '2' indicates the chromosome 2 and the overlapping region is indicated by '3'.
These images are provided as .npz file, .h5 file and .png file in Mendeley data. These datasets can be accessed via  [MendeleyData](https://data.mendeley.com/drafts/h5b3zbtw8v).

### Codes
- [Python code for preprocessing labels](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/blob/main/preprocess_mask.ipynb)
- Python code for cGAN based models 
 
    cGAN with Attention UNet as generator is demonstrated [here](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/cGAN%20models). Other generators like R2UNet, R2AttentionUNet, NestedUNet and UNet are available in [util/networks.py](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/blob/main/cGAN%20models/util/networks.py). Training the models creats two .pth files in the checkpoint directory and upon testing the results are generated as shown in the folder [results](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/cGAN%20models/results). The quantitative results for  the tested model can be evaluated with [evaluation.ipynb](
    
- Python code for literature models. 

## References
<a id="1">[1]</a> R S, Remya; C, Gopakumar; S, Hariharan; Prasad, Hari (2022), “Overlapping and touching G banded chromosome dataset”, Mendeley Data, V1, doi: 10.17632/h5b3zbtw8v.1

<a id="2">[2]</a> R S, Remya; C, Gopakumar; S, Hariharan; Prasad, Hari (2023), “Overlapping and touching G banded chromosome dataset”, Mendeley Data, V2, doi: 10.17632/h5b3zbtw8v.2

<a id="3">[3]</a> Mei, L., Yu, Y., Shen, H., Weng, Y., Liu, Y., Wang, D., ... & Lei, C. (2022). Adversarial Multiscale Feature Learning Framework for Overlapping Chromosome Segmentation. Entropy, 24(4), 522.

<a id="4">[4]</a>https://github.com/liyemei/AMFL
