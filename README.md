# Chromosome segmentation-ChromSegP3GAN
## Overview
The details of dataset, trained models and python codes are available here for the segmentation of overlapping chromosome clusters with two chromosomes.
### Details of Dataset
Clusters with exactly two chromosomes are generated from the G banded metaphase images. The mask images are prepared by the labelling tool. The mask images are preprocessed with the python code to get label with 4 unique values (0,1,2,3) in which '0' indicates the background, '1' indicates the chromosome 1, '2' indicates the chromosome 2 and the overlapping region is indicated by '3'.
These images are provided as .npz file, .h5 file and .png file in Mendeley data. These datasets can be accessed via  [MendeleyData](https://data.mendeley.com/drafts/h5b3zbtw8v).

### Codes
- [Python code for preprocessing labels](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/blob/main/preprocess_mask.ipynb)
- [Python code for cGAN based models](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/cGAN%20models) 
 
    cGAN with Attention UNet as generator is demonstrated [here](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/cGAN%20models). Other generators like R2UNet, R2AttentionUNet, NestedUNet and UNet are available in [util/networks.py](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/blob/main/cGAN%20models/util/networks.py). Training the models creats two .pth files in the checkpoint directory and upon testing the results are generated as shown in the folder [results](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/cGAN%20models/results). The quantitative results for  the tested model can be evaluated with [evaluation.ipynb](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/blob/main/cGAN%20models/evaluation.ipynb).
    
- [Python code for Non-GAN models](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/SegmentationModels(Non-GAN))

  Certain classical semantic segmentation models like UNet, SegNet and other state-of-the-art models like Attention UNet, R2UNet, R2Attention UNet, Nested UNet[3] are extended for our datasets and are available [here](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/SegmentationModels(Non-GAN)). Definitions of Attention UNet, R2UNet, R2Attention UNet  are available in [Nested_UNet.ipynb](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/blob/main/SegmentationModels(Non-GAN)/Nested_UNet.ipynb) and can be trained by necessary modifications in the same code (the replacement of model names as specified in the definitions are just required). The quantitative and qualitative results are demonstrated in the same code. Please note the codes [UNet.ipynb](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/blob/main/SegmentationModels(Non-GAN)/UNet.ipynb) and [SegNet.ipynb](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/blob/main/SegmentationModels(Non-GAN)/SegNet.ipynb) are the extended versions of the references [6] and [7] respectively. 
  
- [Python code for ChromSegP3GAN model](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/ChromSegP3GAN). 

   This model is proposed for efficient segmentation of overlapping and touching G banded chromosomes. The code [ChromSeg_P3GAN_train]() and [ChromSeg_P3GAN_test]() are respectively for training and testing the models. Intermediate qualitative results are saved in [plot](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/ChromSegP3GAN/plot) and the models at specified intervals are saved in [model](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/ChromSegP3GAN/model). The loss values during the training are also saved as shown in [csv](https://github.com/remyaji/chromosome_segmentation-ChromSegP3GAN/tree/main/ChromSegP3GAN/csv). 
   
   #Summary 
   
    ChromSeg-P3GAN, an efficent GAN based model can be used for the segmnetation of overlappin and touching G banded chromosomes. 
    
    
## References
<a id="1">[1]</a> R S, Remya; C, Gopakumar; S, Hariharan; Prasad, Hari (2022), “Overlapping and touching G banded chromosome dataset”, Mendeley Data, V1, doi: 10.17632/h5b3zbtw8v.1

<a id="2">[2]</a> R S, Remya; C, Gopakumar; S, Hariharan; Prasad, Hari (2023), “Overlapping and touching G banded chromosome dataset”, Mendeley Data, V2, doi: 10.17632/h5b3zbtw8v.2

<a id="3">[3]</a> Mei, L., Yu, Y., Shen, H., Weng, Y., Liu, Y., Wang, D., ... & Lei, C. (2022). Adversarial Multiscale Feature Learning Framework for Overlapping Chromosome Segmentation. Entropy, 24(4), 522.

<a id="4">[4]</a>https://github.com/liyemei/AMFL

<a id="5">[5]</a>https://github.com/LeeJunHyun/Image_Segmentation

<a id="6">[6]</a>https://github.com/LilyHu/image_segmentation_chromosomes

<a id="7">[7]</a>https://github.com/ArkaJU/SegNet---Chromosome
