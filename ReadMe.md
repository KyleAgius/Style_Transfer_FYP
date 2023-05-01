# Neural Style Transfer in Games and Videos

This repository is for the code developed during the Final Year Project. It is split into the following sections:

## MCCNet

This code is used for stylising segmented videos and for stylising game elements. It is built upon **Arbitrary Video Style Transfer via Multi-Channel Correlation** by Deng et al. Their paper and original code can be accessed [here](https://arxiv.org/pdf/2009.08003.pdf) and [here](https://github.com/diyiiyiii/MCCNet) respectively.

The main changes to the code are the following:
- ***test_video.py*** was altered to support segmentation. Add the *--segment* tag to the original command to create segmented images and videos. The tag *--default_mask* avoids using the improved mask while the tag *--use_inpaint* makes use of inpaint to better blend the segmented region.
- ***pytorchModel.py*** is a simplified version of *test_video.py* in the form of a pytorch model. It is used in later scripts.
- ***pytorchRunner.py*** is used to apply style transfer on an entire folder of images. The paths to the style and folder need to be adjusted within the script.

## Unity

This is the code to be run within Unity. It contains all the scripts needed to apply the developed method, as well as the ui to apply the style and view logs. Only the simple 2D scene is included since other scenes contain third party assets.

To run, the following should be done.
- Create and open a new Unity Project (Version 2021.3.4f1)
- Import the package *StyleTransfer* 
- Open 2D_Scene

