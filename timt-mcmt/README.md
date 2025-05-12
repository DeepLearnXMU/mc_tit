<!-- GETTING STARTED -->

# TIMT-MCMT
Code for Neural Network 2025 paper: Towards Better Text Image Machine Translation with Multimodal Codebook and Multi-stage Training


## OCRMT30K Dataset
Dataset can be found [here](https://github.com/DeepLearnXMU/mc_tit/tree/main/ocrmt30k_data)

## Install fairseq
```
cd mc_tit/timt-mcmt
pip install -e ./
```

## Training
### Stage 1
This stage is consistent with mc-tit.
### Stage 2
Add the following parameters to the mc-tit stage2 script:

`--mask-probability 0.15`
### Stage 3
Add the following parameters to the mc-tit stage3 script:

`--discriminator-max-step 20000`

### Stage 4
Add the following parameters to the mc-tit stage4 script:

```
--mask-probability 0.15
--discriminator-max-step 20000
```

## Inference
Inference is consistent with mc-tit.

<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []() -->

<p align="right">(<a href="#top">back to top</a>)</p>