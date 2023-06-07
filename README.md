# OtarNet

Optimal Target-oriented Knowledge Transportation For Aspect-Based Multimodal Sentiment Analysis



Author

## Requirement

* PyTorch 1.0.0
* Python 3.7


## Download tweet images and set up image path
- Step 1: Download datasets of Twitter-15 and Twitter-17 in the link (https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view) 
- Step 2: Change the arguments of image_path in "run_multimodal_classifier.py" file
- Step 3: Download the pre-trained weights of ResNet-152  via  (https://download.pytorch.org/models/resnet152-b121ed2d.pth)
- Setp 4: Put the pre-trained ResNet-152 model under the folder named "resnet"



## Code Usage

### (Optional) Preprocessing
- This is optional, because I have provided the pre-processed data under the folder named "absa_data"

```sh
python process_absa_data.py
```

### Training for OTarNet
- This is the training code of tuning parameters on the dev set, and testing on the test set. Note that you can change "CUDA_VISIBLE_DEVICES" based on your available GPUs.

```sh
sh run_multimodal_classifier.sh
```

### Testing for OTarNet
- After training the model, the following code is used for directly loading the trained model and testing it on the test set

```sh
sh run_multimodal_classifier_test.sh
```


