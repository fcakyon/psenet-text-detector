[![PyPI version](https://badge.fury.io/py/psenet-text-detector.svg)](https://badge.fury.io/py/psenet-text-detector)
![CI](https://github.com/fcakyon/psenet-text-detector/workflows/CI/badge.svg)

## PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network
Packaged Version of the Pytorch implementation of PSENet text detector

### Overview
PSENet is designed as a segmentation-based detector with multiple predictions for each text instance. These predictions correspond to different `kernels' produced by shrinking the original text instance into various scales. Consequently, the final detection can be conducted through our progressive scale expansion algorithm which gradually expands the kernels with minimal scales to the text instances with maximal and complete shapes.

<img width="1000" alt="teaser" src="./figures/psenet_demo.png">

## Getting started
### Installation
- Install using conda for Linux, Mac and Windows (preferred):
```console
conda install -c fcakyon psenet-text-detector
```
- Install using pip for Linux and Mac:
```console
pip install psenet-text-detector
```

### Basic Usage
```python
# import package
import psenet_text_detector as psenet

# set image path and export folder directory
image_path = 'figures/idcard.png'
output_dir = 'outputs/'

# apply craft text detection and export detected regions to output directory
prediction_result = psenet.detect_text(image_path, output_dir, cuda=False)
```

### Advanced Usage
```python
# import package
import psenet_text_detector as psenet

# set image path and export folder directory
image_path = 'figures/idcard.png'
output_dir = 'outputs/'

# read image
image = psenet.read_image(image_path)

# load model
psenet_model = psenet.load_psenet_model()

# perform prediction
prediction_result = psenet.get_prediction(image=image,
                               		  model=psenet_model,
                                       	  binary_th=1.0,
                                       	  kernel_num=3,
                                       	  upsample_scale=1,
                                       	  long_size=1280,
                                       	  min_kernel_area=10.0,
                                       	  min_area=300.0,
                                       	  min_score=0.93,
                                       	  cuda=True)

# export detected text regions
exported_file_paths = psenet.export_detected_regions(image_path,
                                              	    image,
                                              	    boxes=prediction_result["boxes"],
                                              	    output_dir=output_dir)

# export box visualization
_ = psenet.visualize_detection(image_path,
            		       image=image,
        		       quads=prediction_result["boxes"],
                    	       output_dir=output_dir)
```
