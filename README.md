# TensorRT_quantization_demo_cifar10
This demo is to show how to build a Resnet18 TensorRT int8 engine. And demonstrate how the size of calibrate dataset influences the final accuracy.

This demo is derived from a TensorRT python sample: int8_caffe_mnist.

To demonstrate how the calibrate dataset size influences the accuracy after int8 quantization, 
the mnist dataset was changed into cifar10 dataset, and the LeNet was changed into ResNet18.

### Run it step by step
0. ```pip install -r requirements.txt``` 
1. You need to change the `ONNX_PATH` (line 123 in sample.py) into your own path where you save the `resnet18.onnx`.
2. You need to change the `cifar10_data_path` (line 124 in sample.py) into your own path where you save the cifar10 test data`test_batch`.
3. You need to change the `calib_data_path` (line 125 in sample.py) into your own path where you save data for cailbration.
4. `total_images` and `batch_size` (line 131 in sample.py) are the total images number you used for calibration and batch size for loading the calibration data. 
   They should also be changed.
    
5. If you want to use the whole test dataset to do the calibration. You can use `convert_to_images.py` to convert the cifar10 `test_batch` file into jpeg images. 
   Note that to change the path into your own path.

6. ```python sample.py```


###Results
Before quanztization, the orignal top-1 accruacy on test_batch is 87.81%. After quantization, the top-1 accuracy is shown as below.

![img.png](img.png)

'10+10' images means that we add another 10 images into the existed '10' images. And '10+10+10' means we add another 10 images into the existed '10+10' images. 

'20' and '30' images means the calibration image set was selected randomly.

To evaluate the quality of calibration set. I adopt my another repo [calib-dataset-eval](https://github.com/shiyongming/TensorRT_quantization_demo_cifar10) 
to calculate and analysis the distribtion of the calibration dataset.

<img src="cifar10_data/calib_dataset_10.png" width="200" alt="微信小程序"/>

It can be seen that, 