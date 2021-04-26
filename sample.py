#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorrt as trt

import numpy as np
import pickle

# For our custom calibrator
from calibrator import MNISTEntropyCalibrator

# For ../common.py
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common

#TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
TRT_LOGGER = trt.Logger()

# This function builds an engine from a Caffe model.
def build_int8_engine(onnx_file_path, calib, batch_size=32):
    # with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, trt.OnnxParser(network,TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size

        config.max_workspace_size = common.GiB(1)
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        config.int8_calibrator = calib

        # Parse Onnx model
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        network.get_input(0).shape = [batch_size, 3, 32, 32]

        # Decide which layers fallback to FP32.
        # If all layers fallback to FP32, you can use 'index>-1'
        for index, layer in enumerate(network):
            print('layer index', index, ':', layer.type)
            if index < 10:
                if layer.type == trt.LayerType.ACTIVATION or \
                        layer.type == trt.LayerType.CONVOLUTION or \
                        layer.type == trt.LayerType.FULLY_CONNECTED or \
                        layer.type == trt.LayerType.SCALE:
                    print('fallback to fp32!')
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)

        # Build engine and do int8 calibration.
        return builder.build_engine(network, config)


def check_accuracy(context, batch_size, test_set, test_labels):
    inputs, outputs, bindings, stream = common.allocate_buffers(context.engine)

    num_correct = 0
    num_total = 0
    print('test_set.shape[0]', test_set.shape[0])
    batch_num = 0
    for start_idx in range(0, test_set.shape[0], batch_size):
        batch_num += 1
        if batch_num % 10 == 0:
            print("Validating batch {:}".format(batch_num))
        # If the number of images in the test set is not divisible by the batch size, the last batch will be smaller.
        # This logic is used for handling that case.
        end_idx = min(start_idx + batch_size, test_set.shape[0])
        effective_batch_size = end_idx - start_idx

        # Do inference for every batch.
        inputs[0].host = test_set[start_idx:start_idx + effective_batch_size]
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream,
                                       batch_size=effective_batch_size)
        # Use argmax to get predictions and then check accuracy
        preds = np.argmax(output[0:effective_batch_size * 10].reshape(effective_batch_size, 10), axis=1)
        labels = test_labels[start_idx:start_idx + effective_batch_size]
        # print(preds)
        # print(labels)
        # print()
        num_total += effective_batch_size
        num_correct += np.count_nonzero(np.equal(preds, labels))
    percent_correct = 100 * num_correct / float(num_total)
    print("Total Accuracy: {:}%".format(percent_correct))


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def batch_data_Normalize(batch_data, mean_list=[0.4914, 0.4822, 0.4465], std_list=[0.2023, 0.1994, 0.2010]):
    for i in range(batch_data.shape[0]):
        batch_data[i, 0, :, :] = (batch_data[i, 0, :, :] - mean_list[0]) / std_list[0]
        batch_data[i, 1, :, :] = (batch_data[i, 1, :, :] - mean_list[1]) / std_list[1]
        batch_data[i, 2, :, :] = (batch_data[i, 2, :, :] - mean_list[2]) / std_list[2]
    return batch_data


def load_cifar_data(cifar10_path):
    testXtr = unpickle(cifar10_path + "test_batch")
    images = np.vstack(testXtr['data']).reshape(-1, 3, 32, 32) / 255
    # images = (images - 0.45)/0.20
    images = batch_data_Normalize(images)
    images = np.ascontiguousarray((images).astype(np.float32))
    labels = np.array(testXtr['labels'])
    labels = np.ascontiguousarray(labels.astype(np.int32).reshape(-1))
    return images, labels


def main():
    ONNX_PATH = "resnet18.onnx"
    cifar10_data_path = '/workspace/hostdir/cifar10_dataset/'
    calib_data_path = '/workspace/hostdir/cifar10_dataset/calib_dataset_40/'
    # calib_data_path = '/workspace/hostdir/cifar10_dataset/test/'

    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    calibration_cache = "mnist_calibration.cache"
    calib = MNISTEntropyCalibrator(calib_data_path, total_images=40, batch_size=10, cache_file=calibration_cache)

    # Inference batch size can be different from calibration batch size.
    batch_size = 32
    with build_int8_engine(ONNX_PATH, calib, batch_size) as engine, engine.create_execution_context() as context:
        # Batch size for inference can be different than batch size used for calibration.
        test_set, test_labels = load_cifar_data(cifar10_data_path)
        check_accuracy(context, batch_size, test_set=test_set, test_labels=test_labels)


if __name__ == '__main__':
    main()
