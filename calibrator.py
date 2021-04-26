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
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np


def data_Normalize(batch_data, mean_list=[0.4914, 0.4822, 0.4465], std_list=[0.2023, 0.1994, 0.2010]):
    batch_data[0, :, :] = (batch_data[0, :, :] - mean_list[0]) / std_list[0]
    batch_data[1, :, :] = (batch_data[1, :, :] - mean_list[1]) / std_list[1]
    batch_data[2, :, :] = (batch_data[2, :, :] - mean_list[2]) / std_list[2]
    return batch_data

# Returns a numpy buffer of shape (num_images, 1, 32, 32)
def load_mnist_jpeg_images(folder_path, total_images=1):
    image_idx = 0
    calib_images = np.zeros((total_images, 3, 32, 32))
    for filename in os.listdir(folder_path):
        print(folder_path + filename)
        img = Image.open(folder_path + filename)
        calib_images[image_idx] = np.array(img).transpose(2, 0, 1) / 255
        # calib_images[image_idx] = (calib_images[image_idx]-0.45)/0.20
        calib_images[image_idx] = data_Normalize(calib_images[image_idx])
        image_idx = image_idx + 1
    print('total_images:', total_images, 'calib_images shape:', calib_images.shape)
    return np.ascontiguousarray(calib_images.astype(np.float32))


class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    # class MNISTEntropyCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, training_data, cache_file, total_images, batch_size=1):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)
        # trt.IInt8MinMaxCalibrator.__init__(self)

        # Prepare data for the the following
        self.data = load_mnist_jpeg_images(training_data, total_images)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        # Print log for every 10 batches
        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
