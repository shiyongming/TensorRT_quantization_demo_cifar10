import pickle
from PIL import Image
import numpy as np

#
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

#
datafile = unpickle("test_batch")
for i in range(0, 10000):
    img = np.reshape(datafile['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    image = Image.fromarray(img)
    picName = 'whole_testset/' + str(datafile['labels'][i]) + '_' + str(i) + '.jpg'
    image.save(picName)
print("test_batch loaded.")
