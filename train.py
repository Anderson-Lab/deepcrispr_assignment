import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0,'.')

import deepcrispr as dc

file_path = 'paper_data/ontar/hela_hart.episgt' #examples/eg_cls_on_target.episgt'
input_data = dc.Episgt(file_path, num_epi_features=4, with_y=True)
x, y = input_data.get_dataset()
x = np.expand_dims(x, axis=2)  # shape(x) = [100, 8, 1, 23]
x = x.transpose([0, 2, 3, 1])
sess = tf.InteractiveSession()

model = dc.DCModel(sess)
model.train(x,y)
