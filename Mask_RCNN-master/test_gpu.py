import tensorflow as tf 

if tf.compat.v1.test.is_gpu_available: 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
