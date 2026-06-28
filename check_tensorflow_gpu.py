import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check the details of the GPU being used (if available)
if tf.config.experimental.list_physical_devices('GPU'):
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    print("TensorFlow is using the GPU:", gpu_devices)
else:
    print("TensorFlow is not using the GPU.")