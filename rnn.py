from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn

h=128
w=257
feature_size=w*h; channels=3
batch_size=24
display_step=10;
testing_step=100;
training_steps = 200000
# Network Parameters
number_of_layers=2; #Start from only one layer

def ConvLSTM(x):
    convlstm_layer1 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,3],
                 output_channels=6,
                 kernel_shape=[3,3],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=1.0,
                 initializers=None,
                 name="conv_lstm_cell1")
    convlstm_layer2 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,6],
                 output_channels=12,
                 kernel_shape=[3,3],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=1.0,
                 initializers=None,
                 name="conv_lstm_cell2")
    convlstm_layer3 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,12],
                 output_channels=1,
                 kernel_shape=[3,3],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=1.0,  
                 initializers=None,
                 name="conv_lstm_cell3")
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [convlstm_layer1, convlstm_layer2, convlstm_layer3])
    print(stacked_lstm)
    initial_state=stacked_lstm.zero_state(batch_size, dtype=tf.float32 )
    outputs,states=tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=x, sequence_length=None, dtype=tf.float32, initial_state=initial_state)
    return outputs, states
# Didn't apply drop_out here

