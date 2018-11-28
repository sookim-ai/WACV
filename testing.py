import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from inference import *
from load_data import *
from rnn import *
import numpy as np

def test(name,sess,loss_op,train_op,X,Y,test_X,test_Y,prediction,last_state,fout_log):
    test_size=len(test_X) 
    fetches = {'final_state': last_state,
              'prediction_image': prediction}
    output_image=[]
    for step in range(test_size):
        eval_out=sess.run(fetches, feed_dict={X:test_X[step]})
        output_image.append(eval_out['prediction_image'])
        print(eval_out['prediction_image'])
    np.save("test_result_"+name+".npy", output_image)




