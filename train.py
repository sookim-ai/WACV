import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from inference import *
from load_data import *
from rnn import *
import numpy as np
import random

def train(sess,loss_op,train_op,X,Y,train_X,train_Y,val_X,val_Y,prediction, last_state,fout_log):
    val_best_loss=10; val_best_step=0;
    alpha=0.9
    count=0
    train_size=len(train_X) 
    val_size=len(val_X)
    saver = tf.train.Saver()
    fetches = {'final_state': last_state,
              'prediction_lonlat': prediction}
    for epoch in range(1):
        for step in range(train_size):
            stepp=int(epoch)*(train_size)+step
            print("train step "+str(stepp))
            train=sess.run(train_op, feed_dict={X:train_X[step], Y: train_Y[step]})
            # Calculate batch loss in validation set
            if stepp%5 == 0:
                val_sum=0
             #   for j in range(val_size):
                j=random.randint(0,val_size-1) 
                lossv= 10000 * sess.run(loss_op,  feed_dict={X:val_X[j], Y:val_Y[j]})
                val_sum=lossv
                loss = val_sum
                #Calculated running average of val_loss
                if stepp == 5:  # loss start from very large val at step=0, so start from 10th step
                    val_loss = loss
                elif stepp >5:
                    val_loss = alpha * val_loss + (1-alpha) * loss
                    #write up
                    fout_log.write("Step " + str(step) + ", Validation running average= " + \
                          "{:.4f}".format(val_loss ** 0.5) + ",Validation Loss= "+\
                          "{:.4f}".format(loss**0.5) + "\n")
                    print(" Step " + str(step) + ", Validation running average= " + \
                          "{:.4f}".format(val_loss ** 0.5) + ",Validatiion Loss= "+\
                          "{:.4f}".format(loss**0.5) + "\n")
                    if stepp > 10 and val_loss < val_best_loss:
                        val_best_loss = val_loss
                        save_path = saver.save(sess, "./model.ckpt")
                        print('found new best validation loss:', val_loss)
                        print("Model saved in path: %s" % save_path)
                        count = 0
                    if (epoch > 5) and (val_loss > val_best_loss):
                        count =  count + 1
                        if (count > 10):
                            print("Iteration "+str(it)+"Training DONE!  \n")
                            break





