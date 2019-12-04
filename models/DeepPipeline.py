from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from utils.Generator import Generator

class DenseRnn():
    def __init__(self,
                 csv_file_location, 
                 logging_root_directory = 'tf_log',
                 verbose = True, 
                 oht_epochs = 50,
                 oht_batchsize=100,
                 rnn_epochs = 3, 
                 rnn_batchsize = 32, 
                 n_outputs = 6,
                 z_initial_learning = 0.0002 ,
                 OHT_dropout = 0.3 ,
                 n_OHT_inputs = 294, 
                 z_decay_steps = 200,
                 n_OHT_neurons = 50, 
                 dropout = 0.5,
                 z_decay_rate = 0.9,
                 z_momentum = 0.999,
                 z_scale = 0.0007,
                 depth = 300,
                 rnn_dropout = 0.5, 
                 rnn_output_size = 30,
                 n_steps = 150,
                 n_inputs = 295,
                 n_neurons = 95,
                 n_neurons_RNN = 32,
                 initial_learning = 0.0008,
                 decay_steps = 1000,
                 decay_rate= 0.9,
                 momentum = 0.999,
                 scale = 0.00007 , 
                 initializer = tf.contrib.layers.xavier_initializer(),
                 regularizer = 'l1'
                ):
        
        self.data = Generator(csv_file_location)

        self.oht_epochs = oht_epochs
        self.oht_batchsize = oht_batchsize
        self.regularizer = regularizer
        self.dropout = dropout
        
        self.rnn_epochs = rnn_epochs
        self.rnn_batchsize = rnn_batchsize
        
        self.rnn_dropout = rnn_dropout
        self.rnn_output_size=rnn_output_size
        
        self.n_outputs = n_outputs
        self.z_initial_learning = z_initial_learning
        self.n_OHT_neurons = n_OHT_neurons
        self.n_OHT_inputs = n_OHT_inputs
        self.OHT_dropout = OHT_dropout
        self.z_decay_steps = z_decay_steps
        self.z_decay_rate = z_decay_rate
        self.z_momentum = z_momentum
        self.z_scale = z_scale
        self.depth = depth
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_neurons_RNN = n_neurons_RNN
        self.initial_learning = initial_learning
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.scale = scale
        
        self.verbose = verbose
        
        
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = logging_root_directory
        self.logdir = "{}/run-{}".format(root_logdir, now)
        # logging setup
        
        self.initializer = initializer
        self.regularizer = tf.contrib.layers.l1_regularizer(scale=self.scale)

        
    def _build(self):
               
        if self.verbose:
            print('[STATUS] resetting and building the TF graph')
        # build into the default graph, Change this for production code
        tf.reset_default_graph()
        
        # define input Tensor placeholders
        self.y = tf.placeholder(tf.int32,[None])
        self.X = tf.placeholder(tf.float32, [None, self.n_inputs])

        
        # placeholder for training status for dropout and batch normalization
        self.training = tf.placeholder_with_default(False,shape=(),name='training')

        
        with tf.name_scope("DENSE"):
        
            
            # set 3 static dense layers with dropout connections, verbose but explicit
            # named for Tensorboard visualization
            
            self.Dense1 = tf.layers.dense(self.X, (self.n_OHT_neurons), activation=tf.nn.elu,  
                                kernel_initializer=self.initializer,
                                kernel_regularizer=self.regularizer,
                               name="Dense1")
            
            # Dropout layer
            self.Dense_drop = tf.layers.dropout(self.Dense1, training = self.training, rate = self.OHT_dropout, name="Drop1")
    
    
            # setting a BN layer between the Dense layers
            self.Bn_norm = tf.layers.batch_normalization(self.Dense_drop,training=self.training, momentum = self.momentum)
                             
            self.Bn_norm_activation = tf.nn.elu(self.Bn_norm)
            
            self.Dense2 = tf.layers.dense(self.Bn_norm_activation, (self.n_OHT_neurons), activation=tf.nn.elu,  
                                kernel_initializer=self.initializer,
                                kernel_regularizer=self.regularizer,
                               name="Dense2")

            self.Dense_drop2 = tf.layers.dropout(self.Dense2, training = self.training, rate = self.OHT_dropout, name="Drop2")
    
    
            self.Dense3 = tf.layers.dense(self.Dense_drop2, (self.n_OHT_neurons), activation=tf.nn.elu,  
                                kernel_initializer=self.initializer,
                                kernel_regularizer=self.regularizer,
                               name="Dense3")

        
            # logits without activation
            self.Dense_logits = tf.layers.dense(self.Dense3, self.n_outputs, name="outputs_logits")
        
            # stop layer for accessing the logits downstream
            self.OHT_stop = tf.stop_gradient(self.Dense_logits,name='stop')

            # outputs as a potential second stage input
            self.OHT_output = tf.layers.dense(self.OHT_stop, self.n_outputs, name="connection") # feed the logits into the RNN
        
        
        #  RNN Graph
        with tf.name_scope("RNN"):
    
            # increase dimensions of X to fit an RNN,  expects 3 dimentions
        
            X_ext = tf.expand_dims(self.X,1)
    
            # setup of the GRU cell array
            
            # forward cells
            cells_fwd = [
         
                 tf.contrib.rnn.DropoutWrapper(
                 tf.contrib.rnn.GRUCell(num_units=(self.n_neurons_RNN-0),
                                        activation=tf.nn.elu, 
                                   kernel_initializer = tf.contrib.layers.xavier_initializer())
                  , input_keep_prob = self.rnn_dropout ),
        
                tf.contrib.rnn.OutputProjectionWrapper(
        
                tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(num_units=(self.n_neurons_RNN-0),
                                       activation=tf.nn.elu, 
                                   kernel_initializer = tf.contrib.layers.xavier_initializer())
                , input_keep_prob = self.rnn_dropout )
                    ,output_size = self.rnn_output_size
                )
    
            ]
        
            # backward cells in an outputprojection wrapper
            cells_bwd = [

            tf.contrib.rnn.OutputProjectionWrapper(
             tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.GRUCell(num_units=self.n_neurons_RNN,
                               activation=tf.nn.elu ,
                               kernel_initializer = tf.contrib.layers.xavier_initializer())
                , input_keep_prob = self.rnn_dropout )
            
            ,output_size = self.rnn_output_size
            ) ]
    
            # 1.5 layer GRU should catch reasonable forward and backward linked features
            # needs a few layers for processing
    
            
            self.multi_layer_cell_fw = tf.contrib.rnn.MultiRNNCell(cells_fwd, state_is_tuple=False)
            self.multi_layer_cell_bw = tf.contrib.rnn.MultiRNNCell(cells_bwd, state_is_tuple=False)
    
            #
            # both outputs ( states over times) and states ( final neuron states ) contain information, 
            # so using both of those may be benefitial. 
            self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.multi_layer_cell_fw, 
                                                                        cell_bw=self.multi_layer_cell_bw,
                                                                        inputs=X_ext, 
                                                                        dtype=tf.float32, 
                                                                        #sequence_length=self.seq_length  # static sequence length
                                                                       )  
        
        with tf.name_scope("RNN_Combine"):

            # combine fwd and reverse
            self.combinedstates = tf.concat([self.states[0],self.states[1]], 1)
        
        
        
        with tf.name_scope("RNN_Supplementary"):
            
            # two dense layers may be enough, too many or not enough depending on the data,  determine empirically
            
            self.supplementary1= tf.layers.dense(self.combinedstates, (self.n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=self.initializer,
                            kernel_regularizer=self.regularizer,
                           name="supplementary_1") # add a dense layer
    
    
    
            self.sup_drop = tf.layers.dropout(self.supplementary1, training = self.training, rate=self.dropout)
    
    
    
    
            self.supplementary2 = tf.layers.dense(self.sup_drop, (self.n_neurons), activation=tf.nn.elu,  
                            kernel_initializer=self.initializer,
                            kernel_regularizer=self.regularizer,
                           name="supplementary_2") # add a dense layer
        
        
        
            # batch normalization at logits level can be benefitial
            self.logitsRNN = tf.layers.dense(self.supplementary2, self.n_outputs, name="outputs_logitsRNN")
            self.RNN_stop = tf.stop_gradient(self.logitsRNN,name='stop_RNN')
        
        
        with tf.name_scope("Combinatorial_NN"):
            # Space to combine the RNN and DNN outputs,
            self.RNN_DNN = tf.concat([self.RNN_stop,self.OHT_output], 1)
            
            # add an additional layer and the corresponding training function
            # this is omitted here, since neither RNN nor DNN perform, and the initial trainin op should be changed 
            # from softmax to sigmoid,to get more information out of the processes.
            
        
        with tf.name_scope("evaluation_Dense"):
            
            # mutually exclusive categories 
            # cross_entropy = tf.reduce_sum(- y * tf.log(logits), 1)  # multiclass form of crossentropy
            # the binary version would then be the binary crossentropy p * -tf.log(q) + (1 - p) * -tf.log(1 - q)
            
            # 
            
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.Dense_logits)
            
            # using multiclas probabilities by sigmoid estimation is more useful if the output of this layer is used for
            # input in a second stage

            self.loss = tf.reduce_mean(self.cross_entropy)

            self.global_step = tf.Variable(0, trainable=False)

            self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss_aftereg = tf.add_n([self.loss]+ self.reg_loss)

            self.learning_rate = tf.train.exponential_decay(self.z_initial_learning,self.global_step,
                                                               self.z_decay_steps,self.z_decay_rate)
            
            self.optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate)
            
            # not implementing a switch here, goes beyond the scope for the example, as none of them will produce better results
            #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate) # recommended for RNN, but in my experience not the best
            #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
            #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            
            
            
            if self.z_scale == 0:
                self.training_op_DNN = self.optimizer.minimize(self.loss)
            else:
                self.training_op_DNN = self.optimizer.minimize(self.loss_aftereg)

            self.prediction = tf.argmax(self.Dense_logits,1)
            
            
        with tf.name_scope("evaluation_RNN"):
            
            # cross entropy on softmax for optimization
            self.r_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logitsRNN)
            
            # add the regularization to the loss
            self.r_loss = tf.reduce_mean(self.r_cross_entropy)
            self.r_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            
            self.r_loss_aftereg = tf.add_n([self.r_loss]+ self.r_reg_loss)
            
            #self.global_step = tf.Variable(0, trainable=False)
            
            #self.learning_rate = tf.train.exponential_decay(self.z_initial_learning,self.global_step,
            #                                                   self.z_decay_steps,self.z_decay_rate)
            self.r_optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate)
            self.training_op_RNN = self.r_optimizer.minimize(self.r_loss_aftereg)
            
            self.r_prediction = tf.argmax(self.logitsRNN,1)
            
        with tf.name_scope("eval"):
            #self.correct = tf.equal(tf.nn.in_top_k(self.Dense_logits,k=1), self.y )
            
            # is expected the top position
            self.correct = tf.nn.in_top_k(predictions=self.Dense_logits, targets=self.y, k=1)
            
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            
            self.correctRNN = tf.nn.in_top_k(predictions=self.logitsRNN, targets=self.y, k=1)
            self.accuracyRNN = tf.reduce_mean(tf.cast(self.correctRNN, tf.float32))
    
        with tf.name_scope("logging"):
    
            self.accuracy_sum = tf.summary.scalar('accuracy',self.accuracy) # summary of Dense Accuracy for logging
            self.loss_sum = tf.summary.scalar('loss',self.loss)
    
            self.accuracyRNN_sum = tf.summary.scalar('accuracyRNN',self.accuracyRNN) # summary for RNN Accuracy for logging
    
            self.summaries = tf.summary.merge_all()
    
            # writ to tensorboard
        
            # seperate records for train and test set to visualize overfitting
            self.train_writer = tf.summary.FileWriter(self.logdir + '/train', tf.get_default_graph())
            self.test_writer = tf.summary.FileWriter(self.logdir + '/test')
    
    
        with tf.name_scope("predict"):
            self.prediction = tf.argmax(self.Dense_logits,1)
            self.predictionRNN = tf.argmax(self.logitsRNN,1)
            
        
        # set initalizer and saver
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
    def run(self, warm_start_checkpoint = 'None' , checkpoint_name = 'trained.ckpt',earlystop_lmt = 7, verbose = False, val_data_size = 3000, pipeline = 'DNN'):
        
        
        # Build the Graph
        self._build()
        # link for tb
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        
        # set validation data
        X_val,y_val = self.data.getbatch(val_data_size, training=False)
        
        
        
        # choose training operation
        if pipeline == 'DNN':
            training_op = self.training_op_DNN
            batchsize = self.oht_batchsize
            n_epochs = self.oht_epochs
            prediction = self.prediction
            accuracy = self.accuracy
        if pipeline == 'RNN':
            training_op = self.training_op_RNN
            batchsize = self.rnn_batchsize
            n_epochs = self.rnn_epochs
            prediction = self.predictionRNN
            accuracy = self.accuracyRNN
            
        bestquess = 0
        earlystop = 0
        
        with tf.Session() as sess:
            
            
            
            # warm start optional
            if warm_start_checkpoint == 'None':
                self.init.run()
                
                if self.verbose:
                    print('[STATUS] Starting current session as sess in default graph')
                
                
            else:
                self.saver.restore(sess, warm_start_checkpoint)
                
                if self.verbose:
                    print('[STATUS] Restoring session %s ' %(warm_start_checkpoint))

            for epoch in range(n_epochs):
        
                for iteration in range(len(self.data.train) //batchsize):
            
            
                    X_batch,y_batch = self.data.getbatch(batchsize)
                
                    
                    sess.run(training_op, feed_dict={self.X: X_batch,self.y:y_batch, self.training : True })
        
                    if iteration % 1000 == 0:
                        X_batch,y_batch = self.data.getbatch(self.oht_batchsize * 4 ) # larger sample set for validation
                        predictions = prediction.eval(feed_dict={self.X: X_batch, self.y: y_batch})
                
                        train_accuracy = accuracy.eval(feed_dict={self.X: X_batch, self.y: y_batch })

                        val_acc = accuracy.eval(feed_dict={self.X: X_val, self.y: y_val })
                        val_pred = prediction.eval(feed_dict={self.X: X_val, self.y: y_val} )
                        
                
            
                        # LOGGING 
                        # collect on both validation and training data
                        step = iteration * epoch 
                        
                        summary, _ = sess.run([self.summaries,extra_update_ops], feed_dict={self.X: X_batch, self.y: y_batch} )
                        summary, _ = sess.run([self.summaries,extra_update_ops], feed_dict={self.X: X_val, self.y: y_val })
                
                        self.train_writer.add_summary(summary, step )
                        self.test_writer.add_summary(summary,step)
                
                
                        
                        cwl = self._cw_roc_auc(y_val,val_pred)
                        if self.verbose:
                            
                            print('[STATUS] Epoch %s : Current column wise roc_auc %s ' %(epoch,cwl) )
                            
                            
                        if cwl > bestquess:
                            bestquess = cwl
                            
                            self.saver.save(sess, '%s_unstable.ckpt' %checkpoint_name  )  
                            
                            earlystop = 0 
                        else:
                            earlystop +=1
                        
                        if earlystop > earlystop_lmt:
                            if verbose:
                                print("[STOP] stopped for your convenience")
                            break
            
            
            
            self.bestquess = bestquess
            
            self.saver.save(sess, checkpoint_name )  
            
            
    
            
    
                
    def _cw_roc_auc(self,y_true,y_pred, verbose=False):
        
        
        
        #y_true_enc = self.data.label_encoder.transform(y_true).reshape(-1, 1)
        #y_pred_enc = self.data.label_encoder.transform(y_pred).reshape(-1, 1)
        
        # transform the predictions and labels to one hot encoding
        
        #rint(accuracy_score(y_pred=y_pred,y_true=y_true))
        
        y_true = self.data.ohot_encoder.transform(y_true.reshape(-1, 1))
        
        try:
            # initial predictions can be outside label scope
            y_pred = self.data.ohot_encoder.transform(y_pred.reshape(-1, 1))
        except:
            if self.verbose:
                print("[WARNING] Predicitons outside labels scope, probably all zeros in one class")
        

        try:
            # ROC AUC is not defined for single classes, meaning if classes are missing from batch, no ROC auc is defined
            return np.mean([roc_auc_score(y_true= np.transpose(y_true)[count], y_score=col) 
                        for count,col in enumerate(np.transpose(y_pred)) ] )
        
        
            # needs to be more explicit
            
            #for count, row in enumerate(np.transpose(y_pred)):
            #    if sum(row) > 0 and sum(np.transpose(y_true) > )
        
        except:
            return 0
     
    
    # def chunker(seq, size):
    #    return (data[pos:pos + size] for pos in range(0, len(data), size) )
    
    # there is no test data, only train and val, so no need for prediction functions
    #def predict(self, checkpoint,  chunksize=10000):
    #    
    #    with tf.Session() as sess:
    #        self.saver.restore(sess, checkpoint)

    #        for group in self.chunker(data,chunksize):
    #            X_test y_test = getbatch(group, testing=True, has_y=False)
    #            results = self.prediction.eval(feed_dict={ self.X:X_test})
    #            reslist = [x for x in results.tolist()]
    #                
    #        return reslist
        
    def evaluate(self,checkpoint, pipeline='DNN' ):
        
        if pipeline == 'DNN':
            prediction = self.prediction
            accuracy = self.accuracy
        if pipeline == 'RNN':
            prediction = self.predictionRNN
            accuracy = self.accuracyRNN
        
        with tf.Session() as sess:
                self.saver.restore(sess, checkpoint)
                X_val,y_val = self.data.getbatch(len(self.data.val), training=False)
                
                val_acc = accuracy.eval(feed_dict={self.X: X_val, self.y: y_val })
                val_pred = prediction.eval(feed_dict={self.X: X_val, self.y: y_val} )
                
                cwl = self._cw_roc_auc(y_val,val_pred)
                
                print("%s predicts with an accuracy of : %f and a Columnwise ROC_AUC score of %f" %(checkpoint,val_acc,cwl) )
