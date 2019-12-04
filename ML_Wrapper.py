from utils.ArgParser import ArgParser
from models.DeepPipeline import DenseRnn
from models.MLPipeline import Classical_ML_benchmark
import argparse
def main(args):

    print("[START] Starting ML Pipeline ")
    ml_pipeline = Classical_ML_benchmark(csv_file_location=args.input_data, verbose=True,
                                         learning_rate=args.params.learning_rate,
                                         max_depth=args.params.max_depth,
                                         max_featues =args.params.max_featues,
                                         n_estimators_GB =args.params.n_estimators_GB,
                                         n_estimators_ET_RF = args.params.n_estimators_ET_RF,
                                         n_estimators_ADA = args.params.n_estimators_ADA
                                        )

    print("[STATUS] Fitting preset classifiers")
    ml_pipeline.fit_all_classifiers()

    ml_pipeline.report()

    print("[START] Starting DL Pipeline ")
    dense_pipeline = DenseRnn(csv_file_location=args.input_data, 
                              oht_epochs = args.params.oht_epochs, 
                              n_neurons=args.params.n_OHT_neurons,
                              z_initial_learning= args.params.z_initial_learning,
                              rnn_batchsize=args.params.rnn_batchsize,
                              rnn_dropout=args.params.dropout,
                              rnn_epochs=args.params.rnn_epochs,
                             )
    
    # initial staged training steps
    dense_pipeline.run(checkpoint_name='DNN.ckpt')
    # stage 2 RNN
    dense_pipeline.run(warm_start_checkpoint='DNN.ckpt', checkpoint_name='RNN.ckpt', pipeline='RNN')
    
    # tensorboard can be used to visualize the tf_log directory, accuracy and logloss are logged.
    
    # evaluate the two predictors, 
    # after hyperparameter tuning, another stage can be added to combine the logits,
    dense_pipeline.evaluate('RNN.ckpt', pipeline='DNN')
    dense_pipeline.evaluate('RNN.ckpt', pipeline='RNN')
    

    print("[STATUS] Finished ")
    
if __name__ == "__main__":
    
    argp = ArgParser()

    print("[START] Initating ML Wrapper")
    main(argp.args)
    
    
    
