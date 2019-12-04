import argparse
import pandas as pd
class ArgParser():
    def __init__(self):
        
        
        
        parser = argparse.ArgumentParser(description='Comparing Machine learning algorythms')
        
        
        parser.add_argument('--infile', type=str, dest='input_data',default='sample.csv',
                    help='input file used to train and evaluate predictors, defaults to sample.csv', )
        
        
        parser.add_argument('--paramfile', type=str, dest='param_data',default='param.csv',
                    help='parameter file to set Hyperparameters in classifiers,  one parameter per line, comma seperated ', )
        
        self.args = parser.parse_args()
        self.args.params = self.parse_params(self.args.param_data)
        
    def parse_params(self, param_data ):
        
        params = pd.read_csv(param_data, header=None, index_col=0).to_dict(orient='index')
        
        parsed_pars = Parameter_storage()
        parsed_pars.learning_rate = float(params['learning_rate'][1])
        parsed_pars.max_depth = int(params['max_depth'][1])
        parsed_pars.max_featues = str(params['max_featues'][1])
        parsed_pars.n_estimators_GB = int(params['n_estimators_GB'][1])
        parsed_pars.n_estimators_ET_RF = int(params['n_estimators_ET_RF'][1])
        parsed_pars.n_estimators_ADA = int(params['n_estimators_ADA'][1])
        parsed_pars.oht_epochs= int(params['oht_epochs'][1])
        parsed_pars.oht_batchsize= int(params['oht_batchsize'][1])
        parsed_pars.rnn_epochs= int(params['rnn_epochs'][1])
        parsed_pars.rnn_batchsize= int(params['rnn_batchsize'][1])
        parsed_pars.z_initial_learning= float(params['z_initial_learning'][1])
        parsed_pars.z_decay_steps= float(params['z_decay_steps'][1])
        parsed_pars.n_OHT_neurons= int(params['n_OHT_neurons'][1])
        parsed_pars.dropout= float(params['dropout'][1])
        parsed_pars.momentum= float(params['momentum'][1])
        parsed_pars.scale= float(params['scale'][1])
        
        return parsed_pars


class Parameter_storage():
    def __init__(self):
        self.name = 'ParamStore'
