import sys, os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))
sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )))
from utils import ensemble_dyn_training


if __name__ == "__main__":
    STATE_DIM = 2
    CONTROL_DIM = 1
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file_name = 'inv_pend_dyn_training_config.txt'
    ensemble_dyn_training(state_dim = STATE_DIM, control_dim = CONTROL_DIM, dir_path = dir_path, config_file_name = config_file_name)