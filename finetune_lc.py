import yaml
from finetune import main 
import pandas as pd
import os 
import time
from loguru import logger

def get_test_frac(config, train_points):
    df = pd.read_csv(config['dataset']['data_path'])
    length = len(df)
    test_frac = (length - train_points) / length
    return test_frac

lc_points = {
#     # 'FreeSolv': [10, 50, 100, 200, 500],
 #     'FreeSolv_2': [10, 50, 100, 200, 500],
#     # 'FreeSolv_5': [10, 50, 100, 200, 500],

#     # 'Lipo': [10, 50, 100, 200, 500],
#      'Lipo_2': [10, 50, 100, 200, 500],
#     # 'Lipo_5': [10, 50, 100, 200, 500],

# #     #'opv': [10, 50, 100, 200, 500],
#      'opv_2': [500], #10, 50, 100, 200, 
# # #    'opv_5': [10, 50, 100, 200, 500],

# # #    'qmug': [10, 50, 100, 200, 500, 1000],
#      'qmug_2': [10, 50, 100, 200, 500, 1000],
# # #    'qmug_5': [10, 50, 100, 200, 500, 1000],


# #    'photoswitch': [10, 20, 50, 100, 200],
#     'photoswitch_2': [10, 20, 50, 100, 200],
#    # 'photoswitch_5': [10, 20, 50, 100, 200],

#        'esol': [10, 20, 50, 100, 200, 500][::-1],
#     'esol_2': [10, 20, 50, 100, 200, 500][::-1],
     'esol_5': [10, 20, 50, 100, 200, 500][::-1],
}

def run_expt(task_name, train_points):
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
    config['task_name'] = task_name
    if config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["expt"]

    elif config['task_name'] == 'FreeSolv_2':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        config['dataset']['regression_bin_classes'] = 2
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_2"]
        config['model']['output_dim'] = 2

    elif config['task_name'] == 'FreeSolv_5':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_5"]
        config['model']['output_dim'] = 5

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["exp"]

    elif config["task_name"] == 'Lipo_2':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        config['dataset']['regression_bin_classes'] = 2
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_2"]  
        config['model']['output_dim'] = 2

    elif config["task_name"] == 'Lipo_5':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_5"]  
        config['model']['output_dim'] = 5

    elif config["task_name"] == 'opv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/opv/opv.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["PCE_ave(%)"]

    elif config["task_name"] == 'opv_2':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/opv/opv.csv'
        config['dataset']['regression_bin_classes'] = 2
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_2"]  
        config['model']['output_dim'] = 2

    elif config["task_name"] == 'opv_5':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/opv/opv.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_5"]  
        config['model']['output_dim'] = 5

    elif config["task_name"] == 'qmug':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qmug/qmug.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["DFT_HOMO_LUMO_GAP_mean_ev"]

    elif config["task_name"] == 'qmug_2':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/qmug/qmug.csv'
        config['dataset']['regression_bin_classes'] = 2
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_2"]  
        config['model']['output_dim'] = 2

    elif config["task_name"] == 'qmug_5':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/qmug/qmug.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_5"]  
        config['model']['output_dim'] = 5

    elif config["task_name"] == 'photoswitch':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/photoswitch/photoswitch.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["E isomer pi-pi* wavelength in nm"]

    elif config["task_name"] == 'photoswitch_2':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/photoswitch/photoswitch.csv'
        config['dataset']['regression_bin_classes'] = 2
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_2"]  
        config['model']['output_dim'] = 2

    elif config["task_name"] == 'photoswitch_5':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/photoswitch/photoswitch.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_5"]  
        config['model']['output_dim'] = 5

    elif config["task_name"] == 'esol':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["y"]

    elif config["task_name"] == 'esol_2':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        config['dataset']['regression_bin_classes'] = 2
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_2"]  
        config['model']['output_dim'] = 2

    elif config["task_name"] == 'esol_5':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        config['dataset']['regression_bin_classes'] = 5
        config['dataset']['splitting'] = 'stratified'
        target_list = ["target_5"]  
        config['model']['output_dim'] = 5

    else:
        raise ValueError('Undefined downstream task!')

    config['dataset']['test_size'] =get_test_frac(config, train_points)

    print(config)
    time_str = time.strftime("%Y%m%d-%H%M%S")

    results_list = []
    for target in target_list:
        config['dataset']['target'] = target
        result = main(config)
        results_list.append([target, result])

    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(
        'experiments/{}_{}_{}_{}finetune.csv'.format(time_str, config['fine_tune_from'], config['task_name'], train_points), 
        mode='a', index=False, header=False
    )

if __name__ == '__main__':
    for i in range(10):
        for task in lc_points.keys():
            for train_points in lc_points[task]:
                try:
                    run_expt(task, train_points)
                except Exception as e:
                    logger.exception(e)