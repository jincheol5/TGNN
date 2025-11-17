import os
import random
import numpy as np
import argparse
import wandb
import torch
from tqdm import tqdm
from tgnn import DataUtils,GraphUtils

def app_data(config: dict):
    match config['app_num']:
        case 1:
            """
            App 1.
            Convert and save graph_list_dict into dataseq_list and store them in chunks.
            All type graph
            train_20
            val_20
            test_20
            """
            graph_list_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",path="trne",dir_type="graph")
            dataset_list_dict=GraphUtils.convert_to_dataset_list_dict(graph_list_dict=graph_list_dict)
            DataUtils.save_dataset_list_dict_to_dataseq_list(
                dataset_list_dict=dataset_list_dict,
                random_src=config['random_src'],
                mode=config['mode'],
                num_nodes=config['num_nodes'],
                chunk_size=config['chunk_size'],
                dir_type=config['mode']
            )

        case 2:
            """
            App 2.
            Convert and save graph_list_dict into dataseq_list and store them in chunks.
            Only one_type graph
            test_50
            test_100
            test_500
            test_1000
            """
            graph_list_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",path="trne",dir_type="graph")
            graph_list=graph_list_dict[config['graph_type']]
            dataset_list=GraphUtils.convert_to_dataset_list(graph_list=graph_list,graph_type=config['graph_type'])

            DataUtils.save_dataset_list_to_dataseq_list(
                dataset_list=dataset_list,
                mode=config['mode'],
                num_nodes=config['num_nodes'],
                chunk_size=config['chunk_size'],
                dir_type=config['mode']
            )


if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--random_src",type=int,default=0)
    parser.add_argument("--mode",type=str,default='test')
    parser.add_argument("--num_nodes",type=int,default=20)
    parser.add_argument("--graph_type",type=str,default='default')
    parser.add_argument("--chunk_size",type=int,default=1)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        'random_src':args.random_src,
        'mode':args.mode,
        'num_nodes':args.num_nodes,
        'graph_type':args.graph_type,
        'chunk_size':args.chunk_size
    }
    app_data(config=config)