import os
import pickle
import torch
from tqdm import tqdm
from typing_extensions import Literal

class DataUtils:
    tgnn_path=os.path.join('..','data','tgnn')
    trne_path=os.path.join('..','data','trne')
    @staticmethod
    def save_to_pickle(data,file_name:str,path:Literal['tgnn','trne'],dir_type:Literal['graph','train','val','test']):
        file_name=file_name+".pkl"
        if path=='tgnn':
            file_path=os.path.join(DataUtils.tgnn_path,dir_type,file_name)
        else: # trne
            file_path=os.path.join(DataUtils.trne_path,dir_type,file_name)
        with open(file_path,'wb') as f:
            pickle.dump(data,f)
        print(f"Save {file_name}")

    @staticmethod
    def load_from_pickle(file_name:str,path:Literal['tgnn','trne'],dir_type:Literal['graph','train','val','test']):
        file_name=file_name+".pkl"
        if path=='tgnn':
            file_path=os.path.join(DataUtils.tgnn_path,dir_type,file_name)
        else: # trne
            file_path=os.path.join(DataUtils.trne_path,dir_type,file_name)
        with open(file_path,'rb') as f:
            data=pickle.load(f)
        print(f"Load {file_name}")
        return data

    @staticmethod
    def get_chunk_files_len(path:Literal['tgnn','trne'],chunk_size:int,dir_type:Literal['train','val','test']):
        if path=='tgnn':
            file_path=os.path.join(DataUtils.tgnn_path,dir_type)
        else: # trne
            file_path=os.path.join(DataUtils.trne_path,dir_type)

        chunk_files=[f for f in os.listdir(file_path) if f.startswith(f"train_20_chunk_{chunk_size}_")]
        return len(chunk_files)

    @staticmethod
    def save_dataset_list(dataset_list:list,file_name:str,chunk_size:int,dir_type:Literal['train','val','test']):
        chunk_list=[dataset_list[i:i+chunk_size] for i in range(0,len(dataset_list),chunk_size)]
        for idx,chunk in tqdm(enumerate(chunk_list),desc=f"Saving {file_name}_chunk..."):
            DataUtils.save_to_pickle(data=chunk,file_name=f"{file_name}_chunk_{chunk_size}_{idx}",path='tgnn',dir_type=dir_type)
        print(f"Save {file_name}! chunk_size: {chunk_size}")

    @staticmethod
    def save_dataset_list_each(dataset_list:list,file_name:str,graph_type:str,chunk_size:int,dir_type:Literal['train','val','test']):
        chunk_list=[dataset_list[i:i+chunk_size] for i in range(0,len(dataset_list),chunk_size)]
        for idx,chunk in tqdm(enumerate(chunk_list),desc=f"Saving {file_name}_{graph_type}_chunk..."):
            DataUtils.save_to_pickle(data=chunk,file_name=f"{file_name}_{graph_type}_chunk_{chunk_size}_{idx}",path='tgnn',dir_type=dir_type)
        print(f"Save {file_name}_{graph_type}! chunk_size: {chunk_size}")

    @staticmethod
    def save_dataset_list_dict(dataset_list_dict:dict,file_name:str,chunk_size:int,dir_type:Literal['train','val','test']):
        dataset_list=[]
        for _,value in dataset_list_dict.items():
            dataset_list+=value
        DataUtils.save_dataset_list(dataset_list=dataset_list,file_name=file_name,chunk_size=chunk_size,dir_type=dir_type)

    @staticmethod
    def save_dataset_list_dict_each(dataset_list_dict:dict,file_name:str,chunk_size:int,dir_type:Literal['train','val','test']):
        for key,value in tqdm(dataset_list_dict.items(),desc=f"Save {file_name} ..."):
            DataUtils.save_dataset_list_each(dataset_list=value,file_name=file_name,graph_type=key,chunk_size=chunk_size,dir_type=dir_type)

    @staticmethod
    def save_model_parameter(model,model_name:str):
        file_name=model_name+".pt"
        file_path=os.path.join(DataUtils.tgnn_path,"inference",file_name)
        torch.save(model.state_dict(),file_path)
        print(f"Save {model_name} model parameter")

    @staticmethod
    def load_model_parameter(model,model_name:str):
        file_name=model_name+".pt"
        file_path=os.path.join(DataUtils.tgnn_path,"inference",file_name)
        model.load_state_dict(torch.load(file_path))
        return model