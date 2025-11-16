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
    def save_to_dataset_list(dataset_list:list,mode:Literal['train'],num_nodes:int,chunk_size:int,dir_type:Literal['train','val','test']):
        """
        dataset 단위 리스트로 저장
        """
        file_path=os.path.join(DataUtils.tgnn_path,dir_type)
        exist_chunk_files=[f for f in os.listdir(file_path) if f.startswith(f"{mode}_{num_nodes}_chunk_dataset_{chunk_size}_")]
        idx_offset=len(exist_chunk_files)
        chunk_list=[dataset_list[i:i+chunk_size] for i in range(0,len(dataset_list),chunk_size)]
        for idx,chunk in tqdm(enumerate(chunk_list),desc=f"Saving {mode}_{num_nodes}_chunk_dataset_{chunk_size}..."):
            DataUtils.save_to_pickle(data=chunk,file_name=f"{mode}_{num_nodes}_chunk_dataset_{chunk_size}_{idx+idx_offset}",path='tgnn',dir_type=dir_type)
        print(f"Save {mode}_{num_nodes}_chunk_dataset_{chunk_size}!")

    @staticmethod
    def save_to_dataseq_list(dataset_list:list,mode:Literal['val','test'],num_nodes:int,chunk_size:int,dir_type:Literal['train','val','test']):
        """
        dataseq 단위 리스트로 저장
        """
        dataseq_list=[]
        for dataset in dataset_list:
            dataseq_list+=[data for dataseq in dataset for data in dataseq]
        
        file_path=os.path.join(DataUtils.tgnn_path,dir_type)
        exist_chunk_files=[f for f in os.listdir(file_path) if f.startswith(f"{mode}_{num_nodes}_chunk_dataseq_{chunk_size}_")]
        idx_offset=len(exist_chunk_files)

        chunk_list=[dataseq_list[i:i+chunk_size] for i in range(0,len(dataseq_list),chunk_size)]
        for idx,chunk in tqdm(enumerate(chunk_list),desc=f"Saving {mode}_{num_nodes}_chunk_dataseq_{chunk_size}..."):
            DataUtils.save_to_pickle(data=chunk,file_name=f"{mode}_{num_nodes}_chunk_dataseq_{chunk_size}_{idx+idx_offset}",path='tgnn',dir_type=dir_type)
        print(f"Save {mode}_{num_nodes}_chunk_dataseq_{chunk_size}!")

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