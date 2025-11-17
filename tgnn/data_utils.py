import os
import random
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
    def save_dataset_list_to_dataseq_list(dataset_list:list,mode:Literal['train','val','test'],num_nodes:int,chunk_size:int,dir_type:Literal['train','val','test']):
        dataseq_list=[]
        for dataset in dataset_list:
            dataseq_list+=[dataseq for dataseq in dataset]
        
        file_path=os.path.join(DataUtils.tgnn_path,dir_type)
        exist_chunk_files=[f for f in os.listdir(file_path) if f.startswith(f"{mode}_{num_nodes}_chunk_{chunk_size}_")]
        idx_offset=len(exist_chunk_files)

        chunk_list=[dataseq_list[i:i+chunk_size] for i in range(0,len(dataseq_list),chunk_size)]
        for idx,chunk in tqdm(enumerate(chunk_list),total=len(chunk_list),desc=f"Saving {mode}_{num_nodes}_chunk_{chunk_size}..."):
            DataUtils.save_to_pickle(data=chunk,file_name=f"{mode}_{num_nodes}_chunk_{chunk_size}_{idx+idx_offset}",path='tgnn',dir_type=dir_type)
        print(f"Save {mode}_{num_nodes}_chunk_{chunk_size}!")

    @staticmethod
    def save_dataset_list_dict_to_dataseq_list(dataset_list_dict:dict,random_src:int,mode:Literal['train','val','test'],num_nodes:int,chunk_size:int,dir_type:Literal['train','val','test']):
        if random_src:
            dataseq_list=[]
            source_info_dict={}
            for graph_type,dataset_list in dataset_list_dict.items():
                source_id_list=[]
                for dataset in dataset_list:
                    random_src_id=random.randrange(num_nodes)
                    source_id_list.append(random_src_id)
                    dataseq_list.append(dataset[random_src_id])
                source_info_dict[graph_type]=source_id_list
            
            # save source_info_dict
            DataUtils.save_to_pickle(data=source_info_dict,file_name=f"source_info_dict",path='tgnn',dir_type=dir_type)

            # save dataseq_list
            file_path=os.path.join(DataUtils.tgnn_path,dir_type)
            exist_chunk_files=[f for f in os.listdir(file_path) if f.startswith(f"{mode}_{num_nodes}_chunk_{chunk_size}_")]
            idx_offset=len(exist_chunk_files)

            chunk_list=[dataseq_list[i:i+chunk_size] for i in range(0,len(dataseq_list),chunk_size)]
            for idx,chunk in tqdm(enumerate(chunk_list),total=len(chunk_list),desc=f"Saving {mode}_{num_nodes}_chunk_{chunk_size}..."):
                DataUtils.save_to_pickle(data=chunk,file_name=f"{mode}_{num_nodes}_chunk_{chunk_size}_{idx+idx_offset}",path='tgnn',dir_type=dir_type)
            print(f"Save {mode}_{num_nodes}_chunk_{chunk_size}!")
        else:
            all_dataset_list=[]
            for _,dataset_list in dataset_list_dict.items():
                all_dataset_list+=dataset_list
            DataUtils.save_dataset_list_to_dataseq_list(dataset_list=all_dataset_list,mode=mode,num_nodes=num_nodes,chunk_size=chunk_size,dir_type=dir_type)

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