import os
import random
import pickle
import threading
import queue
import numpy as np
import pandas as pd
import networkx as nx
import torch
from typing_extensions import Literal
from tqdm import tqdm
from .graph_utils import GraphUtils


class ModelTrainUtils:
    @staticmethod
    def get_data_loader(dataseq:list,batch_size:int):
        """
        Input:
            data_seq: List of data
                data
                    x: [N,1]
                    t: [N,1]
                    src: src id
                    tar: tar id
                    n_mask: [N,]
                    tar_label: label (1.0 or 0.0)
                    edge_index: [2,E]
                    adj_mask: [N,N]
                    label: [N,1], seq_step 마다 다름, 마지막 seq_step 값 참조 필요
            batch_size: batch size of edge_events
        Output:
            data_loader: batch_data_seq
                batch_data
                    x: [B,N,1]
                    t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N,], neighbor mask of target nodes
                    tar_label: [B,1]
                    edge_index: [2,E]
                    adj_mask: [N,N]
                    label: [N,1]
        """
        data_loader=[]
        edge_index=dataseq[-1]['edge_index'] # [2,E]
        adj_mask=dataseq[-1]['adj_mask'] # [N,N]
        label=dataseq[-1]['label'] # [N,1]
        for i in range(0,len(dataseq),batch_size):
            batch_dataseq=dataseq[i:i+batch_size]

            batch_x=torch.stack([e['x'] for e in batch_dataseq],dim=0) # [B,N,1]
            batch_t=torch.stack([e['t'] for e in batch_dataseq],dim=0) # [B,N,1]
            batch_src=torch.tensor([e['src'] for e in batch_dataseq],dtype=torch.int32).unsqueeze(1) # [B,1]
            batch_tar=torch.tensor([e['tar'] for e in batch_dataseq],dtype=torch.int32).unsqueeze(1) # [B,1]
            batch_n_mask=torch.stack([e['n_mask'] for e in batch_dataseq],dim=0) # [B,N,]
            batch_tar_label=torch.tensor([e['tar_label'] for e in batch_dataseq],dtype=torch.float32).unsqueeze(1) # [B,1]

            data_loader.append({
                'x':batch_x, # [B,N,1]
                't':batch_t, # [B,N,1]
                'src':batch_src, # [B,1]
                'tar':batch_tar, # [B,1]
                'n_mask':batch_n_mask, # [B,N,]
                'tar_label':batch_tar_label, # [B,1]
                'edge_index':edge_index, # [2,E]
                'adj_mask':adj_mask, # [N,N]
                'label':label # [N,1]
            })
        return data_loader

    @staticmethod
    def chunk_loader_worker(chunk_paths:str,buffer_queue:queue.Queue,max_buffer:int=2):
        """
        chunk_paths: chunk 파일 리스트
        """
        for path in chunk_paths:
            with open(path,"rb") as f:
                data=pickle.load(f)
            buffer_queue.put(data) # 버퍼가 꽉 차면 자동 대기
        buffer_queue.put(None) # 종료 신호

class EarlyStopping:
    def __init__(self,patience=1):
        self.patience=patience
        self.patience_count=0
        self.prev_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,val_loss:float,model:torch.nn.Module):
        if self.prev_loss==np.inf:
            self.prev_loss=val_loss
            self.best_state={k: v.clone() for k,v in model.state_dict().items()}
            return None
        else:
            if not np.isfinite(val_loss):
                print(f"Loss is NaN or Inf!")
                self.early_stop=True
                model.load_state_dict(self.best_state)
                return model
            
            if self.prev_loss<=val_loss:
                self.patience_count+=1
                if self.patience<self.patience_count:
                    print(f"Loss increases during {self.patience_count} patience!")
                    self.early_stop=True
                    model.load_state_dict(self.best_state)
                    return model
            else:
                self.patience_count=0
                self.prev_loss=val_loss
                self.best_state={k: v.clone() for k,v in model.state_dict().items()}
                return None