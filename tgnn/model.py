import torch
import torch.nn as nn
from typing_extensions import Literal
from .graph_utils import GraphUtils
from .modules import TimeEncoder,MemoryUpdater,TimeProjection,GraphAttention,GraphSum

class TGAT(nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.attention=GraphAttention(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim

    def forward(self,data_loader,device):
        """
        Input:
            data_loader: List of batch_data
                batch_data
                    x: [B,N,1]
                    t: [B,N,1]
                    tar: [B,1]
                    n_mask: [B,N,]
                    edge_index: [2,E]
            device: GPU
        Output:
            step_logit: [seq_len,batch_size,1]
            last_logit: [N,1]
        """
        """
        compute step tR logit
        """
        step_logit_list=[]
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            x=batch['x'] # [B,N,1], float
            t=batch['t'] # [B,N,1], float
            tar=batch['tar'] # [B,1], long
            n_mask=batch['n_mask'] # [B,N,], neighbor node mask

            """
            embedding
            """
            batch_size,num_nodes,_=x.size()

            # target x vector
            batch_idx=torch.arange(batch_size,device=x.device) # [B,]
            tar=tar.squeeze(-1) # [B,]
            tar_x=x[batch_idx,tar,:] # [B,1]

            # target vector
            tar_hidden_ft=torch.zeros((batch_size,self.latent_dim),dtype=torch.float,device=x.device) # [B,latent_dim]
            tar_vec=torch.cat([tar_x,tar_hidden_ft],dim=-1) # [B,node_dim+latent_dim]
            tar_t=torch.zeros((batch_size,1),dtype=torch.float,device=x.device) # [B,1]
            encoded_tar_t=self.time_encoder(tar_t) # [B,latent_dim]
            tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [B,node_dim+latent_dim+latent_dim]

            # neighbor
            hidden_ft=torch.zeros((batch_size,num_nodes,self.latent_dim),dtype=torch.float,device=x.device) # [B,N,latent_dim]
            x=torch.cat([x,hidden_ft],dim=-1) # [B,N,node_dim+latent_dim]
            encoded_t=self.time_encoder(t) # [B,N,latent_dim]
            h=torch.cat([x,encoded_t],dim=-1) # [B,N,node_dim+latent_dim+latent_dim]

            # attention result
            z=self.attention(tar_vec=tar_h,tar_idx=tar.unsqueeze(-1),h=h,neighbor_mask=n_mask) # [B,latent_dim]
            logit=self.linear(z) # [B,1]
            step_logit_list.append(logit)
        """
        compute last tR logit
        """
        x=data_loader[-1]['x'][0] # [N,1]
        t=data_loader[-1]['t'][-1] # [N,1]
        edge_index=data_loader[-1]['edge_index'] # [2,E]
        adj_mask=GraphUtils.get_adj_mask(num_nodes=num_nodes,edge_index=edge_index) # [N,N]
        x=x.to(device)
        t=t.to(device)
        edge_index=edge_index.to(device)
        adj_mask=adj_mask.to(device)

        # target vector (all nodes)
        tar_hidden_ft=torch.zeros((num_nodes,self.latent_dim),dtype=torch.float,device=device) # [N,latent_dim]
        tar_vec=torch.cat([x,tar_hidden_ft],dim=-1) # [N,node_dim+latent_dim]
        tar_t=torch.zeros((num_nodes,1),dtype=torch.float,device=device) # [N,1]
        encoded_tar_t=self.time_encoder(tar_t) # [N,latent_dim]
        tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [N,node_dim+latent_dim+latent_dim]

        # neighbor
        hidden_ft=torch.zeros((num_nodes,self.latent_dim),dtype=torch.float,device=device) # [N,latent_dim]
        x=torch.cat([x,hidden_ft],dim=-1) # [N,node_dim+latent_dim]
        encoded_t=self.time_encoder(t) # [N,latent_dim]
        h=torch.cat([x,encoded_t],dim=-1) # [N,node_dim+latent_dim+latent_dim]

        # attention result
        z=self.attention(tar_vec=tar_h,h=h,neighbor_mask=adj_mask,step='last') # [N,latent_dim]
        last_logit=self.linear(z) # [N,1]

        output={}
        output['step_logit_list']=step_logit_list # List of [B,1]
        output['last_logit']=last_logit # [N,1]
        return output 

class TGN(nn.Module):
    def __init__(self,node_dim,latent_dim,emb:Literal['time','attn','sum']): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.memory_updater=MemoryUpdater(node_dim=node_dim,latent_dim=latent_dim)
        match emb:
            case 'time':
                self.embedding=TimeProjection(latent_dim=latent_dim)
            case 'attn':
                self.embedding=GraphAttention(node_dim=node_dim,latent_dim=latent_dim)
            case 'sum':
                self.embedding=GraphSum(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim
        self.emb=emb

    def forward(self,data_loader,device):
        """
        Input:
            data_loader: List of batch_data
                batch_data
                    x: [B,N,1]
                    t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N,]
                    edge_index: [2,E]
            device: GPU
        Output:
            step_logit: [seq_len,batch_size,1]
            last_logit: [N,1]
        """
        """
        compute step tR logit
        """
        step_logit_list=[]
        last_memory=None
        num_nodes=data_loader[0]['x'].size(1)
        memory=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device)
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            src=batch['src'] # [B,1], long
            tar=batch['tar'] # [B,1], long
            x=batch['x'] # [B,N,1], float
            t=batch['t'] # [B,N,1], float
            n_mask=batch['n_mask'] # [B,N,], neighbor node mask

            """
            1. update memory
            """
            delta_t=self.time_encoder(t) # [B,N,latent_dim]
            updated_memory=self.memory_updater(x=x,memory=memory,source=src,target=tar,delta_t=delta_t) # [N,latent_dim]
            last_memory=updated_memory

            """
            2. embedding
            """
            batch_size,_,_=x.size()
        
            # target x
            batch_idx=torch.arange(batch_size,device=x.device) # [B,]
            tar=tar.squeeze(-1) # [B,]
            tar_x=x[batch_idx,tar,:] # [B,1]

            # target vec
            tar_hidden_ft=updated_memory[tar] # [B,latent_dim]
            tar_vec=torch.cat([tar_x,tar_hidden_ft],dim=-1) # [B,node_dim+latent_dim]
            tar_t=torch.zeros((batch_size,1),dtype=torch.float,device=x.device) # [B,1]
            encoded_tar_t=self.time_encoder(tar_t) # [B,latent_dim]
            tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [B,node_dim+latent_dim+latent_dim]

            # neighbor
            hidden_ft=updated_memory.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,latent_dim]
            x=torch.cat([x,hidden_ft],dim=-1) # [B,N,node_dim+latent_dim]
            encoded_t=self.time_encoder(t) # [B,N,latent_dim]
            h=torch.cat([x,encoded_t],dim=-1) # [B,N,node_dim+latent_dim+latent_dim]

            # attention result
            match self.emb:
                case 'time':
                    tar_memory=updated_memory[tar] # [B,latent_dim]
                    delta_t=t[batch_idx,tar,:] # [B,1]
                    z=self.embedding(target_memory=tar_memory,delta_t=delta_t) # [B,latent_dim]
                case 'attn'|'sum':
                    z=self.embedding(tar_vec=tar_h,tar_idx=tar.unsqueeze(-1),h=h,neighbor_mask=n_mask,step='step') # [B,latent_dim]

            logit=self.linear(z) # [B,1]
            step_logit_list.append(logit)
            memory=updated_memory # set next memory
        """
        compute last tR logit
        """
        x=data_loader[-1]['x'][0] # [N,1]
        t=data_loader[-1]['t'][-1] # [N,1]
        edge_index=data_loader[-1]['edge_index'] # [2,E]
        adj_mask=GraphUtils.get_adj_mask(num_nodes=num_nodes,edge_index=edge_index) # [N,N]
        x=x.to(device)
        t=t.to(device)
        edge_index=edge_index.to(device)
        adj_mask=adj_mask.to(device)

        # target vector (all nodes)
        tar_hidden_ft=last_memory # [N,latent_dim]
        tar_vec=torch.cat([x,tar_hidden_ft],dim=-1) # [N,node_dim+latent_dim]
        tar_t=torch.zeros((num_nodes,1),dtype=torch.float,device=device) # [N,1]
        encoded_tar_t=self.time_encoder(tar_t) # [N,latent_dim]
        tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [N,node_dim+latent_dim+latent_dim]

        # neighbor
        hidden_ft=last_memory # [N,latent_dim]
        x=torch.cat([x,hidden_ft],dim=-1) # [N,node_dim+latent_dim]
        encoded_t=self.time_encoder(t) # [N,latent_dim]
        h=torch.cat([x,encoded_t],dim=-1) # [N,node_dim+latent_dim+latent_dim]

        # attention result
        match self.emb:
            case 'time':
                z=self.embedding(target_memory=last_memory,delta_t=t) # [N,latent_dim]
            case 'attn'|'sum':
                z=self.embedding(tar_vec=tar_h,h=h,neighbor_mask=adj_mask,step='last') # [N,latent_dim]
        last_logit=self.linear(z) # [N,1]

        output={}
        output['step_logit_list']=step_logit_list # List of [B,1]
        output['last_logit']=last_logit # [N,1]
        return output 