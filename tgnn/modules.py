import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Literal
from torch_scatter import scatter_mean

class TimeEncoder(nn.Module):
    def __init__(self,time_dim:int,parameter_requires_grad:bool=True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder,self).__init__()
        self.time_dim=time_dim
        self.w=nn.Linear(1,time_dim)
        self.w.weight=nn.Parameter((torch.from_numpy(1/10**np.linspace(0,9,time_dim,dtype=np.float32))).reshape(time_dim,-1))
        self.w.bias=nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad=False
            self.w.bias.requires_grad=False

    def forward(self,timestamps:torch.Tensor):
        """
        compute time encodings of time in timestamps
        Input:
            timestamps: [batch_size,N,1]
        Output:
            updated_timestamps: [batch_size,N,time_dim]
        """
        output=torch.cos(self.w(timestamps)) # [batch_size,N,time_dim]
        return output

class MemoryUpdater(nn.Module):
    def __init__(self,node_dim,latent_dim):
        super().__init__()
        self.src_mlp=nn.Sequential(
            nn.Linear(in_features=latent_dim+latent_dim+latent_dim+node_dim,out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim)
        )
        self.tar_mlp=nn.Sequential(
            nn.Linear(in_features=latent_dim+latent_dim+latent_dim+node_dim,out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=latent_dim)
        )
        self.gru=nn.GRUCell(input_size=latent_dim,hidden_size=latent_dim)

    def message_aggregate(self,source:torch.Tensor,target:torch.Tensor,source_msg:torch.Tensor,target_msg:torch.Tensor):
        """
        Input:
            source: [batch_size,1]
            target: [batch_size,1]
            source_msg: [batch_size,latent_dim]
            target_msg: [batch_size,latent_dim]
        Output:
            aggregated_msg: [unique_node_size,latent_dim]
        """
        src_tar_nodes=torch.cat([source,target],dim=0).squeeze(-1) # [2*batch_size,]
        src_tar_msg=torch.cat([source_msg,target_msg],dim=0)  # [2*batch_size,latent_dim]
        unique_nodes,inverse_indices=torch.unique(src_tar_nodes,return_inverse=True) # [unique_node_size,],[2*batch_size]
        aggregated_msg=scatter_mean(src=src_tar_msg,index=inverse_indices,dim=0)  # [unique_node_size,latent_dim]
        return unique_nodes,aggregated_msg # [unique_node_size,],[unique_node_size,latent_dim]

    def forward(self,x,memory,source:torch.Tensor,target:torch.Tensor,delta_t:torch.Tensor):
        """
        Input:
            x: [batch_size,N,1]
            memory: [N,latent_dim]
            source: [batch_size,1]
            target: [batch_size,1]
            delta_t: [batch_size,N,latent_dim]
        Output:
            updated_memory
        """
        batch_size,_,_=x.size()
        
        source_batch_indices=torch.arange(batch_size,device=x.device) # [batch_size,]
        source=source.squeeze(-1) # [batch_size,]
        source_memory=memory[source] # [batch_size,latent_dim]
        source_x=x[source_batch_indices,source,:] # [batch_size,1]
        source_delta_t=delta_t[source_batch_indices,source,:] # [batch_size,latent_dim]
        
        target_batch_indices=torch.arange(batch_size,device=x.device)
        target=target.squeeze(-1) # [batch_size,]
        target_memory=memory[target] # [batch_size,latent_dim]
        target_x=x[target_batch_indices,target,:] # [batch_size,1]
        target_delta_t=delta_t[target_batch_indices,target,:] # [batch_size,latent_dim]

        source_msg_input=torch.cat([source_memory,target_memory,source_delta_t,source_x],dim=-1) # [batch_size,latent_dim+latent_dim+latent_dim+node_dim]
        source_msg=self.src_mlp(source_msg_input) # [batch_size,latent_dim]

        target_msg_input=torch.cat([target_memory,source_memory,target_delta_t,target_x],dim=-1) # [batch_size,latent_dim+latent_dim+latent_dim+1]
        target_msg=self.tar_mlp(target_msg_input) # [batch_size,latent_dim]

        unique_nodes,aggregated_msg=self.message_aggregate(
            source=source.unsqueeze(-1),
            target=target.unsqueeze(-1),
            source_msg=source_msg,
            target_msg=target_msg) # [unique_node_size,],[unique_node_size,latent_dim]
        
        pre_memory=memory[unique_nodes] # [unique_node_size,latent_dim]
        new_memory=self.gru(aggregated_msg,pre_memory) # [unique_node_size,latent_dim]
        memory[unique_nodes]=new_memory # [N,latent_dim]

        return memory # [N,latent_dim]

"""
Embedding Modules
1. TimeProjection
2. GraphAttention
3. GraphSum
"""
class TimeProjection(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.w=nn.Linear(in_features=1,out_features=latent_dim)

    def forward(self,target_memory,delta_t):
        """
        Input:
            target_memory: [batch_size,latent_dim] or [N,latent_dim]
            delta_t: [batch_size,1] or [N,1]
        Output:
            z: [batch_size,latent_dim]
        """
        delta_t_vec=self.w(delta_t) # [batch_size,latent_dim] or [N,latent_dim]
        delta_t_vec=delta_t_vec+1 # [batch_size,latent_dim] or [N,latent_dim]
        z=torch.mul(delta_t_vec,target_memory) # [batch_size,latent_dim] or [N,latent_dim]
        return z

class GraphAttention(nn.Module):
    def __init__(self,node_dim,latent_dim):
        super().__init__()
        self.query_linear=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.key_linear=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.value_linear=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.ffn=nn.Sequential(
            nn.Linear(latent_dim+node_dim+latent_dim+latent_dim,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,latent_dim)
        )

    def forward(self,tar_vec,h,neighbor_mask,tar_idx=None,step:Literal['step','last']='step'):
        """
        Input:
            step
                tar_idx: [B,1]
                tar_vec: [batch_size,node_dim+latent_dim+latent_dim], target node feature||time_feature
                h: [batch_size,N,node_dim+latent_dim+latent_dim], all node feature||time_feature
                neighbor_mask: [batch_size,N,], neighbor node mask
            last
                tar_vec: [N,node_dim+latent_dim+latent_dim], target node feature||time_feature
                h: [N,node_dim+latent_dim+latent_dim], all node feature||time_feature
                neighbor_mask: [N,N], neighbor node mask
        Output:
            step
                z: [batch_size,latent_dim]
            last
                z: [N,latent_dim]
        """
        if step=='step':
            feature_dim=tar_vec.size(1)

            q=self.query_linear(tar_vec) # [batch_size,latent_dim]
            k=self.key_linear(h) # [batch_size,N,latent_dim]
            v=self.value_linear(h) # [batch_size,N,latent_dim]

            q=q.unsqueeze(1) # [batch_size,1,latent_dim]
            attention_scores=torch.matmul(q,k.transpose(1,2))/(feature_dim**0.5) # [batch_size,1,latent_dim] X [batch_size,latent_dim,N] = [batch_size,1,N]

            # 이웃 노드 하나도 없는 경우 확인->없을 경우 자기 자신만 true가 되도록 mask 수정
            no_neighbor=~neighbor_mask.any(dim=1) # [B,] bool vec, 이웃 없는 행은 true로
            if no_neighbor.any():
                neighbor_mask[no_neighbor,tar_idx[no_neighbor,0]]=True

            neighbor_mask=neighbor_mask.unsqueeze(1) # [batch_size,1,N]
            attention_scores=attention_scores.masked_fill(~neighbor_mask,float('-inf')) # [batch_size,1,N]

            attention_weight=F.softmax(attention_scores,dim=-1) # [batch_size,1,N]

            neighbor_weight_sum=torch.matmul(attention_weight,v) # [batch_size,1,latent_dim]
            neighbor_weight_sum=neighbor_weight_sum.squeeze(1) # [batch_size,latent_dim]

            z=torch.cat([neighbor_weight_sum,tar_vec],dim=-1) # [batch_size,latent_dim||node+latent_dim+latent_dim]
            z=self.ffn(z) # [batch_size,latent_dim]
        else: # last
            num_nodes,feature_dim=tar_vec.size()

            q=self.query_linear(tar_vec) # [N,latent_dim]
            k=self.key_linear(h) # [N,latent_dim]
            v=self.value_linear(h) # [N,latent_dim]

            q=q.unsqueeze(1) # [N,1,latent_dim] 
            k=k.unsqueeze(0) # [1,N,latent_dim]
            v=v.unsqueeze(0) # [1,N,latent_dim]

            attention_scores=torch.matmul(q,k.transpose(1,2))/(feature_dim**0.5) # [N,1,latent_dim] X [1,latent_dim,N](broadcast됨) = [N,1,N] 

            # 이웃 노드 하나도 없는 경우 확인->없을 경우 자기 자신만 true가 되도록 mask 수정
            no_neighbor=~neighbor_mask.any(dim=1) # [B,] bool vec, 이웃 없는 행은 true로
            if no_neighbor.any():
                idx=torch.arange(num_nodes,device=neighbor_mask.device)
                neighbor_mask[no_neighbor,idx[no_neighbor]]=True

            neighbor_mask=neighbor_mask.unsqueeze(1) # [N,1,N] 
            attention_scores=attention_scores.masked_fill(~neighbor_mask,float('-inf'))

            attention_weight=F.softmax(attention_scores,dim=-1) # [N,1,N]

            neighbor_weight_sum=torch.matmul(attention_weight,v) # [N,1,latent_dim]
            neighbor_weight_sum=neighbor_weight_sum.squeeze(1) # [N,latent_dim]

            z=torch.cat([neighbor_weight_sum,tar_vec],dim=-1) # [N,latent_dim||node+latent_dim+latent_dim]
            z=self.ffn(z) # [N,latent_dim]
        return z

class GraphSum(nn.Module):
    def __init__(self,node_dim,latent_dim):
        super().__init__()
        self.w_1=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.w_2=nn.Linear(in_features=node_dim+latent_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.relu=nn.ReLU()

    def forward(self,tar_vec,h,neighbor_mask,tar_idx=None,step:Literal['step','last']='step'):
        """
        Input:
            step
                tar_vec: [batch_size,node_dim+latent_dim+latent_dim], target node feature||time_feature
                h: [batch_size,N,node_dim+latent_dim+latent_dim], all node feature||time_feature
                neighbor_mask: [batch_size,N,], neighbor node mask
            last
                tar_vec: [N,node_dim+latent_dim+latent_dim], target node feature||time_feature
                h: [N,node_dim+latent_dim+latent_dim], all node feature||time_feature
                neighbor_mask: [N,N], neighbor node mask
        Output:
            step
                z: [batch_size,latent_dim]
            last
                z: [N,latent_dim]
        """
        if step=='step':
            # 이웃 노드 하나도 없는 경우 확인->없을 경우 자기 자신만 true가 되도록 mask 수정
            no_neighbor=~neighbor_mask.any(dim=1) # [B,] bool vec, 이웃 없는 행은 true로
            if no_neighbor.any():
                neighbor_mask[no_neighbor,tar_idx[no_neighbor,0]]=True

            h=self.w_1(h) # [batch_size,N,latent_dim] 
            expanded_neighbor_mask=neighbor_mask.unsqueeze(-1).float() # [batch_size,N,1]
            h_hat=(h*expanded_neighbor_mask).sum(dim=1) # [batch_size,latent_dim]
            h_hat=self.relu(h_hat) # [batch_size,latent_dim] 
            z=torch.cat([tar_vec,h_hat],dim=-1) # [batch_size,node_dim+latent_dim+latent_dim+latent_dim]
            z=self.w_2(z) # [batch_size,latent_dim]
        else: # last
            # 이웃 노드 하나도 없는 경우 확인->없을 경우 자기 자신만 true가 되도록 mask 수정
            num_nodes=tar_vec.size(0)
            no_neighbor=~neighbor_mask.any(dim=1) # [B,] bool vec, 이웃 없는 행은 true로
            if no_neighbor.any():
                idx=torch.arange(num_nodes,device=neighbor_mask.device)
                neighbor_mask[no_neighbor,idx[no_neighbor]]=True

            h=self.w_1(h) # [N,latent_dim]
            h=h.unsqueeze(0) # [1,N,latent_dim]
            expanded_neighbor_mask=neighbor_mask.unsqueeze(-1).float() # [N,N,1]
            h_hat=(h*expanded_neighbor_mask).sum(dim=1) # [N,latent_dim]
            h_hat=self.relu(h_hat) # [N,latent_dim]                       
            z=torch.cat([tar_vec,h_hat],dim=-1) # [N,node_dim+latent_dim+latent_dim+latent_dim]         
            z=self.w_2(z) # [N,latent_dim]                               
        return z 


