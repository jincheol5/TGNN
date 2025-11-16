import os
import pickle

path=os.path.join('..','data','tgnn','train','train_20_chunk_dataset_10_31.pkl')
with open(path,"rb") as f:
    dataset_list=pickle.load(f)

count=0
for dataset in dataset_list:
    for dataseq in dataset:
        label=dataseq[-1]['label']
        if not label:
            count+=1
print(f"count: {count}")