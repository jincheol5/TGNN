
    # @staticmethod
    # def train_old(model,train_data_loader_list,val_data_loader_list=None,validate:bool=False,config:dict=None):
    #     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model.to(device)
    #     optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])

    #     """
    #     Early stopping
    #     """
    #     if config['early_stop']:
    #         early_stop=EarlyStopping(patience=config['patience'])

    #     """
    #     model train
    #     """
    #     for epoch in tqdm(range(config['epochs']),desc=f"Training..."):
    #         model.train()
    #         loss_list=[]
    #         for train_data_loader in tqdm(train_data_loader_list,desc=f"Training {epoch+1} epoch..."):
    #             tar_label_list=[batch['tar_label'] for batch in train_data_loader] # List of [B,1], B는 각 element마다 다를 수 있음
    #             last_label=train_data_loader[-1]['label'] # [N,1]
    #             tar_label_list=[tar_label.to(device) for tar_label in tar_label_list]
    #             last_label=last_label.to(device)

    #             output=model(data_loader=train_data_loader,device=device)
    #             pred_step_logit_list=output['step_logit_list']
    #             pred_last_logit=output['last_logit']
    #             step_loss=Metrics.compute_step_tR_loss(logit_list=pred_step_logit_list,label_list=tar_label_list)
    #             last_loss=Metrics.compute_last_tR_loss(logit=pred_last_logit,label=last_label)
    #             total_loss=step_loss+last_loss
    #             loss_list.append(total_loss)

    #             # back propagation
    #             optimizer.zero_grad()
    #             total_loss.backward()
    #             optimizer.step()

    #         epoch_loss=torch.stack(loss_list).mean().item()
    #         print(f"Epoch Loss: {epoch_loss}")
            
    #         """
    #         Early stopping
    #         """
    #         if config['early_stop']:
    #             val_loss=epoch_loss
    #             pre_model=early_stop(val_loss=val_loss,model=model)
    #             if early_stop.early_stop:
    #                 model=pre_model
    #                 print(f"Early Stopping in epoch {epoch+1}")
    #                 break

    #         """
    #         wandb log
    #         """
    #         if config['wandb']:
    #             wandb.log({
    #                 f"loss":epoch_loss,
    #             },step=epoch)

    #         """
    #         validate
    #         """
    #         if validate:
    #             step_acc,last_acc=ModelTrainer.test(model=model,data_loader_list=val_data_loader_list)
    #             print(f"{epoch+1} epoch tR validation step_acc: {step_acc} last_acc: {last_acc}")

    # @staticmethod
    # def test_old(model,data_loader_list):
    #     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model.to(device)
    #     model.eval()

    #     """
    #     model test
    #     """
    #     with torch.no_grad():
    #         step_acc_list=[]
    #         last_acc_list=[]
    #         for data_loader in tqdm(data_loader_list,desc=f"Evaluating..."):
    #             tar_label_list=[batch['tar_label'] for batch in data_loader]
    #             last_label=data_loader[-1]['label'] # [N,1]
    #             tar_label_list=[tar_label.to(device) for tar_label in tar_label_list]
    #             last_label=last_label.to(device)

    #             output=model(data_loader=data_loader,device=device)
    #             pred_step_logit_list=output['step_logit_list']
    #             pred_last_logit=output['last_logit']

    #             step_acc=Metrics.compute_step_tR_acc(logit_list=pred_step_logit_list,label_list=tar_label_list)
    #             last_acc=Metrics.compute_last_tR_acc(logit=pred_last_logit,label=last_label)
    #             step_acc_list.append(step_acc)
    #             last_acc_list.append(last_acc)
    #     step_acc=float(np.mean(step_acc_list))
    #     last_acc=float(np.mean(last_acc_list))
    #     return step_acc,last_acc
