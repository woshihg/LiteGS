import torch
import os
from ..training.optimizer import get_optimizer
def load_checkpoint(file_path):
    loaded_dict=torch.load(file_path)
    optimizer:torch.optim.Optimizer=loaded_dict['optimizer']
    schedular:torch.optim.lr_scheduler._LRScheduler=loaded_dict['schedular']
    start_epoch:int=loaded_dict['epoch']+1

    parameters={}
    for group in optimizer.param_groups:
        parameters[group['name']]=group['params'][0]

    return parameters["xyz"],parameters["scale"],parameters["rot"],parameters["sh_0"],parameters["sh_rest"],parameters["opacity"],parameters.get("features",None),start_epoch,optimizer,schedular,loaded_dict.get("classifier",None),loaded_dict.get("cls_optimizer",None)

def save_checkpoint(model_path,epoch,optimizer,schedular,classifier=None,cls_optimizer=None):
    os.makedirs(model_path, exist_ok = True) 
    file_path=os.path.join(model_path,"chkpnt{0}.pth".format(epoch))
    save_dict={
        "epoch":epoch,
        "optimizer":optimizer,
        "schedular":schedular
    }
    if classifier is not None:
        save_dict["classifier"] = classifier.state_dict()
    if cls_optimizer is not None:
        save_dict["cls_optimizer"] = cls_optimizer.state_dict()
    torch.save(save_dict,file_path)
    return