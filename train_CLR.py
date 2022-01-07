import sys
import os
import matplotlib
# Pour éviter un plantage Tk à la 120 ème itération...
matplotlib.use('Agg')


sys.path.append(os.getcwd())
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from core.datasets import CIFAR_Dataset,STL10_Dataset
from core.model import SimCLRModel,SimCLR_Trainer
import json

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import argparse
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,choices=['CIFAR','STL10'],default='STL10')
    parser.add_argument('--train_ratio',type=float,help='Train Ratio',default=0.85)
    parser.add_argument('--val_ratio',type=float,help='Validation Ratio',default=0.15)
    # Latent space
    parser.add_argument('--z_dim',type=int,help='Latent Space Size',default=128)
    parser.add_argument('--backbone',default='resnet18',choices=['resnet18','resnet34','resnet50'])
    # Training
    parser.add_argument('--batch_size',type=int,help='Batch size image Count',default=128)
    parser.add_argument('--epochs',type=int,help='Epochs Count',default=100)
    parser.add_argument('--logdir',help='Logging directory (for tensorboard)',default="results")
    args=parser.parse_args()
    config = vars(args)
    config['Comment']='RestartLR defined to None in order cosine scheduler to restart from high LR.'
    config['suffix']=''
    # On commence par créer le dataset....
    if args.dataset=='STL10':
        dataset=STL10_Dataset()
        config['NImages']=100000
        trainindices,valindices=dataset.getValIndices(args.train_ratio,args.val_ratio,seed=42,max_items=config['NImages'])
        config['X_W']=96
        config['X_H']=96
        config['X_C']=3
        config['Lr0']=0.3*config['batch_size']/256
        #config['Norm']=True
        transforms1=A.Compose([
             A.Resize(config['X_H'],config['X_W']),
             A.ToFloat(max_value=255.0),
             ToTensorV2()])
        transforms2=A.Compose([
             A.RandomResizedCrop(config['X_H'],config['X_W'],scale=(0.5,1)),
             A.ColorJitter(),
             A.HorizontalFlip(),
             A.VerticalFlip(),
             A.CoarseDropout(max_holes=1,fill_value=128),
             A.GaussNoise(),
             A.ToFloat(max_value=255.0),
             ToTensorV2()])
    elif args.dataset=='CIFAR':
        dataset=CIFAR_Dataset()
        trainindices,valindices,testindices=dataset.getTVTIndices(args.train_ratio,args.val_ratio,0)
        config['X_W']=32
        config['X_H']=32
        config['X_C']=3
        transforms1=A.Compose([
             A.Resize(config['X_H'],config['X_W']),
             A.ToFloat(max_value=255.0),
             ToTensorV2()])
        transforms2=A.Compose([
             A.RandomResizedCrop(config['X_H'],config['X_W'],scale=(0.5,1)),
             A.ColorJitter(),
             A.HorizontalFlip(),
             A.VerticalFlip(),
             A.GaussNoise(),
             A.CoarseDropout(max_holes=1,fill_value=128),
             A.ToFloat(max_value=255.0),
             ToTensorV2()])

    # Ensuite on crée les augmentations...
    dataset.transform1=transforms1
    dataset.transform2=transforms2
    # On déclare le modèle...
    model=SimCLRModel(
        x_H=config['X_H'],
        x_W=config['X_W'],
        x_depth=config['X_C'],
        z_dim=config['z_dim'],
        backbone=config['backbone'])
    trainer=SimCLR_Trainer(model)
    trainer._nworker=4
    trainer.setOptimizer(Lr0=config['Lr0'])
    trainer.setLogdir(config['logdir']+f"_{config['dataset']}",suffix=f"{config['backbone']}SimCLR_Z{config['z_dim']}_Sz{config['X_H']}{config['suffix']}")
    
    json_config = json.dumps(config, indent = 4)
    # Writing to sample.json
    with open(os.path.join(trainer._logdir,'config.json'), "w") as outfile:
        outfile.write(json_config)
    print("Launch Training...")
    trainer.fit(dataset,trainindices,valindices,batchsize=config['batch_size'],epochs=config['epochs'])