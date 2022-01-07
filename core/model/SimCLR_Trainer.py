import sys
import os

from torch.optim.lr_scheduler import CosineAnnealingLR


sys.path.append(os.getcwd())
import torch
from core.loss import SimCLRLoss
import torch.nn.functional as F
from tqdm import tqdm as tqdm
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import SGD
from torch.utils.data import DataLoader,SubsetRandomSampler
from core.utils.schedulers import CyclicCosineDecayLR


global NUMWORKER
NUMWORKER=4
global LATENTPLOT_N
LATENTPLOT_N=2000

    
class SimCLR_Trainer():
    def __init__(self,model):
        self._optimizer=None
        self._scheduler=None
        self._logwriter=None
        self._logdir=None
        self._logbatch=False
        self._nworker=NUMWORKER
        self._best={}
        self._bestmodel={}
        self._modelpath=''
        self._clipping=None
        self._device=torch.device("cpu")
        if torch.cuda.is_available():
            self._device=torch.device('cuda')
        # On crée les propriétés du modèle et du projecteur.
        self._model=model.to(self._device)
        self._LossName='SimCLR'
        self._Losses={
            'SimCLR':SimCLRLoss(temperature=0.5),
            }
    def setLogdir(self,logdir,suffix="CLR",logbatch=False):
        """Define the Logging directory where we'll write :
        - Results (Tensorboard)
        - Models (Only Bests)

        Args:
            logdir ([type]): [description]
            suffix (str, optional): [description]. Defaults to "HMM".
        """
        self._logbatch=logbatch
        os.makedirs(logdir,exist_ok=True)
        if os.path.exists(logdir):
            logdir = os.path.join(logdir,datetime.now().strftime("%y%m%d_%H%M")+f"-{suffix}")
            self._logwriter=SummaryWriter(logdir)
            self._logdir=logdir
        return

    def setOptimizer(self,Lr0=0.1):
        """Set the optimizer use for training.
        We 'll use  an optimizer that perform well on this task.

        Args:
            Lr0 (float, optional): [description]. Defaults to 0.1.
        """
        assert self._model is not None,"Model should be defined prior Scheduler"
        self._optimizer=SGD([
            {'params':self._model.parameters()},
            ],
            lr=Lr0,momentum=0.9,weight_decay=1e-4)
        self._scheduler=CosineAnnealingLR(self._optimizer,Lr0,0.05,last_epoch=-1)
 
    def step(self,batch,batch_idx,epoch,batch_count,prefix=""):
        """ Iteration Step of our trainer
        It will compute the loss and feed the loss grabber.

        Args:
            batch ([type]): [description]
            batch_index ([type]): [description]
            epoch ([type]): [description]
            batchcount ([type]): [description]

        Returns:
            [dict]: losses
            [dict]: outputs
        """
        X_1,X_2,c=batch
        # We put everything on device...
        X_1=X_1.to(self._device)
        X_2=X_2.to(self._device)
        batch_size=X_1.size()[0]
        # On passe tout ça dans le CNN pour générer h_1 et h_2
        h_1,z_1=self._model(X_1)
        h_2,z_2=self._model(X_2)
        # On calcule la loss sur Z1 et Z2
        criterion=self._Losses[self._LossName]
        loss=criterion(z_1,z_2)
        # On la divise par le nombre de samples.
        if self._logbatch:
            self._logwriter.add_scalar(f'{prefix}Batch_{self._LossName}',loss.item(),batch_idx+epoch*batch_count)
        return {'loss':loss},{'z_1':z_1,'z_2':z_2,'h_1':h_1,'h_2':h_2}

    def train_step(self,batch,batch_idx,epoch,batch_count):
        """The trainstep will call the standard "step" method. But it will backpropagate the loss in order to update our model.

        Args:
            batch ([type]): [description]
            batch_index ([type]): [description]
            epoch ([type]): [description]
            batchcount ([type]): [description]

        Returns:
            [type]: [description]
        """
        self._optimizer.zero_grad()
        losses,outputs=self.step(batch,batch_idx,epoch,batch_count,prefix="Training/")
        losses['loss'].backward()
        self._optimizer.step()
        return losses,outputs
    def val_step(self,batch,batch_idx,epoch,batch_count):
        """The valstep won't update our model so it is necessary to use "no_grad()" method.

        Args:
            batch ([type]): [description]
            batch_index ([type]): [description]
            epoch ([type]): [description]
            batchcount ([type]): [description]

        Returns:
            [type]: [description]
        """
        with torch.no_grad():
            losses,outputs=self.step(batch,batch_idx,epoch,batch_count,prefix="Validation/")
            return losses,outputs

    def fit(self,dataset,trainindices,valindices,batchsize=1,epochs=10):
        assert self._model is not None,"Model should be defined prior Fitting"
        assert self._optimizer is not None,"Optimizer should be defined prior Fitting"
        assert self._logwriter is not None,"Logwriter should be defined prior Fitting"
        # Si gamma, on le met sur le device d'execution
        tloader=DataLoader(dataset,batch_size=batchsize,num_workers=self._nworker,sampler=SubsetRandomSampler(trainindices),persistent_workers=True)
        vloader=DataLoader(dataset,batch_size=batchsize,num_workers=self._nworker,sampler=SubsetRandomSampler(valindices),persistent_workers=True)
        self._model=self._model.to(self._device)
        # On démarre les iterrations
        for epoch in range(epochs):
            print(f'\nTrain Epoch {epoch+1}/{epochs}')
            dataset.train(True)
            self._model.train()
            trainbatch_count=len(trainindices)//batchsize
            trainbatchs=tqdm(enumerate(tloader),total=trainbatch_count)
            # On reset le lossgrabber
            for index,batch in trainbatchs:
                # Boucle d'apprentissage
                losses,_=self.train_step(batch,index,epoch,trainbatch_count)
                trainbatchs.set_postfix(**{loss:value.item() for (loss,value) in losses.items()})
            # Validation
            dataset.train(False)
            self._model.eval()
            valbatch_count=len(valindices)//batchsize
            valbatchs=tqdm(enumerate(vloader),total=valbatch_count)
            z_val=[] # Latent projected Space
            h_val=[] # Latent Space
            c_val=[] # Classes
            x_val=[] # Images (if small dataset)
            for index,batch in valbatchs:
                X1,X2,c=batch
                # Boucle de test
                losses,outputs=self.val_step(batch,index,epoch,valbatch_count)
                if index*batchsize<LATENTPLOT_N:
                    z_val.append(outputs['z_1'])
                    h_val.append(outputs['h_1'])
                    x_val.append(X1)
                    c_val.append(c)
                valbatchs.set_postfix(**{loss:value.item() for (loss,value) in losses.items()})
            z_val=torch.cat(z_val,dim=0).cpu()
            h_val=torch.cat(h_val,dim=0).cpu()
            c_val=torch.cat(c_val,dim=0).cpu()
            x_val=torch.cat(x_val,dim=0).cpu()
            # On plot les espaces latent
            if z_val.shape[1]>2:
                # On ajoute un embedding avec tensorboard pour visualiser tout ça si on a plus que 2 dimensions.
                metadata=getclass(c_val,dataset.classes)
                self._logwriter.add_embedding(h_val,metadata=metadata,label_img=x_val,tag='val_HEmbedding',global_step=epoch)
            if self._scheduler:
                self._scheduler.step()


def getclass(mat,classes):
    res=mat.tolist()
    if classes:
        res=[classes[i] for i in res]
    return res

if __name__=='__main__':
    print('terminé')

