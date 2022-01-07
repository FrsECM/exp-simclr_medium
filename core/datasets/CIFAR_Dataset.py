from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset,random_split
import numpy as np
import cv2


class CIFAR_Dataset(Dataset):
    def __init__(self):
        self.transform1 = None
        self.transform2 = None
        self.cifar=CIFAR10("./data",train=True,download=True)
        self._train=True
        self.classes=[
            'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck']
    def __len__(self):
        return len(self.cifar)
    def __getitem__(self, index):
        img,classe=self.cifar[index]
        x=np.array(img)
        # On definit x1 et x2
        x1=x
        x2=x
        if self.transform1:
            x1=self.transform1(image=x)['image']
        if self.transform2:
            x2=self.transform2(image=x)['image']
        return x1,x2,classe
    def train(self,train=True):
        self._train=train
    def getTVTIndices(self,TrainRatio,ValRatio,TestRatio,seed=42):
        cnt=self.__len__()
        total=TrainRatio+ValRatio+TestRatio
        nbtrain=int(cnt*float(TrainRatio)/total)
        nbtest=int(cnt*float(TestRatio/total))
        nbval=int(cnt*float(ValRatio/total))
        indices=np.arange(self.__len__())
        np.random.seed(seed=seed)
        np.random.shuffle(indices)
        train=indices[:nbtrain]
        val=indices[nbtrain+1:nbtrain+nbval]
        test=indices[nbtrain+nbval+1:]
        return train,val,test



if __name__=="__main__":
    ds=CIFAR_Dataset()
    a=ds[10]