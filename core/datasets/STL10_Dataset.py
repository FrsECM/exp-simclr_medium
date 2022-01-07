from torchvision.datasets import STL10
from torch.utils.data import Dataset,random_split
import numpy as np
import cv2


class STL10_Dataset(Dataset):
    def __init__(self,gray=False):
        self.transform1 = None
        self.transform2 = None
        self.labeled=STL10("./data",split='train',download=True)
        self.unlabeled=STL10("./data",split='unlabeled',download=True)
        self._train=True
        self.classes=['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck','unlabelled']
    def __len__(self):
        return len(self.labeled)+len(self.unlabeled)
    def __getitem__(self, index):
        if index<len(self.labeled):
            img,classe=self.labeled[index]
        else:
            # On retranche la taille du dataset supervisé avant de prendre une donnée non labelisée.
            img,classe=self.unlabeled[index-len(self.labeled)]
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
    def getValIndices(self,TrainRatio,ValRatio,seed=42,max_items=100000):
        cnt=min(self.__len__(),max_items)
        total=TrainRatio+ValRatio
        # On commence par remplir le lot de validation avec les données labelisées...
        indices_labeled=[i for i in np.arange(len(self.labeled))]
        indices_unlabeled=[i+len(self.labeled) for i in np.arange(len(self.unlabeled))]
        # Count
        nbtrain=int(cnt*float(TrainRatio)/total)
        nbval=min(int(cnt*float(ValRatio/total)),len(self.labeled))
        # Seed
        np.random.seed(seed=seed)
        np.random.shuffle(indices_labeled)
        np.random.shuffle(indices_unlabeled)
        indices=indices_labeled+indices_unlabeled
        # Drop unnecessary Items :
        indices=indices[:cnt]
        # we take val first in order to get supervised indices...
        val=indices[:nbval]
        train=indices[nbval+1:]
        return train,val



if __name__=="__main__":
    ds=STL10_Dataset()
    t,v=ds.getValIndices(0.8,0.2,seed=42,max_items=100000)
    a=ds[12]
    a=ds[10]
