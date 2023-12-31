import zipfile
import os, shutil
import requests
import pandas as pd
from torch.utils.data import Dataset
from skimage import io

class FakeDogDataset(Dataset):
    def __init__(self,transform=None):

        # get data
        src_url='https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/?dl=1'
        filename='low-resolution.zip'
        r=requests.get(src_url,stream=True,timeout=None)
        if r.status_code==200:
            print('downloading...')
            r.raw.decode_content=True
            with open(filename,'wb') as f:
                shutil.copyfileobj(r.raw,f)
            print('download successful')
        else:
            print('failure')

        with zipfile.ZipFile('/content/imgs/low-resolution.zip','r') as zip_ref:
            zip_ref.extractall('/content/imgs/')

        for root,_,filenames in os.walk('/content/imgs/low-resolution'):
            for filename in filenames:
                path=os.path.join(root,filename)
                shutil.move(path,'/content/imgs/Dogs')
        shutil.rmtree('/content/imgs/low-resolution')

        # other setup
        self.imgs=pd.Series([x[2] for x in os.walk('/content/imgs/Dogs')][0])
        self.transform=transform


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_path=os.path.join('/content/imgs/Dogs',self.imgs.iloc[idx])
        img=io.imread(img_path)

        if self.transform:
            img=self.transform(img)

        return img