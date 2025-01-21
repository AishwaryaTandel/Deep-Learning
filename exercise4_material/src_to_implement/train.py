import torch
import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import data
import torchvision.models as models
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings
warnings.filterwarnings('ignore')




# load the data from the csv file and perform a train-test-split
df=pd.read_csv("data.csv",sep=';')
Train_dl,Val_dl=train_test_split(df,test_size=0.10,random_state=200,stratify=df[['crack', 'inactive']])
# x=len(Train_dl[(Train_dl["crack"]==0) & (Train_dl["inactive"]==0)])
# y=len(Train_dl[(Train_dl["crack"]==0) & (Train_dl["inactive"]==1)])
# z=len(Train_dl[(Train_dl["crack"]==1) & (Train_dl["inactive"]==0)])
# z1=len(Train_dl[(Train_dl["crack"]==1) & (Train_dl["inactive"]==1)])
# l=[x,y,z,z1]
# max=np.max(l)
# df_1=Train_dl[(Train_dl["crack"]==0) & (Train_dl["inactive"]==0)]
# df_2=Train_dl[(Train_dl["crack"]==0) & (Train_dl["inactive"]==1)]
# df2_=pd.concat([df_2]*126, ignore_index=True)
# df2=df2_.append([df2_[0:4]],ignore_index=True)
# df_3=Train_dl[(Train_dl["crack"]==1) & (Train_dl["inactive"]==0)]
# df3_=pd.concat([df_3]*4, ignore_index=True)
# df3=df3_.append([df3_[0:190]], ignore_index=True)
# df_4=Train_dl[(Train_dl["crack"]==1) & (Train_dl["inactive"]==1)]
# df4_=pd.concat([df_4]*14, ignore_index=True)
# df4=df4_.append([df4_[0:4]], ignore_index=True)
# Upsamled_train_dl=pd.concat([df_1, df2,df3,df4], ignore_index=True)

# print(len(df_1))
# print(len(df2))
# print(len(df3))
# print(len(df4))


# print(x,y,z,z1)
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dl=t.utils.data.DataLoader(ChallengeDataset(Train_dl,"train"),batch_size=24,shuffle=True,num_workers=2)
val_dl=t.utils.data.DataLoader(ChallengeDataset(Val_dl,"val"),batch_size=24,shuffle=True,num_workers=2)
# TODO

# create an instance of our ResNet model
layers=[1, 1, 1, 1]
model = model.ResNet(model.BasicBlock, layers)
# model=models.densenet161(pretrained=True)
# model=models.resnet18(pretrained=True)
# n_feat=model.fc.in_features
# model.fc=nn.Linear(n_feat,2)
# model.classifier=torch.nn.Sequential(torch.nn.Sigmoid())

## either add sigmoid here or we can add BCEwith logistic loss



# TODO
# from torchinfo import summary
#summary(model)
print(model)

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
#crit=nn.BCELoss()
weights=t.tensor([7.0,16.0]).cuda()
# weights=t.tensor([2,2]).cuda()
crit=t.nn.BCELoss(weight=weights)
# crit=nn.BCEWithLogitsLoss()###for resnet18 and other model
# set up the optimizer (see t.optim)
optim=t.optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.01)
# create an object of type Trainer and set its early stopping criterion
trainer=Trainer(model,crit,optim,train_dl,val_dl,early_stopping_patience=15)
# TODO

# go, go, go... call fit on trainer
res = trainer.fit(100)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')