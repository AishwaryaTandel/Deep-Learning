import torch as t
from trainer import Trainer
import sys
import torchvision as tv
import model

epoch = int(20)
#TODO: Enter your model here

layers=[1, 1, 1, 1]
model = model.ResNet(model.BasicBlock, layers)
#crit = t.nn.BCELoss()
weights=t.tensor([4.5,16]).cuda()
crit=t.nn.BCELoss(weight=weights)
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
