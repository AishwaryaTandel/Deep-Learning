import numpy as np
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


# from torchmetrics.functional import f1_score

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # -propagate through the network
        outputs = self._model.forward(x)
        # outputs = self._model(x)
        loss_net = self._crit.forward(outputs, y)
        # -compute gradient by backward propagation
        loss_net.backward()
        # -update weights

        self._optim.step()
        # -return the loss
        return loss_net
        # TODO

    def val_test_step(self, x, y):
        # predict
        outputs = self._model(x)

        # propagate through the network and calculate the loss and predictions
        loss__valid = self._crit(outputs, y)
        # return the loss and the predictions
        return loss__valid, outputs
        # TODO

    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set
        epoch_loss = 0
        # for _,(inputs, labels) in tqdm(enumerate(self._train_dl), total=int(len(self._train_dl)/self._train_dl.batch_size)):
        for inputs, labels in self._train_dl:
            inputs = inputs.cuda()
            labels = labels.cuda()
            # inputs = inputs.to(self._cuda)
            # labels = labels.to(self._cuda)
            loss = self.train_step(inputs, labels)
            epoch_loss += loss
            # _, preds = t.max(outputs.data, 1)
            # train_running_correct += (preds == labels).sum().item()

        avarage_loss = epoch_loss / len(self._train_dl)
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        return avarage_loss
        # TODO

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        # t.no_grad() ###with torch.no_grad()
        # iterate through the validation set
        loss_val = 0
        prediction = []
        # labels=self._val_test_dl[1]
        fscore = 0
        with t.no_grad():
            for data, labels in self._val_test_dl:
                data, labels = data.cuda(), labels.cuda()
                # labels.to(self._cuda)
                loss, pred = self.val_test_step(data, labels)
                loss_val += loss
                prediction.append(pred)
                # print((np.argmax(labels.cpu().detach().numpy(), axis=1)).shape)
                # print((np.round(pred.cpu().detach().numpy())).shape)
                f1__score = f1_score(labels.cpu().detach().numpy(),
                                     np.round(pred.cpu().detach().numpy()), average='weighted')  ###weighted
                fscore = fscore + f1__score

        avarage_loss = loss_val / len(self._val_test_dl)
        finalf1 = fscore / int(len(self._val_test_dl))
        # f1__score = f1_score(labels,prediction)
        return avarage_loss, finalf1
        # return avarage_loss,f1__score

        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        # TODO

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        # TODO
        self.train_loss = []
        self.valid_loss = []
        epoch_ = 0
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

        while True:
            # stop by epoch number
            if epoch_ == epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            # valid_loss,f1score=self.val_test()
            valid_loss, f1score = self.val_test()

            # append the losses to the respective lists
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # self.save_checkpoint(epoch_)
            if epoch_ % 1 == 0:
                self.save_checkpoint(epoch_)

            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if self.best_loss == None:
                self.best_loss = valid_loss
            elif self.best_loss - valid_loss > 0:
                self.best_loss = valid_loss
                # reset counter if validation loss improves
                self.counter = 0
            elif self.best_loss - valid_loss < 0:
                self.counter += 1
                print(f"INFO: Early stopping counter {self.counter} of {self._early_stopping_patience}")
                if self.counter >= self._early_stopping_patience:
                    print('INFO: Early stopping')
                    self.early_stop = True
            if self.early_stop == True:
                break
            # return the losses for both training and validation

            # print(f'Epoch {epoch_ + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
            print(
                f'Epoch {epoch_} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss} \t\tF1_Score: {f1score}')
            epoch_ += 1
        return self.train_loss, self.valid_loss
        # TODO






