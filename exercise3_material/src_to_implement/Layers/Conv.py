import numpy as np
from scipy import signal
import copy

class Conv:
    def __init__(self,stride_shape, convolution_shape,num_kernels):
        self.stride_shape=stride_shape
        if len(stride_shape)==1:
            self.stride_2=1
        else:
            self.stride_2 = self.stride_shape[1]

        self.stride_1 = self.stride_shape[0]
        self.convolution_shape=convolution_shape
        self.c=self.convolution_shape[0]
        self.m=self.convolution_shape[1]
        if len(self.convolution_shape)==2:
            self.n = 1
        else:
            self.n = convolution_shape[2]


        self.num_kernels=num_kernels
        self.trainable=True
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(self.num_kernels,self.c,self.m,self.n))
        self.weights_copy=self.weights.copy()
        #self.bias=np.random.uniform(low=0.0, high=1.0, size=(self.num_kernels,1))
        self.bias = np.random.uniform(low=0.0, high=1.0, size=(self.num_kernels,1,1))
        self._optimizer=None
        self.bias_optimizer=None
        self.memory=None
        #self.b_optimizer=copy.deepcopy(self.Optimizer)


    @property
    def gradient_weights(self):
        pass

    @property
    def gradient_bias(self):
        pass

    def forward(self, input_tensor):
        y=input_tensor.shape
        if len(input_tensor.shape)==3:
            input_tensor=np.reshape(input_tensor,(input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],1))
        self.memory = input_tensor
        self.output_size=(input_tensor.shape[0],self.num_kernels,input_tensor.shape[2],input_tensor.shape[3])
        #self.output_size = (input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_1)), int(np.ceil(input_tensor.shape[3]/self.stride_2)))
        self.output_array = np.zeros(self.output_size)



        for b in range(input_tensor.shape[0]):
            output=self.output_array[b]
            input=input_tensor[b]
            for c in range(self.num_kernels):
                for channel in range(input_tensor.shape[1]):
                        output[c] += signal.correlate2d(input[channel], self.weights[c, channel], mode="same")


        output__=np.zeros((input_tensor.shape[0],self.num_kernels,int(np.ceil(input_tensor.shape[2]/self.stride_1)),int(np.ceil(input_tensor.shape[3]/self.stride_2))))
        for b in range(input_tensor.shape[0]):  ###iterate over batch element###
            for k in range(self.num_kernels):
                for h in range(output__.shape[2]):
                    for w in range(output__.shape[3]):
                        V_start = h * self.stride_1
                        V_end = V_start + self.m
                        H_start = w * self.stride_2
                        H_end = H_start + self.n
                        output__[b, k, h, w] = self.output_array[b,k,V_start,H_start]

        output__= output__+self.bias
        if output__.shape[3]==1:
            output__=np.reshape(output__,(output__.shape[0],output__.shape[1],output__.shape[2]))

        return output__

    def initialize(self,weights_initializer, bias_initializer):
        we_in=weights_initializer
        b_in=bias_initializer
        self.weights=we_in.initialize((self.num_kernels,*self.convolution_shape),np.prod(self.convolution_shape),np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias=b_in.initialize((self.num_kernels,1,1),1,1)#self.num_kernels

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self.bias_optimizer = copy.deepcopy(self._optimizer)
        return self._optimizer

    def backward(self,error_tensor):
        if len(error_tensor.shape)==3:
            error_tensor=np.reshape(error_tensor,(error_tensor.shape[0],error_tensor.shape[1],error_tensor.shape[2],1))
        re_error_tensor=np.zeros((self.memory.shape[0],error_tensor.shape[1],self.memory.shape[2],self.memory.shape[3]))
        self.dx=np.zeros(self.memory.shape)

    ###upsample output
        for b in range(error_tensor.shape[0]):
            for k in range(self.num_kernels):
                for h in range(error_tensor.shape[2]):
                    for w in range(error_tensor.shape[3]):
                        V_start = h * self.stride_1
                        V_end = V_start + self.m
                        H_start = w * self.stride_2
                        H_end = H_start + self.n
                        re_error_tensor[b, k, V_start, H_start] = error_tensor[b,k,h,w]
        ##gradient wrt bias
        self.db=np.zeros_like(self.bias)
        for b in range(error_tensor.shape[0]):
            for h in range(error_tensor.shape[1]):
                self.db[h]+=np.sum(error_tensor[b,h])


        ###gradient wrt weights
        ##padd inputs self.memory  half kernal width
        if len(error_tensor.shape)==3:
            padded_input = np.zeros((self.memory.shape[0], self.memory.shape[1], self.memory.shape[2] + self.m,1))
        else:
            padded_input = np.zeros((self.memory.shape[0], self.memory.shape[1], self.memory.shape[2] + self.m,
                                     self.memory.shape[3] + self.n))

        for b in range(padded_input.shape[0]):
            for c in range(padded_input.shape[1]):
                padded_input[b, c] = np.pad(self.memory[b, c], [(int(np.floor(self.m/2)), int(np.ceil(self.m/2))), (int(np.floor(self.n/2)), int(np.ceil(self.n/2)))], mode='constant')
        if len(error_tensor.shape)==3:
            padded_input = padded_input[:, :, :-1, :]
        else:
            padded_input = padded_input[:, :, :-1, :-1]


        # print(padded_input.shape)
        self.dw=np.zeros_like(self.weights)
        for b in range(self.memory.shape[0]):
            dx_ = padded_input[b]
            # dx__=self.dx[b]
            for c in range(self.num_kernels):
                for channel in range(padded_input.shape[1]):
                    self.dw[c,channel] += signal.correlate2d(dx_[channel], re_error_tensor[b, c], mode="valid")
        ##update_parameter
        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, self.dw)
            self.bias=self.bias_optimizer.calculate_update(self.bias,self.db)


        ##gradient_wrt X
        ##make fisr channel kernal bunch, then second and then 3rd
        kernel_reshape=np.zeros((self.c,self.num_kernels,self.m,self.n))
        for chanel in range(self.weights.shape[1]):
            for kernal in range(self.num_kernels):
                kernel_reshape[chanel,kernal]=self.weights_copy[kernal,chanel]

        #kernel_reshape=np.reshape(self.weights_copy,(self.c,self.num_kernels,self.m,self.n))##mistake loop throught H
        for b in range(self.memory.shape[0]):

            for c in range(self.c):
                for H in range(self.num_kernels):
                    self.dx[b, c] += signal.convolve2d(re_error_tensor[b, H], kernel_reshape[c, H],mode="same")
        if self.dx.shape[3] == 1:
            self.dx = np.reshape(self.dx, (self.dx.shape[0], self.dx.shape[1], self.dx.shape[2]))
        return self.dx

    @property
    def gradient_weights(self):
        return self.dw
    @property
    def gradient_bias(self):
        return self.db

    def calculate_regularization_loss(self):
        self.regulation_loss = 0
        if self.trainable is True:
            if self.optimizer.regularizer is not None:
                self.regulation_loss = self.optimizer.regularizer.norm(self.weights)

        return self.regulation_loss
