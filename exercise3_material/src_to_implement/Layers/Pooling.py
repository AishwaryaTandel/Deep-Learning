import numpy as np

class Pooling:
    def __init__(self,stride_shape,pooling_shape):
        self.stride=stride_shape
        self.pooling=pooling_shape
        self.trainable=False
        self.memory=None
        self.matrix=None

    def forward(self,input_tensor):
        self.memory=input_tensor
        self.input_height=input_tensor.shape[2]
        self.input_width = input_tensor.shape[3]
        self.channel=input_tensor.shape[1]
        self.pooling_height=self.pooling[0]
        self.pooling_width = self.pooling[1]

        H= int(1+(self.input_height-self.pooling_height)/self.stride[0])
        W = int(1 + (self.input_width - self.pooling_width) / self.stride[1])
        Pooling_output=np.zeros((input_tensor.shape[0],self.channel,H,W))

        for self.b in range(input_tensor.shape[0]):
            for self.c in range(self.channel):
                for self.h in range(H):
                    for self.w in range(W):
                        V_start=self.h *self.stride[0]
                        V_end=V_start+self.pooling_height
                        H_start=self.w *self.stride[1]
                        H_end=H_start +self.pooling_width
                        self.height=V_end-V_start
                        self.width=H_end-H_start
                        x=input_tensor[self.b, self.c, V_start: V_end, H_start: H_end]
                        #self.masked_slice(x,input_tensor.shape[0],self.b,self.channel,H,W)
                        Pooling_output[self.b, self.c, self.h, self.w] = np.max(input_tensor[self.b, self.c, V_start: V_end, H_start: H_end])
        return Pooling_output

    def backward(self,error_tensor):
        previous_mat=self.memory
        output=np.zeros_like(previous_mat)

        for b in range(previous_mat.shape[0]):
            previous_mat_1=previous_mat[b] ###reduce dimention for further processing
            for c in range(error_tensor.shape[1]):
                for h in range(error_tensor.shape[2]):
                    for w in range(error_tensor.shape[3]):###change iteration over w
                        V_start = h * self.stride[0]
                        V_end = V_start + self.pooling_height
                        H_start = w * self.stride[1]
                        H_end = H_start + self.pooling_width
                        a_prev_slice=previous_mat_1[c, V_start: V_end, H_start: H_end]
                        mask=self.create_mask(a_prev_slice)
                        output[b, c, V_start:V_end, H_start:H_end] += error_tensor[b, c, h,w] * mask

        return output



    def create_mask(self,x):
        mask= x==np.max(x)
        return mask


