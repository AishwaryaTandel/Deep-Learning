import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json

class ImageGenerator:
    def __init__(self,file_path,label_path,batch_size,Imagesize,rotation=False,mirroring=False,shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.Imagesize = Imagesize
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.epoch = -1
        self.generator = self.gen()
    def gen(self):
        while True:
            with open(self.label_path) as f:
                data = json.load(f)
            index = list(data.keys())
            index = [int(i) for i in index]
            if len(index) % self.batch_size != 0:
                t = len(os.listdir(self.file_path)) // self.batch_size
                q = self.batch_size * (t + 1)
                remaining_img = q - len(os.listdir(self.file_path))
                index.extend(index[0:remaining_img])

            self.num_batches = len(index) // self.batch_size

            width = self.Imagesize[0]
            height = self.Imagesize[1]
            channel = self.Imagesize[2]
            if self.shuffle:
                index = np.random.permutation(index)
                for self.single_batch in range(self.num_batches):
                    self.batch_data = np.zeros((self.batch_size, width, height, channel))
                    self.batch_labels = np.zeros((self.batch_size, 10), dtype=int)
                    batch_images = index[self.batch_size * self.single_batch:(self.batch_size * (self.single_batch + 1))]
                    for i in range(self.batch_size):
                        image_fr = "".join([str(batch_images[i]), '.npy'])
                        image = np.load(os.path.join(self.file_path, image_fr))
                        image = cv2.resize(image, (width, height)).astype(np.float32)
                        image = self.augment(image, self.mirroring, self.rotation)
                        self.batch_data[i, :, :, :] = image[:, :, :]
                        self.batch_labels[i, data[str(batch_images[i])]] = 1
                        if self.single_batch == self.batch_size - 1:
                            index = np.random.permutation(index)
                    yield self.batch_data, self.batch_labels
            else:
                for self.single_batch in range(self.num_batches):
                    self.batch_data = np.zeros((self.batch_size, width, height, channel))
                    self.batch_labels = np.zeros((self.batch_size, 10), dtype=int)
                    batch_images = index[self.batch_size * self.single_batch:(self.batch_size * (self.single_batch + 1))]
                    for i in range(self.batch_size):
                        image_name = "".join([str(batch_images[i]), '.npy'])
                        image = np.load(os.path.join(self.file_path, image_name))
                        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
                        image = self.augment(image, self.mirroring, self.rotation)
                        self.batch_data[i, :, :, :] = image[:, :, :]
                        self.batch_labels[i, data[str(batch_images[i])]] = 1
                    yield self.batch_data, self.batch_labels

    def next(self):
        return self.generator.__next__()

    def augment(self,image,mirroring,rotation):
        if mirroring:
            if np.random.randint(1, 3) == 1:
                image = np.fliplr(image)
            else:
                image = np.flipud(image)
        if rotation:
            image = np.rot90(image, np.random.randint(2, 4))
        return image

    def show(self):
        data_of_img, label = self.batch_data, self.batch_labels
        size_data_first = data_of_img.shape[0]
        labels = []
        for l in label:
            labelList = list(l)
            labels.append(labelList.index(1))
        plt.figure(figsize=(10, 10))
        for i in range(size_data_first):
            plt.subplot((self.batch_size // 3) + 1, 3, i + 1)
            final_image = np.array(data_of_img[i], np.int32)
            plt.imshow(final_image)
            plt.title(self.dict[labels[i]])
            plt.axis("off")
        plt.show()

    def current_epoch(self):
        if self.single_batch == 0:
            self.epoch = self.epoch + 1
        return self.epoch

    def class_name(self, labels):
        label = labels.split('.')[0]
        with open(self.label_path) as f:
            data = json.load(f)
        class_index = data[label]
        final_class_name = self.dict[class_index]
        return final_class_name