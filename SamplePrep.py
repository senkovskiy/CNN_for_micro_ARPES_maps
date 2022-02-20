import h5py
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy import interpolate
from pathlib import Path
import matplotlib.patches as patches
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

class ARPESdata:

    _data_path = 'salsaentry_1/scan_data/data_12'
    _X_scale_path = 'salsaentry_1/scan_data/actuator_1_1'
    _Y_scale_path = 'salsaentry_1/scan_data/actuator_2_1'

    dict_labels_classes = {}
    label_list = []
    total_number_of_samples = 0
    samples = None
    labels = None

    def __init__(self, data_folder: str, filename: str) -> None:
        # Encapsulation:
        self.__data_folder = data_folder                                  # e.g. "C:\\Users\\Boris\\pyproj\\ARPES_GUI"
        self.__filename = filename                                        # e.g. "Dev_2021-12-10_01-38-23.nxs"
        ARPESdata.data_folder = data_folder
        ARPESdata.filename = filename
        ARPESdata.class_name = "full_ARPES_data"

    @classmethod
    def HDF5_loader(cls):
        cls.data_folder = ARPESdata.data_folder
        cls.filename = ARPESdata.filename
        file_path = Path(cls.data_folder)/cls.filename
        try:
            file = h5py.File(file_path, "r")
        except:
            raise Exception("Problem to load file from the data folder")
        print(f"Load file: {ARPESdata.filename}") 
        cls.ARPES_data = np.array(file[ARPESdata._data_path])
        cls.X_scale = np.array(file[ARPESdata._X_scale_path][0]) 
        cls.Y_scale = np.array(file[ARPESdata._Y_scale_path])

    # Abstraction: 
    def __define_class_name(self, class_name: str):
        self.class_name = class_name

    def __define_class_area(self, claass_XY_start_end):
        self.X_start = claass_XY_start_end[0]
        self.Y_start = claass_XY_start_end[1]
        self.X_end = claass_XY_start_end[2]
        self.Y_end = claass_XY_start_end[3]
        X_start_index = np.where(self.X_scale ==  self.X_start)[0][0]
        X_end_index = np.where(self.X_scale == self.X_end)[0][0]
        Y_start_index = np.where(self.Y_scale == self.Y_start)[0][0]
        Y_end_index = np.where(self.Y_scale == self.Y_end)[0][0]
        self.class_data = self.ARPES_data[Y_start_index:Y_end_index, X_start_index:X_end_index, :, :]
        self.number_of_samples = self.class_data.shape[0]*self.class_data.shape[1]
        return self.class_data
       
    @classmethod
    def create_classes(cls, class_label_list, XY_start_end_list):
        for i, class_label in enumerate(class_label_list):
            cls.dict_labels_classes[class_label] = ARPESdata(ARPESdata.data_folder, ARPESdata.filename)                 # see https://stackoverflow.com/questions/13260557/create-new-class-instance-from-class-method      
            cls.dict_labels_classes[class_label].__define_class_name(class_label)
            cls.dict_labels_classes[class_label].__define_class_area(XY_start_end_list[i])
            
            for label in range(cls.dict_labels_classes[class_label].number_of_samples):
                cls.label_list.append(i)

        cls.total_number_of_samples = len(cls.label_list)

    '''
    def create_classes_as_np(self, class_label_list, XY_start_end_list):
        for i, class_label in enumerate(class_label_list):
            ARPESdata.dict_labels_classes[class_label] = self.__class__(self.data_folder, self.filename)                 # see https://stackoverflow.com/questions/13260557/create-new-class-instance-from-class-method      
            ARPESdata.dict_labels_classes[class_label].__define_class_name(class_label)
            ARPESdata.dict_labels_classes[class_label].__define_class_area(XY_start_end_list[i])
            
            for label in range(ARPESdata.dict_labels_classes[class_label].number_of_samples):
                ARPESdata.label_list.append(i)

        ARPESdata.total_number_of_samples = len(ARPESdata.label_list)
    '''
    @classmethod
    def convert_classes_labels_to_tensors(cls, height, width):
        sample_list = []
        for class_label, class_item in cls.dict_labels_classes.items():
            print("class_item: ", class_item.class_data)
            class_item = torch.from_numpy(class_item.__reshape_array(class_item.class_data))
            sample_list.append(class_item)

        cls.samples = torch.cat(sample_list, 0)
        cls.samples = cls.__transformation(cls.samples, height, width)
        cls.labels = torch.LongTensor(cls.label_list)
        return cls.samples, cls.labels
        
    @classmethod
    def __transformation(cls, samples, height, width):                          # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        samples_transform = torch.zeros(samples.shape[0], 1, height, width)
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((height,width)), transforms.ToTensor()])
        for i in range(samples.shape[0]):
            samples_transform[i][0] = transform(samples[i][0])
        return samples_transform
        
    def __reshape_array(self, array):
        im = np.reshape(array, (array.shape[0]*array.shape[1], 1, array.shape[2],-1))      # e.g. (8,30,96,941) go to (240, 1, 96, 941) - 240 samples, 96-angle, 941-energy
        return im

    @staticmethod
    def image_6_samples(samples, labels, start=238):
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.gca().set_title(labels[i+start]) # gca means "get current axes"
            #plt.imshow(transform(samples[i+start][0]), extent=[0, 10, 0, 10]) # samples[i+start][0]
            plt.imshow(samples[i+start][0])
        plt.show()

    @staticmethod
    def split(images, labels, test_size=0.2, random_state=1, shuffle=True):
        return train_test_split(images, labels, test_size=test_size, random_state=random_state, shuffle=shuffle)
  
    def safe_class_as_np(self):
        np.save(self.class_name, self.class_data)      
                             
    def draw_class_area(self, edgecolor='r'):
        # show integrated image:
        integrated_image = np.sum(self.ARPES_data, axis=(2,3)) # np.rot90
        f, axarr = plt.subplots(2,1) 
        # prepare image to plot, set axis:
        axarr[0].imshow(integrated_image, extent=[self.X_scale[0], self.X_scale[-1],self. Y_scale[0], self.Y_scale[-1]], origin="lower") 
        axes = plt.gca() 
        axarr[0].axes.set_xlim([self.X_scale[0], self.X_scale[-1]]) 
        axarr[0].axes.set_ylim([self.Y_scale[0], self.Y_scale[-1]]) 

        X_width = self.X_end-self.X_start
        Y_width = self.Y_end-self.Y_start
        rect1 = patches.Rectangle((self.X_start, self.Y_start), X_width, Y_width, linewidth=1, edgecolor=edgecolor, facecolor='none')
        axarr[0].axes.add_patch(rect1) 
        
        integrated_class_image = np.sum(self.class_data, axis=(2,3))   
        # print('integrated_seleced.shape ', integrated_seleced.shape)
        axarr[1].imshow(integrated_class_image, extent=[self.X_start, self.X_end, self.Y_start, self.Y_end], origin="lower") 
        axarr[1].axes.set_xlim([self.X_start, self.X_end])
        axarr[1].axes.set_ylim([self.Y_start, self.Y_end]) 
        plt.show()
        plt.close()

    @property
    def data_folder(self):
        return self.__data_folder 

  #  @data_folder.setter
  #  def data_folder(self, new_folder):
  #      if isinstance(new_folder, str): 
  #          self.__data_folder = new_folder
  #      else:
  #          raise Exception("The data_folder should be a string")     
    @property
    def filename(self):
        return self.__filename

   # @filename.setter
   # def filename(self, new_filename):
   #     if isinstance(new_filename, str): 
   #         self.__filename = new_filename
   #     else:
   #         raise Exception("The filename should be a string")


    
    

        
