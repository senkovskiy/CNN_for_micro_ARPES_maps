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
from sklearn.preprocessing import StandardScaler

data_folder = Path("C:\\Users\\Boris\\pyproj\\ARPES_GUI")
filename = data_folder/"D1MoSWSe_2020-12-16_16-02-55.nxs"
filename = data_folder/"Dev_2021-12-10_01-38-23.nxs"

f = h5py.File(filename, "r")

ARPES_data = np.array(f['salsaentry_1/scan_data/data_12'])#[:, ::-1]
print("ARPES_data.shape : ", ARPES_data.shape)                            #  ARPES_data.shape: (45, 101, 96, 941) - (Y, X, angle, energy)

X_scale = np.array(f['salsaentry_1/scan_data/actuator_1_1'][0]) 
Y_scale = np.array(f['salsaentry_1/scan_data/actuator_2_1']) 
#print("X_scale :", X_scale)
#print("X_scale.shape :", X_scale.shape)

	#axes.set_aspect(2*(k_scale[-1] - k_scale[0])/(angle_scale[-1] - angle_scale[0]))         

def selected_area(X_start, Y_start, X_end, Y_end, ARPES_data):
	# prepare rectangle to plot, set limits:
	#X_start = 150
	#Y_start = 120
	#X_end = 165
	#Y_end = 124
	X_width = X_end-X_start
	Y_width = Y_end-Y_start
	X_start_index = np.where(X_scale == X_start)[0][0]
	X_end_index = np.where(X_scale == X_end)[0][0]
	#print('X_start_index, X_end_index: ', X_start_index, X_end_index)
	Y_start_index = np.where(Y_scale == Y_start)[0][0]
	Y_end_index = np.where(Y_scale == Y_end)[0][0]
	return ARPES_data[Y_start_index:Y_end_index, X_start_index:X_end_index, :, :]

def draw_area(X_start, Y_start, X_end, Y_end, ARPES_data, edgecolor='r'):
	#show integrated image:
	integrated_image = np.sum(ARPES_data, axis=(2,3)) #np.rot90
	f, axarr = plt.subplots(2,1) 
	# prepare image to plot, set axis:
	axarr[0].imshow(integrated_image, extent=[X_scale[0], X_scale[-1], Y_scale[0], Y_scale[-1]], origin="lower") 
	axes = plt.gca() 
	axarr[0].axes.set_xlim([X_scale[0], X_scale[-1]]) 
	axarr[0].axes.set_ylim([Y_scale[0],Y_scale[-1]]) 

	selected = selected_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	X_width = X_end-X_start
	Y_width = Y_end-Y_start
	rect1 = patches.Rectangle((X_start, Y_start), X_width, Y_width, linewidth=1, edgecolor=edgecolor, facecolor='none')
	axarr[0].axes.add_patch(rect1) 
	integrated_seleced = np.sum(selected, axis=(2,3))   
	# print('integrated_seleced.shape ', integrated_seleced.shape)
	axarr[1].imshow(integrated_seleced, extent=[X_start, X_end, Y_start, Y_end], origin="lower") 
	axarr[1].axes.set_xlim([X_start, X_end])
	axarr[1].axes.set_ylim([Y_start, Y_end]) 
	plt.show()
	plt.close()

def graphene_area(ARPES_data):
	X_start = 150
	Y_start = 120
	X_end = 165
	Y_end = 124
	graphene_area = selected_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	draw_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	np.save('graphene', graphene_area)

def WSe2_area(ARPES_data):
	X_start = 171
	Y_start = 118
	X_end = 180
	Y_end = 123
	WSe2_area = selected_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	draw_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	np.save('WSe2', WSe2_area)

def MoS2_area(ARPES_data):
	X_start = 130
	Y_start = 135
	X_end = 143
	Y_end = 139
	MoS2_area = selected_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	draw_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	np.save('MoS2', MoS2_area)

def Hetero_area(ARPES_data):
	X_start = 155
	Y_start = 130
	X_end = 162.5
	Y_end = 132.5
	Hetero_area = selected_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	draw_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	np.save('Hetero', Hetero_area)

def hBN_area(ARPES_data):
	X_start = 130
	Y_start = 118
	X_end = 131.5
	Y_end = 124
	hBN_area = selected_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	draw_area(X_start, Y_start, X_end, Y_end, ARPES_data)
	np.save('hBN', hBN_area)

np.save('ARPES_data', ARPES_data)

#path = '/numpy_data'
#os. mkdir(path)
#os.chdir(path)
#cwd = os.getcwd()
#print("Current working directory: {0}".format(cwd))
#p.save('graphene', selected_area1)

graphene_area(ARPES_data)
load_graphene = np.load('graphene.npy')
print("load_graphene.shape", load_graphene.shape)

WSe2_area(ARPES_data)
load_WSe2 = np.load('WSe2.npy')
print("load_WSe2.shape", load_WSe2.shape)

MoS2_area(ARPES_data)
load_MoS2 = np.load('MoS2.npy')
print("load_MoS2.shape", load_MoS2.shape)

hBN_area(ARPES_data)
load_hBN = np.load('hBN.npy')
print("load_hBN.shape", load_hBN.shape)

Hetero_area(ARPES_data)
load_Hetero = np.load('Hetero.npy')
print("load_hBN.shape", load_Hetero.shape)

def reshape(array):
	im = np.reshape(array, (array.shape[0]*array.shape[1], 1, array.shape[2],-1))
	#print("im.shape", im.shape) # (36, 1, 96, 941)
	#print("im.shape", im.shape) #(96, 941)
	return im

def plot_reshaped(array):
	pass

#plt.imshow(reshape(load_hBN)[1,:,:,:][0], extent=[0, 100, 0, 100])
#plt.show()

classes = {'graphene', 'WSe2', 'MoS2', 'hBN'}

graphene_samples = torch.from_numpy(reshape(load_graphene))
print('graphene_samples', graphene_samples.shape)

WSe2_samples = torch.from_numpy(reshape(load_WSe2))
print('WSe2_samples', WSe2_samples.shape)

MoS2_samples = torch.from_numpy(reshape(load_MoS2))
print('MoS2_samples', MoS2_samples.shape)

hBN_samples = torch.from_numpy(reshape(load_hBN))
print('hBN_samples', hBN_samples.shape)

#Hetero_samples = torch.from_numpy(reshape(load_Hetero))
#print('Hetero_samples', Hetero_samples.shape)

amount_of_graphene_samples = list(graphene_samples.shape)[0] # 240
amount_of_WSe2_samples = list(WSe2_samples.shape)[0] # 180
amount_of_MoS2_samples = list(MoS2_samples.shape)[0] # 208
amount_of_hBN_samples = list(hBN_samples.shape)[0] # 36
#amount_of_Hetero_samples = list(Hetero_samples.shape)[0] #

labels_list = []
for i in range(amount_of_graphene_samples):
	labels_list.append(0)	#[1,0,0]
for i in range(amount_of_WSe2_samples):
	labels_list.append(1)
for i in range(amount_of_MoS2_samples):
	labels_list.append(2)
for i in range(amount_of_hBN_samples):
	labels_list.append(3)
#for i in range(amount_of_Hetero_samples):
#	labels_list.append(4)
labels = torch.LongTensor(labels_list)
print('labels:', labels)

samples = torch.cat((graphene_samples, WSe2_samples, MoS2_samples, hBN_samples), 0) # Hetero_samples
print("samples[1,:,:,:][0].shape : ", samples[1,:,:,:][0].shape)
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((48,48))])
#rescale_samples = transform(samples[1,:,:,:][0])
#print("rescale_samples", rescale_samples.size)

#def image_6_samples(start):
#	for i in range(6):
#		plt.subplot(2,3,i+1)
#		plt.gca().set_title(labels[i+start]) # gca means "get current axes"
#		plt.imshow(transform(samples[i+start][0]), extent=[0, 10, 0, 10]) # samples[i+start][0]
#	plt.show()

#image_6_samples(238)

def transformation(height, width, samples):              # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
	samples_transform = torch.zeros(samples.shape[0], 1, height, width)
	transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((height,width)), transforms.ToTensor()])
	for i in range(samples.shape[0]):
		samples_transform[i][0] = transform(samples[i][0])
	return samples_transform

#transformation(48, 48, samples)

def image_6_samples(samples, labels, start):
	for i in range(6):
		plt.subplot(2,3,i+1)
		plt.gca().set_title(labels[i+start]) # gca means "get current axes"
		#plt.imshow(transform(samples[i+start][0]), extent=[0, 10, 0, 10]) # samples[i+start][0]
		plt.imshow(samples[i+start][0])
	plt.show()

image_6_samples(transformation(48,48,samples), labels, 238)

# test, train preparation:
images = transformation(48,48,samples)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1, shuffle=True)

#sc=StandardScaler()
#X_train = sc.fit_transform(X_train.numpy())
#X_test =  sc.transform(X_test.numpy())

image_6_samples(X_train, y_train, 238)


conv1=nn.Conv2d(1, 2, 5) 
pool =nn.MaxPool2d(2, 2)
conv2=nn.Conv2d(2, 6, 5)
print('images.shape: ', images.shape) #  torch.Size([456, 1, 48, 48])
x = conv1(images) 
print(x.shape)  # [456, 2, 44, 44]
x=pool(x)
print(x.shape) # [456, 2, 22, 22]
x=conv2(x)
print(x.shape) # [456, 6, 18, 18]
x=pool(x)
print(x.shape) # [456, 6, 9, 9]

class SimpleCNN(nn.Module):
	def __init__(self): # input_size, hidden_size, num_classes
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 2, 5) # input_chan_size, output_chan_size, kernel    # out_channels tells us how many filters to use - in other words, how many feature maps we want for the convolutional layer.
		# (W-F+2P)/S + 1, W=48, F=5, P=0, S=1  => 48-5+1 = 44 
		self.pool = nn.MaxPool2d(2, 2)
		# reduces the images by a factor of 2 => size (..., 2, 22, 22)
		self.conv2 = nn.Conv2d(2, 6, 5) 
		self.fc1 = nn.Linear(6*9*9, 120) # flatten
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,4)                        # 4 class

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x))) 
		x = x.view(-1,6*9*9) # flatten
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
		# no softfmax - it is alredy includede in our loss

# Hyper-parameters
num_epochs = 110
batch_size = 4
learning_rate = 0.001

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()  # 		applies nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)
										# Y must not be one-hot encoded, also Y_pred has raw scores, no Softmax!	
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = X_train.shape[0] # 273
print('n_total_steps', n_total_steps)

for epoch in range(num_epochs):

	#Forward pass:
	y_predicted = model(X_train)
	loss = criterion(y_predicted, y_train)
	
	#Backward and optimize
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()	

	if (epoch+1) % 20 == 0:
		#print('y_predicted', y_predicted)
		#print('torch.max(y_predicted, 1).indices', torch.max(y_predicted, 1).indices)
		print(f'epoch: {epoch+1}, loss = {loss.item():.3f}')

with torch.no_grad():
	n_correct = 0
	n_samples = 0
	
	output = model(X_test)
	#print('output', output)
	_, predicted = torch.max(output.data, 1)
	print('test', y_test)
	print('predicted', predicted)
	n_samples = y_test.size(0)
	n_correct = (predicted == y_test).sum().item()

	acc = 100.0 * n_correct / n_samples
	print(f'Accuracy on the test samples: {acc} %')


def APES_image_classification(ARPES_data):
	spatial_height = ARPES_data.shape[0]
	spatial_width = ARPES_data.shape[1]
	print('spatial_hight, spatial_width = ', spatial_height, spatial_width)
	resized = torch.from_numpy(reshape(ARPES_data))
	print('resized.shape', resized.shape)
	resized_images = transformation(48,48,resized)

	with torch.no_grad():

		output = model(resized_images)
		_, predicted = torch.max(output.data, 1)

	#plot:
	predicted_spatial = np.reshape(predicted, (ARPES_data.shape[0],-1))
	print('predicted_spatial.shape', predicted_spatial.shape)
	print('predicted_spatial: ', predicted_spatial)
	plt.imshow(predicted_spatial, interpolation='none', extent=[X_scale[0], X_scale[-1], Y_scale[0], Y_scale[-1]], origin="lower")
	plt.show()


APES_image_classification(ARPES_data)