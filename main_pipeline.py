from turtle import width
from SamplePrep import ARPESdata

data_folder = "C:\\Users\\Boris\\pyproj\\ARPES_GUI"
filename = "Dev_2021-12-10_01-38-23.nxs"

data = ARPESdata(data_folder, filename)
data.HDF5_loader()
#print(ARPESdata.data_folder)
#print(ARPESdata.filename)

class_label_list = ["graphene", "MoS2", "hBN"]
XY_start_end_list = [[150, 120, 165, 124],      # graphene
                    [130, 135, 143, 139],       # MoS2
                    [130, 118, 131.5, 124]]     # hBN

data.create_classes(class_label_list, XY_start_end_list)

print(data.dict_labels_classes["graphene"].class_data.shape)
print(data.dict_labels_classes["graphene"].data_folder)

data.dict_labels_classes["graphene"].draw_class_area()
print(data.label_list)
print("total_number_samples: ", data.total_number_of_samples)
width, height = 48, 48
images, labels = data.convert_classes_labels_to_tensors(width, height)
data.image_6_samples(images, labels, 237)

X_train, X_test, y_train, y_test = data.split(images, labels)





