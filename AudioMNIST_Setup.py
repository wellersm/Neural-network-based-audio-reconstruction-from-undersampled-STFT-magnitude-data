import csv
import os
import random

# Make a list of all the data
path = 'AudioMNIST/data'
annot_list = []
for folder in os.listdir(path):
    if os.path.isdir(path + '/' + folder):
        for file in os.listdir(path + '/' + folder):
            file_path = path + '/' + folder + '/' + file
            label = file[0]
            annot_list.append((file_path, label))

# Shuffle the annotation list
random.seed(17)
random.shuffle(annot_list)

# Split in train and test lists
train_size = int(0.8*len(annot_list))
train_list = annot_list[:train_size]
test_list = annot_list[train_size:]

# Save the train and test lists
with open('Train.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for item in train_list:
        csv_writer.writerow([item[0], item[1]])

with open('Test.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for item in test_list:
        csv_writer.writerow([item[0], item[1]])