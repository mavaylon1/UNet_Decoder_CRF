from glob import glob
import re
import random
import
import shutil


img_path = ""
ann_path = ""
imgs = glob(img_path+"/*"")

imgs_titles = [x[60:] for x in imgs] #The 60 is the length of the string for the part of the path before the image file names

animals=[]
for image in imgs2:
    animals.append(re.findall("^(.+?)(?=\\d).*",image)[0])

animal_types=list(set(animals)) #The unique strata of pet breeds

train_paths=[]
for animal in animal_types:
    res = [i for i in imgs if animal in i]
    train_paths.append(random.sample(res, round(.8*len(res))))

train=(list(itertools.chain.from_iterable(train_paths))) #combines all the list of train data from each breed into one list

train_titles = [x[60:] for x in train]
ann = glob(ann_path+'/*')
ann_paths=[]
for titles in train_titles:
    ann_paths.append([i for i in ann if titles in i])
ann_train=(list(itertools.chain.from_iterable(ann_paths)))

for file in train:
    shutil.move(file,'/Users/mavaylon/Research/Pet_Stratified_Training/imgs')

for file in ann_train:
    shutil.move(file,'/Users/mavaylon/Research/Pet_Stratified_Training/ann')
