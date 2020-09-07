import random
seed = 7
random.seed(seed)
randI = random.sample(range(1,16), 15)
randI = [str(i) for i in randI]
# print(randI)

import os
import torch
import re
import shutil
from PIL import Image


def tiff2jpeg(namef, nameS, c, image_tiff):
    # Check directory exists or not
    directory = os.path.dirname('./dataset/Semi_BOEdata/' + namef + '/' + nameS + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    outfile = directory +'/'+ nameS + str(c) + '.jpeg'
    image_tiff.thumbnail(image_tiff.size)
    image_tiff.save(outfile, "JPEG", quality=100)


data_dir = './dataset/Publication_Dataset'
torch.manual_seed(seed)

if not os.path.exists(data_dir):
    print("BOE dataset not exists")
    exit()

patternName = re.compile(r'(?<=Publication_Dataset/)\D+')

# Define number of training samples for each class
train_ratio = 0.025
val_ratio = 0.025
test_ratio = 0.25
unlabel_ratio = 1 - train_ratio - val_ratio - test_ratio


# remove the folder before regenerating the result
if os.path.exists('./dataset/Semi_BOEdata/'):
    shutil.rmtree('./dataset/Semi_BOEdata/')


# Total 15 persons, each person consists of  AMD, DME, Normal

amd_count = 1
for i, j, k in os.walk(data_dir):
    file_counts = len(k)
    if file_counts > 1:

        i = i.replace('\\','/')

        # Calculate how many files for train_org, val and test
        train_count = int(train_ratio * file_counts)
        val_count = int(val_ratio * file_counts)
        test_count = int(test_ratio * file_counts)
        unlabel_count = int(file_counts - train_count - val_count - test_count)
        temp_list = list(k)

        # File Partition
        random.shuffle(temp_list)
        file_train_list = temp_list[0:train_count]
        file_val_list = temp_list[train_count:train_count+val_count]
        file_test_list = temp_list[train_count+val_count:train_count+val_count+test_count]
        file_unlabel_list = temp_list[train_count+val_count+test_count:]

        cla = patternName.findall(i)[0]  # find class: AMD,DME or Normal
        for fileN in k:
            # label = 0
            f = i + '/' + fileN
            print(f)
            image_tiff = Image.open(f)
            if cla == 'AMD':
                pa = re.compile(r'(?<=AMD)\d+')
            elif cla == 'DME':
                pa = re.compile(r'(?<=DME)\d+')
            elif cla == 'NORMAL':
                pa = re.compile(r'(?<=NORMAL)\d+')
            else:
                print('Error!')

            CI = randI.index(pa.findall(i)[0])  # find index in randI to determine whether (train_org, test, val) or unlabel


            if fileN in file_train_list:
                print('to train')
                tiff2jpeg('train', cla, amd_count, image_tiff)
            elif fileN in file_val_list:
                print('to val')
                tiff2jpeg('val', cla, amd_count, image_tiff)
            elif fileN in file_test_list:
                print('to test')
                tiff2jpeg('test', cla, amd_count, image_tiff)
            elif fileN in file_unlabel_list:
                print('to unlabel')
                tiff2jpeg('unlabel', cla, amd_count, image_tiff)
            else:
                print("Error")

            amd_count += 1

print(amd_count) #3231