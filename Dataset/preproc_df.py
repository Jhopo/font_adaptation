import glob
import os
import sys
from shutil import copyfile

data_dir = '../../dataset/Fonts/'
target_dir = '../../dataset/big/'

filenames = sorted(glob.glob(os.path.join(data_dir, '*.ttf')))
for filename in filenames:
    if 'DF' in filename:
        font_name = filename.split('/')[-1]
        copyfile(os.path.join(data_dir, font_name), os.path.join(target_dir, font_name))
