'''
    Partitions the data to 9:1 training to testing ratio.
'''


import sys, os
from shutil import copyfile
from PIL import Image

def cut_dot(fn):
    if fn.endswith(".jpg") or fn.endswith(".JPG"):
        return fn.removesuffix(".JPG") + ".xml"

def dir_copy(src, dest, filelist, ratio, c=0):
    while c < ratio:
        for i in filelist:
            if i.endswith(".JPG"):
                copyfile(f"{src}/{i}", f"{dest}/{i}")
                xml = cut_dot(i)
                copyfile(f"{src}/{xml}", f"{dest}/{xml}")
            c += 1
            if c >= ratio:
                return c

def ratio(src, dest1, dest2, r):
    filelist = os.listdir(src)
    test = r * len(filelist)
    cohort1 = dir_copy(src, dest1, filelist, test)
    train = (1.0 - r) * len(filelist)
    cohort2 = dir_copy(src, dest2, filelist, train, cohort1)

if __name__ == '__main__':
    ratio("workspace/data/images", "workspace/see-trash/images/test", "workspace/see-trash/images/train", 0.1)

