# 汉明距离表示两个（相同长度）字对应位不同的数量，我们以
# d（x,y）表示两个字x,y之间的汉明距离。对两个字符串进行异或运算，
# 并统计结果为1的个数，那么这个数就是汉明距离。
# 向量相似度越高，对应的汉明距离越小。如10001001和10010001有2位不同。
from PIL import Image
from functools import reduce
import pandas as pd
import numpy as np
# import time


# 计算Hash
def phash(img):
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    return reduce(
        lambda x, y: x | (y[1] << y[0]),
        enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())),
        0
    )

# 计算汉明距离
def hamming_distance(a, b):
    return bin(a ^ b).count('1')


# 计算图片相似度
def is_imgs_similar(img1, img2):
    return True if hamming_distance(phash(img1), phash(img2)) <= 5 else False


if __name__ == '__main__':
    # img1_path = 'C:/Users/lwq_1997/Desktop/test/GroundDisconnector_270.jpg'
    # img2_path = "C:/Users/lwq_1997/Desktop/test/GroundDisconnector_270.jpg"

    # img1 = Image.open(img1_path)
    # img2 = Image.open(img2_path)

    # start_time = time.time()
    dataB = np.mat([8, 9, 2])
    dataC = np.mat([3, 5, 1])
    a = is_imgs_similar(dataB, dataC)
    # end_time = time.time()
    print(a)

