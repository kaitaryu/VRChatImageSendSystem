
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import collections
import random
import time
from pythonosc import udp_client
from pythonosc import dispatcher
import math
class ImageCompression():
    def __init__(self,image,start_point,size = None):

        self.image = image.copy()
        self.start_point = start_point
        self.count_data,_ = self.GetBackColer()

    def Get_Color(self):
        return (self.image[:,:,0] * (2**16)) + (self.image[:,:,1] * (2**8)) + (self.image[:,:,2])
    def GetBackColer(self):
        coler_list = self.Get_Color()
        count_data = collections.Counter(np.reshape(coler_list,-1))
        count_list =  list(count_data.keys())
        r = int(count_list[0]) & 255
        g = (int(count_list[0]) >> 8) & 255
        b = (int(count_list[0]) >> 16)& 255
        return count_data,[b,g,r]
    def Split4(self):

        x = self.image.shape[0]//2
        y = self.image.shape[1]//2
        if min(x,y ) <= 0:
            return []
        split_list = []
        split_list.append(ImageCompression(self.image[:x,:y].copy(),[self.start_point[0],self.start_point[1]]))
        split_list.append(ImageCompression(self.image[x:,:y].copy(),[self.start_point[0] + x,self.start_point[1]]))
        split_list.append(ImageCompression(self.image[:x,y:].copy(),[self.start_point[0],y +self.start_point[1]]))
        split_list.append(ImageCompression(self.image[x:,y:].copy(),[self.start_point[0] + x, y + self.start_point[1]]))
        return split_list
 
    def GetColerCount(self,imagelist):
        coler_count = 0
        count = 0
        if len(imagelist) <=0 :
            return 256
        for image_class in imagelist:
            coler_count +=len(list(image_class.count_data.keys()))
            count += 1
        coler_count /= count
        return coler_count

    def SplitMain(self):
        #最も同一色が多くなる分割方法を探す
        split4 = self.Split4()
        split_vertical =self.SplitVertical()

        split_beside = self.SplitBeside()


        return_split = split4
        min_count = self.GetColerCount(split4)  


        return return_split


def Coler2int(image):
    return (image[:,:,0] * (2**16)) + (image[:,:,1] * (2**8)) + (image[:,:,2]) 

def Int2Color(num):
    r = int(num) & 255
    g = (int(num) >> 8) & 255
    b = (int(num) >> 16)& 255
    return [b,g,r]

# 減色処理
def Kmeans_Color(src, K):
    # 次元数を1落とす
    Z = src.reshape((-1, 3))

    # float32型に変換
    Z = np.float32(Z)

    # 基準の定義
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # K-means法で減色
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # UINT8に変換
    center = np.uint8(center)

    res = center[label.flatten()]

    # 配列の次元数と入力画像と同じに戻す
    return res.reshape((src.shape))
def ImageSize(image,backcoler = [0,0,0]):
    if image.shape[0] < image.shape[1]:
        start = (image.shape[1]-image.shape[0])//2
        resize = np.zeros((image.shape[1],image.shape[1],3))
        resize[:,:,0] = backcoler[0]
        resize[:,:,1] = backcoler[1]
        resize[:,:,2] = backcoler[2]
        resize[start:start + image.shape[0],:,:] = image[:,:,:]
    else:
        start = (image.shape[0]-image.shape[1])//2
        resize = np.zeros((image.shape[0],image.shape[0],3))
        resize[:,:,0] = backcoler[0]
        resize[:,:,1] = backcoler[1]
        resize[:,:,2] = backcoler[2]
        resize[:,start:start + image.shape[1],:] = image[:,:,:]
    return resize

def imageResize(image,size = 512):
    if image.shape[0] < image.shape[1]:
        fxy = size/image.shape[1]
    else:
        fxy = size/image.shape[0]
    dst = cv2.resize(image,dsize = None,fx = fxy,fy = fxy)
    return dst
def GetBackColer(image):
    coler2int_list = Coler2int(image)
    count_data = collections.Counter(np.reshape(coler2int_list,-1))
    count_list =  list(count_data.keys())
    return count_data,Int2Color(count_list[0]),count_list[0],coler2int_list

def SendOSC(coler,starts,size,client,def_size = 512,bs_size=512):
    #
    b = def_size/bs_size
    client.send_message("/avatar/parameters/ColerR", coler[2])
    client.send_message("/avatar/parameters/ColerG", coler[1])
    client.send_message("/avatar/parameters/ColerB", coler[0])
    #なんか微妙にずれる、謎。とりあえず０．1づつずらす。
    size_bit = math.log2(int((size[0]*b)))
    add_state_X = (int(starts[0][0]*b)) & 256 
    add_state_Y = (int(starts[0][1]*b) & 256) 
    state_size = (add_state_Y >> 3) + (add_state_X  >> 4) + size_bit
    client.send_message("/avatar/parameters/data_01", state_size)
    for i,start in enumerate(starts):
        start_X = int(start[0]*b) & 255
        start_Y = int(start[1]*b) & 255

        client.send_message("/avatar/parameters/data_" + '{:0=2}'.format(i * 2 + 2), start_X)
        client.send_message("/avatar/parameters/data_" + '{:0=2}'.format(i * 2 + 3), start_Y)
    for i in range( len(starts),5):
        start_X = int(starts[0][0]*b) & 255
        start_Y = int(starts[0][1]*b) & 255
        client.send_message("/avatar/parameters/data_" + '{:0=2}'.format(i * 2 + 2), start_X)
        client.send_message("/avatar/parameters/data_" + '{:0=2}'.format(i * 2 + 3), start_Y)
    print(len(starts))
    time.sleep(0.05)
    client.send_message("/avatar/parameters/Plot", True)
    time.sleep(0.05)
    client.send_message("/avatar/parameters/Plot", False)
    time.sleep(0.03)
def MakeImage(colerlist,image,backcoler_int,def_size = 512,bs_size=512):
    b = def_size/bs_size
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter('./out/video.mp4',fourcc, 10.0, (image.shape[0],image.shape[1]))
    client = udp_client.SimpleUDPClient("127.0.0.1",9000)
    client.send_message("/avatar/parameters/Plot", False)
    make_image = np.zeros(image.shape)
    backcoler_coler = Int2Color(backcoler_int)
    make_image[:,:,0] = backcoler_coler[0]
    make_image[:,:,1] = backcoler_coler[1]
    make_image[:,:,2] = backcoler_coler[2]
    count = 0
    tmp_start_point = None
    image_size = None
    image_coler = None

    for coler in colerlist:
        out_coler = Int2Color(list(coler.count_data.keys())[0])
        start_x = coler.start_point[0]
        start_y = coler.start_point[1]
        end_x = coler.start_point[0] + coler.image.shape[0]
        end_y = coler.start_point[1] + coler.image.shape[1]
        make_image[start_x:end_x,start_y:end_y,0] = out_coler[0]
        make_image[start_x:end_x,start_y:end_y,1] = out_coler[1]
        make_image[start_x:end_x,start_y:end_y,2] = out_coler[2]

        if tmp_start_point is None:
            tmp_start_point = []
            tmp_start_point.append(coler.start_point)
            image_size = coler.image.shape[:2]
            image_coler = list(coler.count_data.keys())[0]
        elif (((coler.image.shape[0] != image_size[0]) or
                (coler.image.shape[1] != image_size[1]) or  
                ((int(coler.start_point[0] * b)&256) != (int(tmp_start_point[0][0] * b)&256)) or
                ((int(coler.start_point[1] * b)&256) != (int(tmp_start_point[0][1] * b)&256)) or  
                (image_coler != list(coler.count_data.keys())[0])) or
                len(tmp_start_point) >= 5):
            SendOSC(Int2Color(image_coler),tmp_start_point,image_size,client,def_size,bs_size)
            tmp_start_point = []
            tmp_start_point.append(coler.start_point)
            image_size = coler.image.shape[:2]
            image_coler = list(coler.count_data.keys())[0]
        else:
            tmp_start_point.append(coler.start_point)
        if count % 50 == 0:
            video.write(np.array(make_image,np.int8))
    SendOSC(Int2Color(image_coler),tmp_start_point,image_size,client,def_size,bs_size)
    video.release()
    return make_image

def MakeColerBoxList(dst):
    split_list = [ImageCompression(dst,[0,0])]
    
    image_list = []
    while len(split_list) > 0:
        tmp_split = []
        for split_class in split_list:
            coler_count = len(list(split_class.count_data.keys()))
            if coler_count > 1:
                list_ = split_class.SplitMain()
                tmp_split.extend(list_)
            else:
                if (list(split_class.count_data.values())[0]>= 64):
                    print(split_class.image.shape,split_class.start_point,list(split_class.count_data.keys())[0])
                image_list.append(split_class)
        split_list = tmp_split
    return image_list

def MakeColerBoxList_Main(dst,count_data):
    coler2int_list = Coler2int(dst)
    coler_list = list(count_data.keys())
    dst_copy = np.zeros(dst.shape)
    box_list = []
    for i,coler_int in enumerate(coler_list):
        tmp_list =  MakeColerBoxList(dst_copy)
        coler = Int2Color(coler_int)
        for coler_box in tmp_list:
            #一つでも指定した色を含んでいれば投入する
            key_list = [int(s) for s in coler_box.count_data.keys()]
            start_x = coler_box.start_point[0]
            start_y = coler_box.start_point[1]
            end_x = coler_box.start_point[0] + coler_box.image.shape[0]
            end_y = coler_box.start_point[1] + coler_box.image.shape[1]
            
            if np.any( coler2int_list[start_x:end_x,start_y:end_y]== coler_int) and np.any(np.array(key_list) == 0) :
                coler_box.image[:,:,0] = coler[0]
                coler_box.image[:,:,1] = coler[1]
                coler_box.image[:,:,2] = coler[2]
                add_coler_box = ImageCompression(coler_box.image,coler_box.start_point)
                box_list.append(add_coler_box)
        
        dst_copy[coler2int_list == coler_int,0] = coler[0]
        dst_copy[coler2int_list == coler_int,1] = coler[1]
        dst_copy[coler2int_list == coler_int,2] = coler[2]
    return box_list
def main():
    # 入力画像を取得
    img = cv2.imread("./image/2022-11-05 182254.png")
    dst = imageResize( img,64)
    dst = Kmeans_Color(dst, K=32)

    #黒色はmaskに使うのでBを１上げる
    count_data,backcoler,backcoler_int,coler2int_list = GetBackColer(dst)
    dst[coler2int_list == 0,0] = 1
    #RGBを1次元に圧縮する
    count_data,backcoler,backcoler_int,coler2int_list = GetBackColer(dst)
    
    print(count_data)
    print(backcoler)

    dst = ImageSize(dst,backcoler)
    image_list = MakeColerBoxList_Main(dst,count_data) 

    #image_list = MakeColerBoxList(dst)
    #image_list = MakeColerBoxList_sort(image_list)
    print(len(image_list))
    dst_make = MakeImage(image_list,dst,backcoler_int,bs_size=64)
    cv2.imwrite("./out/out.png",dst)
    cv2.imwrite("./out/out_2.png",dst_make)

if __name__ == "__main__":
    main()