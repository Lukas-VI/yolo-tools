#yolo-ultimate_推理up_scale
#为了使得使用小分辨率的yolov8模型可以检测到大分辨率的目标，一种新的方法——Upsampling。
#Upscale的基本思想是把推理的大图片给切成小图片，然后对每个小图片进行推理，得到小图片的预测结果，然后把预测结果放大到原图的尺寸上。
#具体的算法如下：
#1. 首先，将原图划分成若干个小图块，每个小图块的大小按实际输入的大小和模型训练时的大小进行调整。
#2. 对每个小图块进行推理，得到预测结果。
#3. 将预测结果放大到原图的尺寸上。
#4. 将所有预测结果叠加到一起，得到最终的预测结果。

#导入所需的库
from pyexpat import model
import re
from tempfile import tempdir
import cv2
import numpy as np
import time as t
import os
from ultralytics import YOLO
import math
from PIL import Image


# 设置参数
class Config:
    def __init__(self, model_path, img_path,model_size):
        self.model_path = model_path  # 模型路径
        self.img_path = img_path      # 输入图片路径
        self.model_size = model_size  # 模型输入大小
    
    def display_config(self):
        print(f"模型路径: {self.model_path}")
        print(f"输入图片路径: {self.img_path}")
        print(f"模型输入大小: {self.model_size}")






def upscale_divide(img, model_size):
    # 切割图片,对无法整除的部分图片加上黑边补齐，并保证在实际图片上切割次数最少
    h, w, _ = img.shape
    h_step = int(np.ceil(h / model_size))        # 计算切割次数
    w_step = int(np.ceil(w / model_size))        # 计算切割次数
    h_pad = (h_step * model_size - h) // 2           # 计算补齐的黑边高度
    w_pad = (w_step * model_size - w) // 2           # 计算补齐的黑边宽度
    img = cv2.copyMakeBorder(img, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_list = []
    for i in range(h_step):   
        for j in range(w_step):
            print("cutting chunk ({},{})".format(j,i))
            img_list.append(img[i*model_size:i*model_size+model_size, j*model_size:j*model_size+model_size])
            #cv2.imshow("result", img)
    return img_list

def original_image_size(img):
    # 计算原始图片的大小
    h, w, _ = img.shape
    size=[h, w]
    return size

def chunk_count(img, model_size):
    global n, m
    # 计算切割图片块的数量
    h, w, _ = img.shape
    n = math.ceil(h / model_size)
    m = math.ceil(w / model_size)

def once_inference(little_img):
    # 进行一次推理
    ticks = str(t.asctime())
    # Load a model
    model = YOLO(config.model_path)  # pretrained YOLOv8n model
    print(ticks)
    # Run batched inference on a list of images
    results = model.cuda()(little_img)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        #result.show()  # display to screen
        
        im=result.plot()
        print("1 plot processed")
        
    #函数返回单个小图块的预测结果
    return im

def proceess_img(img_list):
    # 处理图片
    lg=len(img_list)
    i=0
    for j,img in enumerate(img_list):
        # 进行一次推理
        im = once_inference(img)
        # 保存预测结果，替换img_list中的元素原图
        
        img_list[j] = im
        #print(t.time())
        i+=1
        print ("{}/{}".format(i,lg))

    return img_list


def combine_images(img_list, model_size):
    # 合并图片
    global img
    chunk_count(img, model_size)
    new_img_size0 = model_size * n
    new_img_size1 = model_size * m

    new_img = Image.new('RGB', (new_img_size0, new_img_size1), 'white')

    for i, f in enumerate(img_list):
        row = int(i / n)       # 计算行，比如有n=3，第0个图片在第0行，求行数就是i//n
        col = i % n            # 计算列，
        # 确保 img 为 numpy 数组类型，并转换颜色格式
        #if isinstance(img, np.ndarray):
        
        img = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB)  # 转换颜色格式
        img = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL Image
        img = img.resize((model_size, model_size))  # 调整大小
        new_img.paste((img,  row * model_size, col * model_size))

    result = np.array(new_img)  # 转换成 NumPy 数组返回
    return result


# 主函数
if __name__ == '__main__':
    
    # 设置参数
    config = Config(
    model_path="D:/DeepLearning/yolov8/runs/detect/train5/weights/best.pt",
    img_path="D:/DeepLearning/teset.jpg", 
    model_size=1280
    )
    
    # 显示参数
    config.display_config()

    # 读取图片
    img = cv2.imread(config.img_path)
    # 切割图片
    img_list = upscale_divide(img,config.model_size)
    # 进行推理
    proceess_img(img_list)
    # 合并图片
    #size=original_image_size(img)

    result = combine_images(img_list, config.model_size)
    # 保存结果
    cv2.imwrite("{}.jpg".format(str(t.asctime())), result)
    print("Done!")
    #使用cv2.imshow()显示结果
    cv2.imshow("result", result)
    cv2.waitKey(0)
    #cv2.destroyAllWindows() 
    #打开存放预测结果的图片的文件夹
    os.system("explorer .")