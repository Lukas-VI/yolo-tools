#数据集缺项查找
#读取整个文件夹中的图片和txt文件名，并对比两者的差异，找到缺少的图片或txt名，并输出
import os

def find_missing_files(folder_path):
    #获取文件夹中的所有文件名
    file_names = os.listdir(folder_path)
    #获取文件夹中的所有txt文件名
    txt_names = [name for name in file_names if name.endswith('.txt')]
    #去除txt文件名中的后缀
    txt_names = [name[:-4] for name in txt_names]
    #获取文件夹中的所有图片文件名
    img_names = [name for name in file_names if name.endswith('.jpg') or name.endswith('.png')]
    #去除图片文件名中的后缀
    img_names = [name[:-4] for name in img_names]
    #获取txt文件名和图片文件名的差集
    missing_names = list(set(txt_names) ^ set(img_names))
    #输出缺少的文件名，输出为
    print('Missing files:', missing_names)

#调用函数
folder_path = 'D:/dn/boset/set'
find_missing_files(folder_path)