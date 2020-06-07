import os
import json
from PIL import Image
import csv
import codecs
import xlwt

__filelist = []


def _file_extension(file_name):
    return os.path.splitext(file_name)[1]


def _getAllfiles(Image_Path, file_extension_check=True):
    '''file_extension_check means to check the file is image or not'''
    image_extension = ['.bmp', '.jpg', '.jpeg', '.png', '.tiff']
    filenames = os.listdir(Image_Path)
    for file in filenames:
        fileabspath = os.path.join(Image_Path, file)
        if os.path.isdir(fileabspath):
            _getAllfiles(fileabspath)
        else:
            if _file_extension(file) in image_extension:
                __filelist.append(fileabspath)
    return __filelist


def FileNameExtract_To_txt(Ori_Path='', Saved_path_file_name=''):  # filename为写入CSV文件的路径，data为要写入数据列表.
    if Ori_Path == '':
        return
    data = _getAllfiles(Ori_Path)
    file = open(Saved_path_file_name, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    __filelist.clear()
    print(Saved_path_file_name + " has been saved successfully")


def FileNameExtract_To_csv(Ori_Path='', Saved_path_file_name=''):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    if Ori_Path == '':
        return
    datas = _getAllfiles(Ori_Path)
    file_csv = codecs.open(Saved_path_file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_NONE)
    for data in datas:
        writer.writerow(data)
    __filelist.clear()
    print(Saved_path_file_name + " has been saved successfully")


def FileNameExtract_To_xls(Ori_Path='', Saved_path_file_name=''):
    if Ori_Path == '':
        return
    datas = _getAllfiles(Ori_Path)
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

    i = 0
    for data in datas:
        sheet1.write(i, 0, data)  # 将数据写入第 i 行，第 j 列
        i = i + 1
    f.save(Saved_path_file_name)  # 保存文件
    __filelist.clear()
    print(Saved_path_file_name + " has been saved successfully")


if __name__ == '__main__':
    Path = R'C:\Users\ren5szh\Desktop\Station1Crop\\'
    if Path == '':
        pass
    else:

        FileNameExtract_To_txt(Saved_path_file_name=Path + 'files.txt', Ori_Path=Path)
        FileNameExtract_To_csv(Saved_path_file_name=Path + 'files.csv', Ori_Path=Path)
        FileNameExtract_To_xls(Saved_path_file_name=Path + 'files.xls', Ori_Path=Path)
