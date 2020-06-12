import os
import csv
import codecs
import xlwt

__all__=["FileNameExtract_To_txt","FileNameExtract_To_csv","FileNameExtract_To_xls"]
__filelist = []
__file_extension_selection = []


def _file_extension(file_name):
    return os.path.splitext(file_name)[1]


def _getAllfiles(Input_Path):
    if not os.path.exists(Input_Path):
        print("Your input path" + Input_Path + "is not exist")
        return
    '''file_extension_check means to check the file is image or not'''
    filenames = os.listdir(Input_Path)
    for file in filenames:
        fileabspath = os.path.join(Input_Path, file)
        if os.path.isdir(fileabspath):
            _getAllfiles(fileabspath)
        else:
            if len(__file_extension_selection):
                if _file_extension(file) in __file_extension_selection:
                    __filelist.append(fileabspath)
            else:
                __filelist.append(fileabspath)
    return __filelist


def FileNameExtract_To_txt(Input_Path='', Saved_path_file_name='',
                           file_extension_selection=[]):  # filename为写入CSV文件的路径，data为要写入数据列表.
    for item in file_extension_selection:
        __file_extension_selection.append(item)
    if not os.path.exists(Input_Path):
        print("Your input path" + Input_Path + "is not exist")
        return
    data = _getAllfiles(Input_Path)
    file = open(Saved_path_file_name, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    __filelist.clear()
    __file_extension_selection.clear()
    print(Saved_path_file_name + " has been saved successfully")


def FileNameExtract_To_csv(Input_Path='', Saved_path_file_name='',
                           file_extension_selection=[]):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    for item in file_extension_selection:
        __file_extension_selection.append(item)
    if not os.path.exists(Input_Path):
        print("Your input path" + Input_Path + "is not exist")
        return
    datas = _getAllfiles(Input_Path)
    file_csv = codecs.open(Saved_path_file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv,delimiter=' ',quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(str(data))
    __filelist.clear()
    __file_extension_selection.clear()
    print(Saved_path_file_name + " has been saved successfully")


def FileNameExtract_To_xls(Input_Path='', Saved_path_file_name='', file_extension_selection=[]):
    for item in file_extension_selection:
        __file_extension_selection.append(item)
    if not os.path.exists(Input_Path):
        print("Your input path" + Input_Path + "is not exist")
        return
    datas = _getAllfiles(Input_Path)
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

    i = 0
    for data in datas:
        sheet1.write(i, 0, data)  # 将数据写入第 i 行，第 j 列
        i = i + 1
    f.save(Saved_path_file_name)  # 保存文件
    __filelist.clear()
    __file_extension_selection.clear()
    print(Saved_path_file_name + " has been saved successfully")


if __name__ == '__main__':
    Path =""  # os.getcwd()
    if Path:
        FileNameExtract_To_txt(Input_Path=Path, Saved_path_file_name=Path + 'files.txt',
                               file_extension_selection=['.py','.bmp'])
        FileNameExtract_To_csv(Input_Path=Path, Saved_path_file_name=Path + 'files.csv',
                               file_extension_selection=['.py','.bmp'])
        FileNameExtract_To_xls(Input_Path=Path, Saved_path_file_name=Path + 'files.xls',
                               file_extension_selection=['.py','.bmp'])
