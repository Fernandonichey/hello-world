from sklearn.model_selection import train_test_split
import os

__filelist = []
__file_extension_selection = ['.bmp']


def _classes_encoder(root_dir):
    classes_from_folder = {}
    classes = os.listdir(root_dir)
    with open('classes_ont_hot.txt', 'w')  as f:
        for index, class_name in enumerate(classes):
            classes_from_folder[class_name] = index
            out_string = class_name + ' ' + str(index) + '\n'
            f.write(out_string)

    return classes_from_folder


def _file_extension(file_name):
    return os.path.splitext(file_name)[1]


def _getAllfiles(Input_Path, label):
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
                    __filelist.append([fileabspath, label])
            else:
                __filelist.append([fileabspath, label])
    return __filelist


def list_all_images_label(root_dir):
    classes_names = _classes_encoder(root_dir)

    folders = os.listdir(root_dir)

    for folder in folders:
        full_path = os.path.join(root_dir, folder)
        _getAllfiles(full_path, classes_names[folder])

    return __filelist


def split_train_test_data_into_txt(root_dir, val_size=0.2, test_size=0.25):
    result_dict = list_all_images_label(root_dir)
    train, test = train_test_split(result_dict, test_size=test_size)

    train, val = train_test_split(train, test_size=val_size)

    with open('train.txt', 'w') as train_saver:
        for item in train:
            s = item[0] + ' ' + str(item[1]) + '\n'
            train_saver.write(s)
    with open('val.txt', 'w') as val_saver:
        for item in val:
            s = item[0] + ' ' + str(item[1]) + '\n'
            val_saver.write(s)

    with open('test.txt', 'w') as test_saver:
        for item in test:
            s = item[0] + ' ' + str(item[1]) + '\n'
            test_saver.write(s)

    print(f'trainging data size: {len(train)}')
    print(f'validation data size: {len(val)}')
    print(f'testing data size: {len(test)}')


if __name__ == '__main__':
    path = R"C:\Users\ren5szh\Desktop\Bonding"
    split_train_test_data_into_txt(path)
