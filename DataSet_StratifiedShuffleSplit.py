from sklearn.model_selection import train_test_split
import os

__file_extension_selection = ['.bmp']


def _classes_encoder(root_dir):
    classes_from_folder = {}
    classes = os.listdir(root_dir)
    with open(os.path.join(root_dir, 'classes_ont_hot.txt'), 'w')  as f:
        for index, class_name in enumerate(classes):
            classes_from_folder[class_name] = index
            out_string = class_name + ' ' + str(index) + '\n'
            f.write(out_string)

    return classes_from_folder


def _file_extension(file_name):
    return os.path.splitext(file_name)[1]


def _getAllfiles(Input_Path, label, test_size=0.2, val_size=0.2):
    __filelist = []
    if not os.path.exists(Input_Path):
        print("Your input path" + Input_Path + "is not exist")
        return
    '''file_extension_check means to check the file is image or not'''
    filenames = os.listdir(Input_Path)
    for file in filenames:
        fileabspath = os.path.join(Input_Path, file)
        out_string = fileabspath + ' ' + str(label) + '\n'
        if os.path.isdir(fileabspath):
            _getAllfiles(fileabspath)
        else:
            if len(__file_extension_selection):
                if _file_extension(file) in __file_extension_selection:
                    __filelist.append(out_string)
            else:
                __filelist.append(out_string)
    train, test = train_test_split(__filelist, test_size=test_size)
    train, val = train_test_split(train, test_size=val_size)
    return train, val, test


def list_all_images_label(root_dir, test_size, val_size):
    classes_names = _classes_encoder(root_dir)
    folders = os.listdir(root_dir)

    __train_set = []
    __test_set = []
    __val_set = []

    for folder in folders:
        full_path = os.path.join(root_dir, folder)
        train, val, test = _getAllfiles(full_path, classes_names[folder], test_size=test_size, val_size=val_size)
        for item in train:
            __train_set.append(item)
        for item in val:
            __val_set.append(item)
        for item in test:
            __test_set.append(item)
    return __train_set, __val_set, __test_set


def split_train_test_data_into_txt(root_dir,  val_size=0.2, test_size=0.2):
    train, val, test = list_all_images_label(root_dir, test_size, val_size=val_size)

    with open(os.path.join(root_dir, 'train.txt'), 'w') as train_saver:
        for item in train:
            train_saver.write(item)
    with open(os.path.join(root_dir, 'val.txt'), 'w') as val_saver:
        for item in val:
            val_saver.write(item)
    with open(os.path.join(root_dir, 'test.txt'), 'w') as test_saver:
        for item in test:
            test_saver.write(item)

    print(f'trainging data size: {len(train)}')
    print(f'validation data size: {len(val)}')
    print(f'testing data size: {len(test)}')


if __name__ == '__main__':
    path = R"C:\Users\ren5szh\Desktop\Bonding"
    split_train_test_data_into_txt(path)
    # Stratify_split_train_test_into_txt(root_dir = path, n_splits=2)
