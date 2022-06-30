import cv2
import copy
import numpy as np
import os
import random
import shutil
import albumentations as A

random.seed(133334)

name_map = {
    '待定': 'Others',
    '纤维': 'qianwei',
    '渣点 - 残胶 - 白胶': 'zhadian_canjiao_baise',
    '渣点 - 残胶 - 黑胶': 'zhadian_canjiao_heise',
    '渣点 - 残渣 - 焊装铁渣': 'zhadian_canzha_hanzhuangtiezha',
    '渣点 - 残渣 - 铰链': 'zhadian_canzha_jiaolian',
    '渣点 - 漆渣 - 电泳': 'zhadian_qizha_dianyong',
    '渣点 - 漆渣 - 清漆': 'zhadian_qizha_qingqi',
    '渣点 - 漆渣 - 色漆': 'zhadian_qizha_seqi',

}


Cameraparameters = {
    # CCD parameters
    "PixelFormat":  "Mono8",   # or YUV422Packed,  camera pixel format
    "Width": 1280,
    "Height": 960,
    # white balance
    "WhiteBalance_R":   128,   # WB Red
    "WhiteBalance_G":   128,   # WB Green
    "WhiteBalance_B":   128,   # WB Blue
    # capture image trigger
    "TriggerSelector": "FrameStart",
    "TriggerMode": "On",
    "TriggerSource": "Software",
    "TriggerDelayAbs":	1.5,   #/us  camera capture image with 1.5us delay after receive software trigger
    # image brightness setting
    "ExposureTimeRaw":  300 ,  # /us  abs exposure time
    "GainRaw": 850,  # used for camera gain setting
    "BlackLevelRaw": 1023,  # /the bigger the image brighter
    "GammaEnable": 0,
    "GammaSelector": "User",
    "Gamma": 1.2,  # the bigger the image brighter
}


class_map_from_Chanese_to_En = {

    '0_qianwei': ['纤维', '蓝纤',],
    '1_rongji': ['橙点','黄点' , '蓝点', '溶剂', ],
    '2_zhadian_canjiao_baise': ['残胶', '色漆'],
    '3_zhadian_canjiao_heise': ['黑胶', '涂胶', '底涂胶', '黑'],
    '4_zhadian_canzha_hanzhuangtiezha' : ['焊球', '焊渣', '铁', '铁屑', '色漆' ],
    '5_zhadian_qizha_dianyong': ['电泳', '电泳层', '电泳渣', ''],
    '6_zhadian_qizha_hongGanlu': ['烘干炉杂质', '滑翘渣', '炉灰', '炉渣', '炉渣杂质', '炉渣子杂质'],
    '7_zhadian_qizha_qingqi': ['黑点', '清漆', '漆渣', '异色漆片','清漆层'],
    '8_zhadian_qizha_seqi': ['漆渣', '色漆', '色漆层', '异色漆片', '黑点', '杂质'],
    '9_zhadian_canzha_jiaolian': [''],
    '10_zhadian_qizha_zhongtu': ['中涂漆渣', '漆渣','中涂层','色漆' ],
    '11_wuran': ['油污', ],
}




def create_raw_folder_from_path(src_path, dst_path):
    ''' used for create the same folder structure only with source path '''
    folders = os.listdir(src_path)
    for folder in folders:
        if not os.path.isdir(os.path.join(src_path,folder)):
            continue
        subfoler = os.path.join(dst_path,folder)
        if os.path.exists(subfoler):
            continue
        else:
            os.mkdir(subfoler)


def count_sample_distribution(path):
    '''
    path: the path used for count
    return1: and dict with folder name and count result
    return2: dict with folder name and files list
    '''
    assert os.path.exists(path)
    folders = os.listdir(path)
    statistic = { }
    statistic_with_filename = {}
    total = 0
    for folder in folders:
        prefix_with_folder = os.path.join(path,folder)
        if not os.path.isdir(prefix_with_folder):
            continue
        files = os.listdir(prefix_with_folder)
        statistic[folder] = len(files)
        statistic_with_filename[folder] = files
        total +=len(files)
    statistic['total'] = total
    return statistic, statistic_with_filename


def display_image(window_name = 'img_to_show', img = None):
    assert img
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def get_img_capture_date(img_full_path, byday = True, by_mintues = False):
    '''
    return: '2022:04:20 10:23:07'
    '''
    import exifread
    img = exifread.process_file(open(img_full_path, 'rb'))
    str_time = img['Image DateTime']

    if byday:
        day_minitues = str_time.values.split(' ')
        return day_minitues[0]
    if by_mintues:
        return str_time


def tta(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tta_transform = A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
    ])
    augmented_img = tta_transform(image = img)['image']
    img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
    return img

def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.MotionBlur(blur_limit=5, p=0.4),
        A.Blur(blur_limit=5, p=0.5),
        A.RandomGamma(p =0.5),
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
        ]),
        A.ColorJitter(p =0.3),
        A.GaussNoise(p=0.3),
        # A.RandomShadow(p = 0.2),
        A.RandomBrightnessContrast(p=0.5),
        A.ChannelShuffle(p=0.5),
        A.RandomRotate90(p=0.3),
        A.CLAHE(clip_limit=4,p=0.4),
        A.HueSaturationValue(p=0.3),
        A.RGBShift(p=0.3),

    ])

    augmented_img = transform(image=img)['image']
    img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
    # display_image(img)
    return img


def black_hole(folderpath ,output_folder_path = None, imgname = None, show=False, process_edge=True):
    # path = r'C:\Users\yi.ren5\Desktop\test_hole_recognize\2.png'
    path = folderpath + '\\' + imgname
    img = cv2.imread(path)
    if img is None:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img0 = copy.deepcopy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma = 2
    gray_equalize = gray / 255.0
    gray_gamma = np.power(gray_equalize, 1.5) * 255
    gray_new = gray_gamma.astype(np.uint8)
    if show:
        display_image('gamma', gray_new)


    # equalize = cv2.equalizeHist(gray)
    ret, thresh = cv2.threshold(gray_new, 50, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # sss = np.squeeze(np.max(contours,axis= 0))
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    cont = contours[0]

    cont_min = np.squeeze(np.min(cont, axis=0))
    cont_max = np.squeeze(np.max(cont, axis=0))
    # cv2.drawContours(img0,contours[i],-1,(0,0,255),2)
    # if imgname.split('.')[0] in ['8','10']:

    roi_h = np.abs(cont_min[1] - cont_max[1])
    roi_w = np.abs(cont_min[0] - cont_max[0])
    # hhh = min(roi_h,roi_w)
    width = max(roi_w, roi_h)
    finetune = False
    if finetune:
        if roi_w > roi_h:
            squeezed = np.squeeze(cont)
            result = np.where(squeezed == np.amin(squeezed))
            # point_left_min_indices = np.squeeze(np.where(cont == np.min(cont, axis=0)))
            index = list(result[0])[0]
            point_left = squeezed[index]
            radius = int(roi_w / 2)
            center_x = point_left[1]
            center_y = point_left[0] + radius
            y_min = center_x - radius
            y_max = center_x + radius

            x_min = center_y - radius
            x_max = center_y + radius

            roi_ = img0[y_min: y_max, x_min: x_max]
            if show:
                # cv2.rectangle(img0, ( center_y - radius, center_x - radius),(center_y + radius, center_x +radius),(0,0,255), 5)
                cv2.namedWindow('rectify', cv2.WINDOW_NORMAL)
                cv2.imshow('rectify', roi_)
                cv2.waitKey(0)

        else:
            roi_ = img0[cont_min[1]: cont_min[1] + width, cont_min[0]:cont_min[0] + width]
            # pass
    else:
        roi_ = img0[cont_min[1]: cont_min[1] + width, cont_min[0]:cont_min[0] + width]
    if show:
        display_image('cropped', roi_)


    img_output_name = imgname.split('.')[0] + '- roi.jpg'
    if output_folder_path is None:
        output_folder_path = folderpath
    output_fullpath = os.path.join(output_folder_path, img_output_name)
    if process_edge:
        # radius_w = cont_max[1] - cont_min[1]
        # radius_h = cont_max[0] - cont_min[0]
        # center_x = int(radius_w/2)
        # center_y = int(radius_h/2)
        center_x = center_y = int(width / 2)
        roi_ = circle_roi_generate(roi_, (center_x, center_y, int(width / 2)), 90, show=show)
        if show:
            cv2.namedWindow('img0', cv2.WINDOW_NORMAL)
            cv2.imshow('img0', roi_)
            cv2.waitKey(0)

        cv2.imencode('.jpg', roi_)[1].tofile(output_fullpath)
    else:
        cv2.imencode('.jpg', roi_)[1].tofile(output_fullpath)
    if show:
        cv2.rectangle(img0, (cont_min[0], cont_min[1]), (cont_max[0], cont_max[1]), (0, 0, 255), thickness=5)
        display_image('img_with_bounding_box', img0)
    # break

def circle_roi_generate(img, circle_para, radius_shift=1, show=False):
    "Generate circle ROI region"
    # dst = np.zeros(img.shape, np.uint8)
    radius_shift = int(radius_shift)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask = cv2.circle(mask, (circle_para[0], circle_para[1]), circle_para[2] - radius_shift, 255, cv2.FILLED)
    roi = cv2.bitwise_and(img, img, mask=mask)

    if show:
        display_image('roi', roi)

    roi = roi[radius_shift: roi.shape[1] - radius_shift, radius_shift: roi.shape[0] - radius_shift]
    # cv2.subtract(img, img,dst, mask)
    # img.copyto(dst, mask)
    if show:
        cv2.namedWindow('img0', cv2.WINDOW_NORMAL)
        cv2.imshow('img0', roi)
        cv2.waitKey(0)


    return roi


def dirty_from_painting(folderpath, imgname, show=False, method='circle', save_gray_img=False):
    '''

    '''
    path = folderpath + '\\' + imgname
    img = cv2.imread(path)
    if img is None:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    # img0 = copy.deepcopy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == 'circle':  # here is used houghcircles to find the circle
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            x = circles[0][0]
            y = circles[0][1]
            r = circles[0][2]

            if show:
                cv2.circle(img, (x, y), r, (0, 255, 0), thickness=4)
                cv2.namedWindow('cirle', cv2.WINDOW_NORMAL)
                cv2.imshow("cirle", img)
                cv2.waitKey(0)
            #
            r = r - 10  # here inner a little bit
            x_min = x - r
            x_max = x + r
            y_min = y - r
            y_max = y + r
            roi = img[y_min:y_max, x_min: x_max]

            if save_gray_img:
                suffix = '_gray_roi.jpg'
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                suffix = '_roi.jpg'

            if show:
                cv2.namedWindow('cirle', cv2.WINDOW_NORMAL)
                cv2.imshow("cirle", roi)
                cv2.waitKey(0)

            ''' circle roi '''
            img_output_name = imgname.split('.')[0] + suffix
            output_fullpath = os.path.join(folderpath,
                                           img_output_name)  # special used for the path with Chinese characters.
            cv2.imencode('.jpg', roi)[1].tofile(output_fullpath)
        else:
            return False
    else:
        ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_clo = cv2.dilate(thresh, kernel, iterations=2)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
        for item in stats:
            if show:
                cv2.rectangle(img, (item[0], item[1]), ((item[0] + item[2]), (item[1] + item[3])), color=(0, 0, 255),
                              thickness=3)
            if item[-1] < (img.shape[0] * img.shape[1]) / 2 and item[-1] > 5000:
                roi = img[item[1]: item[1] + item[3], item[0]:item[0] + item[2]]
                img_output_name = imgname.split('.')[0] + '-roi.jpg'
                output_fullpath = os.path.join(folderpath,
                                               img_output_name)  # special used for the path with Chinese characters.
                cv2.imencode('.jpg', roi)[1].tofile(output_fullpath)
                # cv2.imwrite(output_fullpath, roi)
                print(f'{output_fullpath} have writen out successfully!')
        if show:
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.waitKey(-1)

    return True


def preprocess_image(folder, output_path):
    files = os.listdir(folder)
    print('filelist:', files)
    # black_hole(folder, '黑胶（清漆层-汽车白色）.jpg', show=True, process_edge = True)
    class_folder = folder.split('\\')[-1]
    dst_folder = os.path.join(output_path, class_folder)
    for file in files:
        fullpath = os.path.join(folder, file)
        if os.path.isfile(fullpath):
            '''here used to skip the processed img'''
            img_processed = file.split('.')[0] + '- roi.jpg'
            if img_processed in files:
                continue
                # black_hole(folder, file, show= False)
            else:
                black_hole(folder,output_folder_path=  dst_folder, imgname= file, show=False, process_edge=True)
                continue
            # dirty_from_painting(folder, file,method= 'circle', save_gray_img= False)

        else:
            preprocess_image(fullpath, output_path)

def image_offline_aumentation(folder ,numbers_to_aug_per_img = 3):
    # folder = r'C:\Users\yi.ren5\Desktop\Desktop_files\湘潭\Sample\zhadian_qizha_seqi'
    files = os.listdir(folder)
    print('filelist:', files)
    # black_hole(folder, '黑胶（清漆层-汽车白色）.jpg', show=True, process_edge = True)
    for file in files:
        fullpath = os.path.join(folder, file)
        if os.path.isfile(fullpath):
            '''here used to skip the processed img'''
            img = cv2.imread(fullpath)
            if img is None:
                img = cv2.imdecode(np.fromfile(fullpath , dtype= np.uint8), -1)
            for i in range(numbers_to_aug_per_img):
                img_aug_name = file.split('.')[0] + f'- tta{i}.jpg'
                img_aug_fullpath = os.path.join(folder,img_aug_name)
                img_aug  = tta(img)
                if os.path.exists(img_aug_fullpath):
                    continue
                else:
                    cv2.imencode('.jpg', img_aug)[1].tofile(img_aug_fullpath)
        else:
            image_offline_aumentation(fullpath,numbers_to_aug_per_img)

def dataset_train_test_split(source_data_folder, out_put_folder,test_per, subclass_filter = 20):
    '''
    used for move sample data from the source images
    input folder
    '''
    train_folder = os.path.join(out_put_folder, 'Train')
    test_folder = os.path.join(out_put_folder,'Test')

    folders = os.listdir(source_data_folder)
    for fo in folders:
        if not os.path.isdir(os.path.join(source_data_folder,fo)):
            continue
        files = os.listdir(os.path.join(source_data_folder,fo))
        if len(files) < subclass_filter: # if subclass sample quantity is less than filer, ignore
            continue
        list_numbers = np.arange(0,len(files)).tolist()
        choosed_index = random.sample(list_numbers,int(test_per*len(files)))

        if not os.path.exists(os.path.join(train_folder, fo)): # check the train folder exist or not,
            os.mkdir(os.path.join(train_folder, fo))
        if not os.path.exists(os.path.join(test_folder, fo)): # check the test folder exist or not,
            os.mkdir(os.path.join(test_folder, fo))

        for index in list_numbers:
            if index in choosed_index:   # for  is test  data
                src_folder_path = os.path.join(source_data_folder, fo)
                dst_folder_path = os.path.join(test_folder, fo)
                shutil.move(os.path.join(src_folder_path, files[index]), os.path.join(dst_folder_path, files[index]))
            else:   # for train data
                src_folder_path = os.path.join(source_data_folder, fo)
                dst_folder_path = os.path.join(train_folder, fo)
                shutil.move(os.path.join(src_folder_path, files[index]), os.path.join(dst_folder_path, files[index]))

def get_singleimagelabel_byname(img_name, name_dict):
    # label = ''
    import jieba
    file_name =  jieba.cut(img_name)
    file_name_list = ','.join(file_name).split(',')
    pure_file_name = []
    for item in file_name_list:
        if not isChinese(item):
            continue
        pure_file_name.append(item)

    temp_dict_record = {}
    for key, value in name_dict.items():
        temp_dict_record[key] = len(set(pure_file_name)&set(value))
    label = max(temp_dict_record, key = lambda x : temp_dict_record[x])
    return label

def isChinese(word):
    if '\u4e00' <= word <= '\u9fff':
        return True
    else:
        return False

def add_classnamemap_to_dicttxt(class_map):
    import jieba
    added_word = []
    assert type(class_map) is dict
    for key, value in class_map.items():
        for item in value:
            jieba.add_word(item)
            added_word.append(item)
    print("added word to jieba dict: ", added_word)

def data_movement(src_path =None , dst_path = None):
    dst_path = r'C:\Users\yi.ren5\Desktop\新建文件夹\data_for_test'
    src_path = r'C:\Users\yi.ren5\Desktop\新建文件夹\test_roi'
    # create_raw_folder_from_path(src_path,dst_path = dst_path)
    add_classnamemap_to_dicttxt(class_map_from_Chanese_to_En)
    files = os.listdir(src_path)
    for file in files:
        label = get_singleimagelabel_byname(file, class_map_from_Chanese_to_En)
        dst_path_ = os.path.join(dst_path, label) + '\\' + file
        src_path_ = os.path.join(src_path, file)
        shutil.move(src_path_, dst_path_)


if __name__ == '__main__':
    src_path = r'D:\Projects\项目评估\漆面缺陷检测\Images\Total_image_448'
    train_data = r'D:\Projects\项目评估\漆面缺陷检测\Images\data_for_train\train'
    #step1 : Create ROI folder from source folder list
    src_path = r'D:\Projects\项目评估\漆面缺陷检测\Images\data_for_train\train'
    dst_path = r'D:\Projects\项目评估\漆面缺陷检测\Images\data_for_train\test'
    # ROI_path = r'D:\Projects\项目评估\漆面缺陷检测\Images\Total_image_448_roi'
    # data_for_train = r'D:\Projects\项目评估\漆面缺陷检测\Images\data_for_train'
    # create_raw_folder_from_path(src_path, dst_path)
    # step2: Cut roi from source image and save to ROI folder
    src_path = r'C:\Users\yi.ren5\Desktop\新建文件夹\test'
    dst_path = r'C:\Users\yi.ren5\Desktop\新建文件夹\test_roi'
    # preprocess_image(src_path, output_path= dst_path)
    #step3: Image offline aumentation

    image_offline_aumentation( folder= r"./timm/images/validation/" )
    #step4: Train test split
    # dataset_train_test_split(source_data_folder= ROI_path, out_put_folder= data_for_train,test_per=.2, subclass_filter= 20)
    # result, data = count_sample_distribution(train_data)
    # print(result)
    # data_movement()


