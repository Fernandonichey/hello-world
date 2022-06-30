import os
import csv
import numpy as np

class_to_idx = {
    '0': '0_qianwei',
    '1': '1_rongji',
    '2': '2_zhadian_canjiao_baise',
    '3': '3_zhadian_canjiao_heise',
    '4': '4_zhadian_canzha_hanzhuangtiezha',
    '5': '5_zhadian_qizha_dianyong',
    '6': '6_zhadian_qizha_hongGanlu',
    '7': '7_zhadian_qizha_qingqi',
    '8': '8_zhadian_qizha_seqi',
    '9': '9_zhadian_canzha_jiaolian',
    '10': '10_zhadian_qizha_zhongtu',
    '11': '11_wuran',
}


def check_pred_right(index):
    '''check top1 right or not'''
    if index[0] == index[5]:
        return True
    else:
        return False


def check_pred_top2(index):
    ''' check top 2 right or not'''
    if index[5] in index[:2]:
        return True
    else:
        return False


def check_top2_confused(index):
    ''' check top 2 right or not'''
    if index[5] == index[1]:
        return True
    else:
        return False


result = {}
tta_merger_para = {
    "mean": 0.5,
    "gmean": 0.0,
    "max": 1.0,
    "min": 0.0,
    "sum": 1.5,
    "tsharpen": 0.56903558, }


def tta_merger():
    return True


def tta_result_check(line):
    tta_file_name = line[0][-9:-5]
    pred_prob = list(map(eval, line[1:10]))
    temp = {}
    temp['GT'] = [(eval(line[10]))]
    temp['pred'] = pred_prob
    # for i in range(len(index)):
    #     temp[index[i]] = [eval(prob[i])]

    if tta_file_name == ' tta':
        file_name = line[0][:-10] + '.jpg'
    else:
        file_name = line[0]

    if file_name in result.keys():
        value = result[file_name]
        for k, v in temp.items():
            if k in value.keys():
                if k == 'GT':
                    value[k].append(v[0])
                else:
                    value[k].append(v)
            else:
                value[k] = v
            # value[k].append[v[0]]
        result[file_name] = value
    else:
        temp['pred'] = [temp['pred']]
        result[file_name] = temp


def tta_file_output():
    with open("output/tta/topk_tta.csv", 'r', encoding='utf-8')  as f:
        files = csv.reader(f)
        for line in files:
            tta_result_check(line)
    # finish the file parse
    with open("output/tta/tta_result.csv", "w", encoding= 'utf-8') as f:
        for k, v in result.items():
            temp = []
            temp.append(k)
            gt = list(set(v['GT']))[0]
            # temp.append(gt)
            pred = v['pred']
            pred_array = np.power(np.array(pred), tta_merger_para['tsharpen'])
            pred_mean = np.mean(pred_array, axis=0)
            pred_mean_list = pred_mean.tolist()

            pred_sort = sorted(pred_mean, reverse=True)

            pred_sort_index = [pred_mean_list.index(x) for x in pred_sort]
            # temp.append(pred_sort[0:5])
            f.write('{0},{1},{2},{3}\n'.format(
                k,','.join([str(v) for v in pred_sort_index[0:5]]),  str(gt),str(pred_sort[0:5])
            ))




if __name__ == '__main__':
    # tta_file_output()

    confused_label = {}
    total_number = 0
    top1_correct = 0
    top2_correct = 0
    metric_total = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, }
    metric_top1 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, }
    y_pred = []
    y_true = []
    with open("output/tta/tta_result.csv", 'r', encoding='utf-8')  as f:
        files = csv.reader(f)
        for line in files:
            file_name = line[0]
            # tta_result_check(line)
            index = line[1:]
            total_number += 1
            metric_total[index[5]] += 1
            y_pred.append(line[1])
            y_true.append(line[6])
            if check_pred_right(index):
                top1_correct += 1
                metric_top1[index[5]] += 1
            if check_top2_confused(index):
                confused_label[file_name] = (index[0:2] + list(index[5]))
            if check_pred_top2(index):
                top2_correct += 1


    ''' calculate the tta result '''
    # for kk1, vva1 in result.items():
    #     for key, value in vva1.items():
    #         if key == "GT":
    #             vva2 = value[0]
    #         else:
    #             vva2 = np.mean(np.power(value,0.56903558),axis=0)
    #         vva1[key] = vva2

    accurate_per_class = {}
    for k, v in metric_total.items():
        accurate_per_class[class_to_idx[k]] = metric_top1[k] / metric_total[k]

    print(f'test top1 correct rate: {top1_correct / total_number}')
    print(f'test top2  correct rate: {top2_correct / total_number}')
    print(f'confused index: {confused_label}')
    print(f'metric total: {metric_total} \n metric_top1 : {metric_top1}', )
    print(f'accurate per class in Lab: {accurate_per_class}')

    from sklearn.metrics import confusion_matrix, f1_score

    confusion_matrix_ = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(f'confusion matrix:\n {confusion_matrix_}')
    print(f'f1 score {f1_score(y_true, y_pred, average=None)}')
