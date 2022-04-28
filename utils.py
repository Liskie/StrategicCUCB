import contextlib
import json
import collections
import os
from typing import List, Union

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import time

def read_file(filename):
    '''
    :param filename:
    :return: file's json data
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def combine_all():
    filename_list = ['/data/private/lsd2018/programe/crowd-data/train.json',
                     '/data/private/lsd2018/programe/crowd-data/test-crowd.json',
                     '/data/private/lsd2018/programe/crowd-data/dev-crowd.json']
    total_list = []
    for filename in filename_list:
        data = read_file(filename)
        for item in data:
            total_list.append(item)
    # f = open('/data/private/lsd2018/programe/crowd-data/all-crowd.json','w')
    # f.write('[')
    # for i in range(len(total_list)-2):
    #     f.write(json.dumps(total_list[i],ensure_ascii=False)+',\n')
    # f.write(json.dumps(total_list[len(total_list)-1],ensure_ascii=False)+']')
    return total_list


def anno_work_count(data):
    '''
    统计每个标注者在所有任务中的标注个数
    '''
    count_list = []
    for item in data:
        temp_list = []
        for ann in item['annotations']:
            temp_list.append(ann['user'])
        temp_list = list(set(temp_list))
        for i in temp_list:
            count_list.append(i)
    ctr = collections.Counter(count_list)
    cc = sorted(ctr.items())
    print('The number of each worker participating in the labeling')
    print(cc)


def get_worker_list(data):
    worker_list = []
    for item in data:
        for ann in item['annotations']:
            worker_list.append(ann['user'])
    return list(set(worker_list))


def get_single_annotations(data, worker_id):
    '''
    获取单个worker的所有标注
    '''
    annotator_works = []
    for item in data:
        temp_dict = {}
        temp_dict['id'] = ''
        temp_dict['text'] = ''
        temp_dict['tags'] = ['O'] * len(item['text'])
        count = 0
        for ann in item['annotations']:
            if ann['user'] == worker_id:
                temp_dict['id'] = item['id']
                temp_dict['text'] = item['text']
                for count in range(len(item['text'])):
                    if count == ann['start_offset']:
                        temp_dict['tags'][count] = "B-" + ann['label']
                    elif ann['start_offset'] < count < ann['end_offset']:
                        temp_dict['tags'][count] = "I-" + ann['label']
                    else:
                        continue
        if temp_dict['id'] != '':
            annotator_works.append(temp_dict)
        else:
            continue
    # print(annotator_works)
    return annotator_works


def get_worker_list_annotations(data, worker_id_list):
    '''
    得到一个list的worker的所有标注
    '''
    total_list = []
    for worker in worker_id_list:
        temp_data = get_single_annotations(data, worker)
        # print(len(temp_data))
        for i in temp_data:
            total_list.append(i)
    print('\nThe worker in list is:', worker_id_list)
    print(f'\nThe workers in list annotate {len(total_list)} sentences')
    return total_list


def get_sliver(data):
    bio_list = []
    for item in data:
        new_dict = {}
        if bool(item['bestUsers']) != False:
            new_dict['id'] = item['id']
            new_dict['text'] = item['text']
            # new_dict['text'] = item['text']
            tags = ['O'] * len(item['text'])
            label = []
            for ann in item['annotations']:
                if ann['user'] == item['bestUsers'][0]:
                    # label = []
                    # start = 0
                    # end = 0
                    for j in range(len(item['text'])):
                        # print(j)
                        if j == ann['start_offset']:
                            tags[j] = "B-" + str(ann['label'])
                            # start = j
                        elif ann['start_offset'] < j < ann['end_offset']:
                            tags[j] = "I-" + str(ann['label'])
                            # end = j+1
                        else:
                            continue
                # label.append(item['text'][start:end])
            new_dict['tags'] = tags
            # new_dict['label_text'] = set(label)
            bio_list.append(new_dict)
        else:
            new_dict['id'] = item['id']
            new_dict['text'] = item['text']
            new_dict['tags'] = ['O'] * len(item['text'])
            bio_list.append(new_dict)
    return bio_list


def get_sliver_dict(data):
    bio_dict = {}
    for item in data:
        if bool(item['bestUsers']) != False:
            bio_dict[item['id']] = []
            tags = ['O'] * len(item['text'])
            for ann in item['annotations']:
                if ann['user'] == item['bestUsers'][0]:
                    # label = []
                    # start = 0
                    # end = 0
                    for j in range(len(item['text'])):
                        # print(j)
                        if j == ann['start_offset']:
                            tags[j] = "B-" + str(ann['label'])
                            # start = j
                        elif ann['start_offset'] < j < ann['end_offset']:
                            tags[j] = "I-" + str(ann['label'])
                            # end = j+1
                        else:
                            continue
                # label.append(item['text'][start:end])
            bio_dict[item['id']] = tags
        else:
            bio_dict[item['id']] = ['O'] * len(item['text'])
    return bio_dict


def get_f1(worker_data, sliver_data):
    y_true = []
    y_pred = []
    start = time.time()
    for pred_item in worker_data:
        y_pred.append(pred_item['tags'])
        for true_item in sliver_data:
            if true_item['id'] == pred_item['id']:
                y_true.append(true_item['tags'])
    print('-' * 60, '\n', classification_report(y_true, y_pred, digits=3), '-' * 60)
    end = time.time()
    print(f'run {end - start}s\n')
    return f1_score(y_true, y_pred, average='macro')


def get_f1_with_dict(worker_data, silver_dict):
    y_true = []
    y_pred = []
    start = time.time()
    for pred_item in worker_data:
        y_pred.append(pred_item['tags'])
        y_true.append(silver_dict[pred_item['id']])
    print('-' * 60, '\n', classification_report(y_true, y_pred, digits=3), '-' * 60)
    end = time.time()
    print(f'run {end - start}s\n')
    return f1_score(y_true, y_pred, average='macro')


data = read_file('all-crowd.json')


def get_worker_ids() -> List[int]:
    '''
    Return all worker ids present in the annotations.
    Returns:
        List of integer worker ids
    '''
    return get_worker_list(data)


def get_super_arm_f1_by_id(worker_id: Union[List[int], int]) -> float:
    '''
    Calculate the macro mean F-1 score of all arms(workers) in the super-arm.
    Args:
        worker_ids: List of worker ids
    Returns:
        Float value of F-1 score
    '''
    with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
        if isinstance(worker_id, list):
            worker_data = get_worker_list_annotations(data, worker_id)
        elif isinstance(worker_id, int):
            worker_data = get_worker_list_annotations(data, [worker_id])
        else:
            raise TypeError('Input is neither an id nor list of ids.')
        sliver_data_dict = get_sliver_dict(data)
        f1_score = get_f1_with_dict(worker_data, sliver_data_dict)
        return f1_score


if __name__ == "__main__":
    data = read_file('all-crowd.json')
    worker_list = [2, 3, 4, 5, 6, 7]
    anno_work_count(data)
    worker_data = get_worker_list_annotations(data, worker_list)

    # sliver data frame:list
    # sliver_data = get_sliver(data)
    # f1_score = get_f1(worker_data,sliver_data)
    # print(f1_score)

    # sliver data frame:dict
    sliver_data_dict = get_sliver_dict(data)
    f1_score = get_f1_with_dict(worker_data, sliver_data_dict)
    print(f1_score)
