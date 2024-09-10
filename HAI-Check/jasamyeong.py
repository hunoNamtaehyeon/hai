import datetime
import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from soynlp.hangle import jamo_levenshtein
from tqdm import tqdm

from make_base_dict import MakeBaseDictionary
from icecream import ic

def overlap_checking(word, comname):
    # word = word1.replace(' ', '')
    word = word.replace(comname, '#%#')
    rep_count = word.count('#%#')
    word = " " * len(comname) + word
    comname_leng = len(comname)
    word_leng = len(word)
    count_lst = []
    for i in range(word_leng - comname_leng + 1):
        cnt = 0
        for let_a, let_b in zip(word[i:i + comname_leng], comname):
            if let_a == let_b:
                cnt += 1
        count_lst.append(cnt)
    # print(count_lst)
    max_args = np.argmax(count_lst)
    result = word[max_args:max_args + comname_leng]
    if len(result) > len(word):
        return word, max(count_lst)
    return result, max(count_lst)


def find_word_indices(sent1, word1):
    indices = []
    index = sent1.find(word1)
    while index != -1:
        indices.append(index + 1)
        index = sent1.find(word1, index + 1)
    return indices


def main(args, base_dict):
    with open('./jasa_dict_0910_test.json', 'w') as f:
        json.dump(base_dict, f, ensure_ascii=False, indent=4)
    ##### 과업 내용 출력
    print("#"*50)
    print("자사명 오기재 검출 중 : jasamyeong.py")
    print("#"*50)
    #####
    
    tokenizer = Okt()
    output_dict = {}
    comname = args.company_name
    com_leng = len(comname)
    max_leven = jamo_levenshtein(' ' * com_leng, comname)
    for user, rows in tqdm(base_dict.items()):
        if user not in output_dict:
            output_dict[user] = {}
        for col, docs in rows.items():
            if col not in output_dict[user]:
                output_dict[user][col] = {}
            final_lst = []
            for sent_ind, sent in docs.items():
                mini_dict = {'sent_ind': sent_ind + 1, 'sent': sent}
                ov_ck = overlap_checking(sent, comname)
                leven_ov = jamo_levenshtein(ov_ck[0], comname)

                if leven_ov < 0.7:
                    test_word = ''.join([item[0] for item in tokenizer.pos(ov_ck[0]) if item[1] != 'Josa'])
                    # print(user, col, sent_ind, '|', f'[{test_word}] | {sent}')
                    mini_dict['word_ind'] = find_word_indices(sent, test_word)
                    mini_dict['key_word'] = test_word
                    final_lst.append(mini_dict)
                elif com_leng >= 4 and (ov_ck[1] == com_leng - 2 and ' ' not in ov_ck[0] and leven_ov < 1):
                    test_word = ''.join([item[0] for item in tokenizer.pos(ov_ck[0]) if item[1] != 'Josa'])
                    # print(user, col, sent_ind, '|', f'[{test_word}] | {sent}')
                    mini_dict['word_ind'] = find_word_indices(sent, test_word)
                    mini_dict['key_word'] = test_word
                    final_lst.append(mini_dict)
                elif com_leng >= 6 and ov_ck[1] == com_leng - 3 and ' ' in ov_ck[0]:
                    new_keys = max(ov_ck[0].split(' '), key=len)
                    jamo_ov_ck = overlap_checking(new_keys, comname)
                    leven_jamo = jamo_levenshtein(new_keys, comname)
                    if jamo_ov_ck[1] >= com_leng - 3 and leven_jamo < max_leven // 2:
                        # print(user, col, sent_ind, '|', f'[{new_keys}] | {sent}')
                        mini_dict['word_ind'] = find_word_indices(sent, new_keys)
                        mini_dict['key_word'] = new_keys
                        final_lst.append(mini_dict)
                if final_lst != []:
                    output_dict[user][col] = final_lst
                else:
                    output_dict[user][col] = 0
    print(output_dict)
    return output_dict


def dict_transform(output_dict):
    print('Dict Transforming...')
    transformed_data = {}
    for key, inner_dict in tqdm(output_dict.items()):
        inner_data = {}
        for inner_key, values_dict in inner_dict.items():
            inner_data[f'자사명 오기재 검출{inner_key + 1}'] = json.dumps(values_dict, ensure_ascii=False)
        transformed_data[key] = inner_data
    countdf = pd.DataFrame.from_dict(transformed_data, orient='index')
    
    ##### "자사명 오기재 검출n" -> "검출n"
    ic(args.colname)
    countdf.columns = ["검출" + i[-1] for i in args.colname]
    #####
    
    return countdf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/자기소개서_1111_20230630030038.xls")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--columns", type=str, default='자기소개서1')
    parser.add_argument("--company_name", type=str, default="한국토지주택공사")
    args = parser.parse_args()

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    
    ##### 분석용 컬럼 받아놓기 (다른 모든 과업.py에는 아래 코드가 있는데 여기에만 없었음.)
    args.colname = dict_class.colname
    #####
    
    df = dict_class.df
    output_dict = main(args, base_dict)
    finaldf = dict_transform(output_dict)
    
    ##### 분석용 컬럼 -> 원래 컬럼으로 변환
    df.columns = dict_class.origin_colname
    #####
    
    jasadf = pd.concat([df, finaldf], axis=1)
    filename = '_'.join(args.load_path.split('_')[-2:]).split('.')[0]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    # jasadf.to_csv(os.path.join(args.save_path, 'wrong_jasa_' + formatted_datetime + '.csv'),
    #               encoding='utf-8',
    #                 # encoding='cp949',
    #               errors='ignore',
    #               index=None)
    jasadf.to_excel(os.path.join(args.save_path, 'wrong_jasa_' + formatted_datetime + '.xlsx'),
                #   encoding='utf-8',
                #     # encoding='cp949',
                #   errors='ignore',
                  index=None)

##############################################################################
# import datetime
# import json
# import os
# from argparse import ArgumentParser

# import numpy as np
# import pandas as pd
# from konlpy.tag import Okt
# from soynlp.hangle import jamo_levenshtein
# from tqdm import tqdm

# from make_base_dict import MakeBaseDictionary


# def overlap_checking(word, comname):
#     # word = word1.replace(' ', '')
#     word = word.replace(comname, '#%#')
#     rep_count = word.count('#%#')
#     word = " " * len(comname) + word
#     comname_leng = len(comname)
#     word_leng = len(word)
#     count_lst = []
#     for i in range(word_leng - comname_leng + 1):
#         cnt = 0
#         for let_a, let_b in zip(word[i:i + comname_leng], comname):
#             if let_a == let_b:
#                 cnt += 1
#         count_lst.append(cnt)
#     # print(count_lst)
#     max_args = np.argmax(count_lst)
#     result = word[max_args:max_args + comname_leng]
#     if len(result) > len(word):
#         return word, max(count_lst)
#     return result, max(count_lst)


# def find_word_indices(sent1, word1):
#     indices = []
#     index = sent1.find(word1)
#     while index != -1:
#         indices.append(index + 1)
#         index = sent1.find(word1, index + 1)
#     return indices


# def main(args, base_dict):
#     tokenizer = Okt()
#     output_dict = {}
#     comname = args.company_name
#     com_leng = len(comname)
#     max_leven = jamo_levenshtein(' ' * com_leng, comname)
#     for user, rows in tqdm(base_dict.items()):
#         if user not in output_dict:
#             output_dict[user] = {}
#         for col, docs in rows.items():
#             if col not in output_dict[user]:
#                 output_dict[user][col] = {}
#             final_lst = []
#             for sent_ind, sent in docs.items():
#                 mini_dict = {'sent_ind': sent_ind + 1, 'sent': sent}
#                 ov_ck = overlap_checking(sent, comname)
#                 leven_ov = jamo_levenshtein(ov_ck[0], comname)

#                 if leven_ov < 0.7:
#                     test_word = ''.join([item[0] for item in tokenizer.pos(ov_ck[0]) if item[1] != 'Josa'])
#                     # print(user, col, sent_ind, '|', f'[{test_word}] | {sent}')
#                     mini_dict['word_ind'] = find_word_indices(sent, test_word)
#                     mini_dict['key_word'] = test_word
#                     final_lst.append(mini_dict)
#                 elif com_leng >= 4 and (ov_ck[1] == com_leng - 2 and ' ' not in ov_ck[0] and leven_ov < 1):
#                     test_word = ''.join([item[0] for item in tokenizer.pos(ov_ck[0]) if item[1] != 'Josa'])
#                     # print(user, col, sent_ind, '|', f'[{test_word}] | {sent}')
#                     mini_dict['word_ind'] = find_word_indices(sent, test_word)
#                     mini_dict['key_word'] = test_word
#                     final_lst.append(mini_dict)
#                 elif com_leng >= 6 and ov_ck[1] == com_leng - 3 and ' ' in ov_ck[0]:
#                     new_keys = max(ov_ck[0].split(' '), key=len)
#                     jamo_ov_ck = overlap_checking(new_keys, comname)
#                     leven_jamo = jamo_levenshtein(new_keys, comname)
#                     if jamo_ov_ck[1] >= com_leng - 3 and leven_jamo < max_leven // 2:
#                         # print(user, col, sent_ind, '|', f'[{new_keys}] | {sent}')
#                         mini_dict['word_ind'] = find_word_indices(sent, new_keys)
#                         mini_dict['key_word'] = new_keys
#                         final_lst.append(mini_dict)
#                 if final_lst != []:
#                     output_dict[user][col] = final_lst
#                 else:
#                     output_dict[user][col] = 0
#     return output_dict


# def dict_transform(output_dict):
#     print('Dict Transforming...')
#     transformed_data = {}
#     for key, inner_dict in tqdm(output_dict.items()):
#         inner_data = {}
#         for inner_key, values_dict in inner_dict.items():
#             inner_data[f'자사명 오기재 검출{inner_key + 1}'] = json.dumps(values_dict, ensure_ascii=False)
#         transformed_data[key] = inner_data
#     countdf = pd.DataFrame.from_dict(transformed_data, orient='index')
#     return countdf


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--load_path", type=str, default="../data/자기소개서_1111_20230630030038.xls")
#     parser.add_argument("--save_path", type=str, default="./outputs")
#     parser.add_argument("--columns", type=str, default='자기소개서1')
#     parser.add_argument("--company_name", type=str, default="한국토지주택공사")
#     args = parser.parse_args()

#     dict_class = MakeBaseDictionary(args)
#     dict_class.preprocessing()
#     base_dict = dict_class.make_base_dict()
#     df = dict_class.df
#     output_dict = main(args, base_dict)
#     finaldf = dict_transform(output_dict)
#     jasadf = pd.concat([df, finaldf], axis=1)
#     filename = '_'.join(args.load_path.split('_')[-2:]).split('.')[0]
#     current_datetime = datetime.datetime.now()
#     formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
#     os.makedirs(args.save_path, exist_ok=True)
#     jasadf.to_csv(os.path.join(args.save_path, 'wrong_jasa_' + formatted_datetime + '.csv'),
#                   encoding='utf-8',
#                   errors='ignore',
#                   index=None)




