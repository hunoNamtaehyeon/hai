import datetime
import json
import multiprocessing
import os
from argparse import ArgumentParser
from itertools import combinations, permutations
from time import time
import copy

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from make_base_dict_test import MakeBaseDictionary

from pprint import pprint


def create_vocab_dict(base_dict, min_counts, min_word_len): ##### min_counts번 이상 나온 단어들 체크
                            
    ##### 과업 내용 출력
    print("#"*50)
    print("어절 표절 full 검출 중 : pyojeol_full.py")
    print("#"*50)
    #####
    
    vocab_dict = {}
    for u, data in tqdm(base_dict.items()):
        for col, docs in data.items():
            for si, sent in docs.items():
                words = nltk.word_tokenize(sent)
                for word in words:
                    if word not in vocab_dict:
                        vocab_dict[word] = {'count': 0, 'index': []}
                    vocab_dict[word]['count'] += 1
                    # sent_index는 @를 포함한 인덱스임. 골뱅이 삭제시 인덱스 당겨져서 안맞는다.
                    vocab_dict[word]['index'].append((u, col, si))
    sorted_vocab_dict = {word: data for word, data in sorted(vocab_dict.items(), key=lambda item: item[1]['count'])}
    filtered_vocab_dict = {word: data for word, data in sorted_vocab_dict.items() if
                           (len(word) >= min_word_len) & (min_counts <= data['count'])}
    cnt_lst = [v['count'] for k, v in filtered_vocab_dict.items()]
    return filtered_vocab_dict, cnt_lst


def compare_sentences(windows1, windows2):
    if set(windows1).intersection(set(windows2)):
        return True
    return False


def split_into_slides(sentence, slide_length):
    # Split the sentence into slides of given length
    words = sentence.split()
    slides = []
    for i in range(len(words) - slide_length + 1):
        slide = ' '.join(words[i:i + slide_length])
        slides.append(slide)
    return slides


def compare2sent(sent1, sent2):
    words1 = split_into_slides(sent1, args.slide_length)
    words2 = split_into_slides(sent2, args.slide_length)

    common_lines = [w1 for w1 in words1 if w1 in words2]
    if len(common_lines) == 0:
        ##### return False, '', 0
        ##### return 값 추가
        return False, '', 0, [], []
        #####
    else:
        sum_line, result = find_longest_sentence(common_lines)
        ##### return True, sum_line, len(sum_line.split(' '))
        ##### return 값 추가
        return True, sum_line, len(sum_line.split(' ')), result, [len(i.split(' ')) for i in result]
        #####


def find_longest_sentence(sentences):
    result = []
    current_sentence = ""
    for sentence in sentences:
        words = sentence.split()
        if not current_sentence:
            current_sentence = sentence
        elif current_sentence.split()[-1] == words[-2]:
            current_sentence = current_sentence + " " + words[-1]
        else:
            result.append(current_sentence)
            current_sentence = sentence
    if current_sentence:
        result.append(current_sentence)
    longest_sentence = max(result, key=lambda x: (len(x.split(' ')), len(x)))
    
    ##### return longest_sentence
    ##### 짧은 어절 적재
    while longest_sentence in result:
        result.remove(longest_sentence)
        
    result_c = result.copy()

    for r in result_c:
        if r in longest_sentence:
            while r in result:
                result.remove(r)
    # if longest_sentence == "관찰을 통한 고객의 특성을 기록하여 유형별로 분류해두는 습관은 늘 상대방보다":
    # print(longest_sentence, result)
    return longest_sentence, result
    #####            
    


def process_data(datas):
    result_dict = {}
    short_dict = {}
    # for pair1, pair2 in combinations(datas['index'], 2):
    user_ind1, doc_ind1, sent_ind1 = datas[0]
    user_ind2, doc_ind2, sent_ind2 = datas[1]
    sentence1 = base_dict[user_ind1][doc_ind1][sent_ind1]
    sentence2 = base_dict[user_ind2][doc_ind2][sent_ind2]
    # 결국 두 개의 문장이 서로 겹치는지 아닌지를 보고 싶은 행위임.
    dup_condition = user_ind1 != user_ind2 and user_ind1 < args.df1_len  # ((user_ind1, doc_ind1) != (user_ind2, doc_ind2)) & (user_ind1!=user_ind2)
    if dup_condition:  # True인 경우에만 비교 시작
        compare_bool, join_sent, inter_leng, short_sents, short_leng = compare2sent(sentence1, sentence2)
        if compare_bool:
            if user_ind1 not in result_dict:
                result_dict[user_ind1] = {}
            if doc_ind1 not in result_dict[user_ind1]:
                result_dict[user_ind1][doc_ind1] = {}
            if sent_ind1 not in result_dict[user_ind1][doc_ind1]:
                result_dict[user_ind1][doc_ind1][sent_ind1] = {}
            result_dict[user_ind1][doc_ind1][sent_ind1]['user_ind'] = user_ind2
            result_dict[user_ind1][doc_ind1][sent_ind1]['doc_ind'] = doc_ind2
            result_dict[user_ind1][doc_ind1][sent_ind1]['sent_ind'] = sent_ind2
            result_dict[user_ind1][doc_ind1][sent_ind1]['common_sent'] = join_sent
            result_dict[user_ind1][doc_ind1][sent_ind1]['common_eojeol_cnt'] = inter_leng
            
            ##### 짧은 어절 부분 추가
            result_dict[user_ind1][doc_ind1][sent_ind1]['short_common_sent'] = short_sents
            result_dict[user_ind1][doc_ind1][sent_ind1]['short_common_eojeol_cnt'] = short_leng
            #####
            
            # if join_sent == "관찰을 통한 고객의 특성을 기록하여 유형별로 분류해두는 습관은 늘 상대방보다":
            #     if (user_ind1, doc_ind1, sent_ind1) == (0,0,2):
            #         print(user_ind1, doc_ind1, sent_ind1)
            #         print(join_sent, short_sents)

            # if len(result_dict[user_ind1][doc_ind1][sent_ind1]['short_common_sent']) > 1:
            #     print(result_dict[user_ind1][doc_ind1][sent_ind1])
    return result_dict


def make_result_parallel(vocab_lsts, base_dict, num_processes=20):
    print('=' * 20)
    print("make_result_parallel START!")
    print('=' * 20)
    fillist = []
    shortlist = []
    # num_processes = 20
    is_progress = True

    # datas = [data for word, data in vocab_dict.items()] # 여기에서 셀프인지 아닌지를 판단할 수 있으면 좋을 것 같은데...
    # datas = detoxing_self(vocab_dict)
    pool = multiprocessing.Pool(num_processes)
    results = list(tqdm(pool.imap(process_data, vocab_lsts), total=len(vocab_lsts)))

    pool.close()
    pool.join()
    
    filtered_list = [item for item in results if item is not None]
    ##### for item in tqdm(filtered_list):
    #####     for k, v in item.items():
    #####         if {k: v} not in fillist:
    #####             fillist.append({k: v})
    ##### 코드 간소화 -> 속도 향상 많이 됨
    fillist = [i for i in tqdm(filtered_list) if i !={}]
    #####
    combined_dict = {}
    for sub_dict in tqdm(fillist):
        for key, value in sub_dict.items():
            if key not in combined_dict:
                combined_dict[key] = {}
            for inner_key, inner_value in value.items():
                if inner_key not in combined_dict[key]:
                    combined_dict[key][inner_key] = {}
                ##### combined_dict[key][inner_key].update(inner_value)
                ##### 이미 작성된 검출 내용을 새로작성할지
                # if key == 0 & inner_key == 0:
                #     print(inner_value)
                for a,b in inner_value.items():
                    try:
                        #!@#!@#!@ 0321 짧은 어절 수정사항
                        # 두 검출 딕셔너리 비교시 업데이트 기준 1.긴어절 2.짧은어절 / 검출딕셔너리(긴어절:10, 짧은어절:0)[승] vs 검출딕셔너리(긴어절:8, 짧은어절:5)[패]
                        # 근데 아마 위와 같은 상황은 아예없거나 있어도 완전완전 극소수임
                        if b['common_eojeol_cnt'] > combined_dict[key][inner_key][a]['common_eojeol_cnt']:
                            print("기준 검출!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            combined_dict[key][inner_key].update(inner_value)
                        elif b['common_eojeol_cnt'] == combined_dict[key][inner_key][a]['common_eojeol_cnt']:
                            tmp_dict = copy.deepcopy(b)
                            if sum(tmp_dict['short_common_eojeol_cnt']) >= sum(combined_dict[key][inner_key][a]['short_common_eojeol_cnt']):
                                combined_dict[key][inner_key].update(inner_value)
                        #!@#!@#!@
                    except:
                        combined_dict[key][inner_key].update(inner_value)
                #####

    return combined_dict


##### make_seconds_output_dict 함수는 그냥 싹 다 복붙하시면 됩니다.
###########################################################################
def make_seconds_output_dict(output_dict):
    output_dict2 = {}
    for u, rows in tqdm(output_dict.items()):
        for col, val in rows.items():
            for si, pyo_val in val.items():
                if u not in output_dict2:
                    output_dict2[u] = {}
                if col not in output_dict2[u]:
                    output_dict2[u][col] = {}
                if si not in output_dict2[u][col]:
                    output_dict2[u][col][si] = {}
    
                    target_ui = pyo_val['user_ind']
                    target_di = pyo_val['doc_ind']
                    target_si = pyo_val['sent_ind']
                    output_dict2[u][col][si]['user_ind'] = target_ui
                    output_dict2[u][col][si]['doc_ind'] = target_di
                    output_dict2[u][col][si]['sent_ind'] = target_si
                    output_dict2[u][col][si]['common_sent'] = pyo_val['common_sent']
                    output_dict2[u][col][si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                    output_dict2[u][col][si]['short_common_sent'] = pyo_val['short_common_sent']
                    output_dict2[u][col][si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']

                    if target_ui not in output_dict2:
                        output_dict2[target_ui] = {}
                    if target_di not in output_dict2[target_ui]:
                        output_dict2[target_ui][target_di] = {}
                    if target_si not in output_dict2[target_ui][target_di]:
                        output_dict2[target_ui][target_di][target_si] = {}
                        output_dict2[target_ui][target_di][target_si]['user_ind'] = u
                        output_dict2[target_ui][target_di][target_si]['doc_ind'] = col
                        output_dict2[target_ui][target_di][target_si]['sent_ind'] = si
                        output_dict2[target_ui][target_di][target_si]['common_sent'] = pyo_val['common_sent']
                        output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                        output_dict2[target_ui][target_di][target_si]['short_common_sent'] = pyo_val['short_common_sent']
                        output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                        
                    if pyo_val['common_eojeol_cnt'] > output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt']:
                        output_dict2[target_ui][target_di][target_si]['user_ind'] = u
                        output_dict2[target_ui][target_di][target_si]['doc_ind'] = col
                        output_dict2[target_ui][target_di][target_si]['sent_ind'] = si
                        output_dict2[target_ui][target_di][target_si]['common_sent'] = pyo_val['common_sent']
                        output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                        output_dict2[target_ui][target_di][target_si]['short_common_sent'] = pyo_val['short_common_sent']
                        output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                        
                    elif pyo_val['common_eojeol_cnt'] == output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt']:
                        if sum(pyo_val['short_common_eojeol_cnt']) > sum(output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt']):
                            output_dict2[target_ui][target_di][target_si]['user_ind'] = u
                            output_dict2[target_ui][target_di][target_si]['doc_ind'] = col
                            output_dict2[target_ui][target_di][target_si]['sent_ind'] = si
                            output_dict2[target_ui][target_di][target_si]['common_sent'] = pyo_val['common_sent']
                            output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                            output_dict2[target_ui][target_di][target_si]['short_common_sent'] = pyo_val['short_common_sent']
                            output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                
                    
                else:  # 문장번호까지 모두 같은경우 더 길게 뽑힌 어절 출력
                    if pyo_val['common_eojeol_cnt'] > output_dict2[u][col][si]['common_eojeol_cnt']:
                        target_ui = pyo_val['user_ind']
                        target_di = pyo_val['doc_ind']
                        target_si = pyo_val['sent_ind']
                        output_dict2[u][col][si]['user_ind'] = target_ui
                        output_dict2[u][col][si]['doc_ind'] = target_di
                        output_dict2[u][col][si]['sent_ind'] = target_si
                        output_dict2[u][col][si]['common_sent'] = pyo_val['common_sent']
                        output_dict2[u][col][si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                        output_dict2[u][col][si]['short_common_sent'] = pyo_val['short_common_sent']
                        output_dict2[u][col][si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                        if target_ui not in output_dict2:
                            output_dict2[target_ui] = {}
                        if target_di not in output_dict2[target_ui]:
                            output_dict2[target_ui][target_di] = {}
                        if target_si not in output_dict2[target_ui][target_di]:
                            output_dict2[target_ui][target_di][target_si] = {}

                            output_dict2[target_ui][target_di][target_si]['user_ind'] = u
                            output_dict2[target_ui][target_di][target_si]['doc_ind'] = col
                            output_dict2[target_ui][target_di][target_si]['sent_ind'] = si
                            output_dict2[target_ui][target_di][target_si]['common_sent'] = pyo_val['common_sent']
                            output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                            output_dict2[target_ui][target_di][target_si]['short_common_sent'] = pyo_val['short_common_sent']
                            output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                            
                        if pyo_val['common_eojeol_cnt'] > output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt']:
                            output_dict2[target_ui][target_di][target_si]['user_ind'] = u
                            output_dict2[target_ui][target_di][target_si]['doc_ind'] = col
                            output_dict2[target_ui][target_di][target_si]['sent_ind'] = si
                            output_dict2[target_ui][target_di][target_si]['common_sent'] = pyo_val['common_sent']
                            output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                            output_dict2[target_ui][target_di][target_si]['short_common_sent'] = pyo_val['short_common_sent']
                            output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                        elif pyo_val['common_eojeol_cnt'] == output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt']:
                            if sum(pyo_val['short_common_eojeol_cnt']) > sum(output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt']):
                                output_dict2[target_ui][target_di][target_si]['user_ind'] = u
                                output_dict2[target_ui][target_di][target_si]['doc_ind'] = col
                                output_dict2[target_ui][target_di][target_si]['sent_ind'] = si
                                output_dict2[target_ui][target_di][target_si]['common_sent'] = pyo_val['common_sent']
                                output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                                output_dict2[target_ui][target_di][target_si]['short_common_sent'] = pyo_val['short_common_sent']
                                output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                                
                    elif pyo_val['common_eojeol_cnt'] == output_dict2[u][col][si]['common_eojeol_cnt']:        
                        if sum(pyo_val['short_common_eojeol_cnt']) > sum(output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt']):
                            target_ui = pyo_val['user_ind']
                            target_di = pyo_val['doc_ind']
                            target_si = pyo_val['sent_ind']
                            output_dict2[u][col][si]['user_ind'] = target_ui
                            output_dict2[u][col][si]['doc_ind'] = target_di
                            output_dict2[u][col][si]['sent_ind'] = target_si
                            output_dict2[u][col][si]['common_sent'] = pyo_val['common_sent']
                            output_dict2[u][col][si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                            output_dict2[u][col][si]['short_common_sent'] = pyo_val['short_common_sent']
                            output_dict2[u][col][si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                            if target_ui not in output_dict2:
                                output_dict2[target_ui] = {}
                            if target_di not in output_dict2[target_ui]:
                                output_dict2[target_ui][target_di] = {}
                            if target_si not in output_dict2[target_ui][target_di]:
                                output_dict2[target_ui][target_di][target_si] = {}
                                
                                output_dict2[target_ui][target_di][target_si]['user_ind'] = u
                                output_dict2[target_ui][target_di][target_si]['doc_ind'] = col
                                output_dict2[target_ui][target_di][target_si]['sent_ind'] = si
                                output_dict2[target_ui][target_di][target_si]['common_sent'] = pyo_val['common_sent']
                                output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                                output_dict2[target_ui][target_di][target_si]['short_common_sent'] = pyo_val['short_common_sent']
                                output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                                
                            if pyo_val['common_eojeol_cnt'] > output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt']:
                                output_dict2[target_ui][target_di][target_si]['user_ind'] = u
                                output_dict2[target_ui][target_di][target_si]['doc_ind'] = col
                                output_dict2[target_ui][target_di][target_si]['sent_ind'] = si
                                output_dict2[target_ui][target_di][target_si]['common_sent'] = pyo_val['common_sent']
                                output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                                output_dict2[target_ui][target_di][target_si]['short_common_sent'] = pyo_val['short_common_sent']
                                output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
                            elif pyo_val['common_eojeol_cnt'] == output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt']:
                                if sum(pyo_val['short_common_eojeol_cnt']) > sum(output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt']):
                                    output_dict2[target_ui][target_di][target_si]['user_ind'] = u
                                    output_dict2[target_ui][target_di][target_si]['doc_ind'] = col
                                    output_dict2[target_ui][target_di][target_si]['sent_ind'] = si
                                    output_dict2[target_ui][target_di][target_si]['common_sent'] = pyo_val['common_sent']
                                    output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val['common_eojeol_cnt']
                                    output_dict2[target_ui][target_di][target_si]['short_common_sent'] = pyo_val['short_common_sent']
                                    output_dict2[target_ui][target_di][target_si]['short_common_eojeol_cnt'] = pyo_val['short_common_eojeol_cnt']
    # pprint(output_dict2)
    return output_dict2
###########################################################################
    
    
    
    
def make_rate_dict(updated_output_dict, base_dict):
    rate_dict = {}
    for u, rows in tqdm(base_dict.items()):
        if u not in rate_dict:
            rate_dict[u] = {"total_eojeol_cnt": 0, "total_pyojeol_eojeol_cnt": 0, 'docs_rates': {}}  # 모든 u 다 생성
        user_tot_sent_cnt = 0
        user_tot_pyojeol_sent_cnt = 0
        for col, docs in rows.items():
            tot_words_length = sum([len(sent.split()) for sent in base_dict[u][col].values()])
            if col not in rate_dict[u]['docs_rates']:
                rate_dict[u]['docs_rates'][col] = {}
            rate_dict[u]['docs_rates'][col]['total_sent_cnt'] = len(docs)  # len(docs)는 결국 문장의 갯수와 같음.
            if u in updated_output_dict and col in updated_output_dict[u]:
                ##### pyo_eojeol_cnt = sum([v['common_eojeol_cnt'] for v in
                #####                       updated_output_dict[u][col].values()])  # 해당 위치(u, col)에서 표절로 검출된 어절의 수
                ##### 긴 어절 수 + 짧은 어절 수
                pyo_eojeol_cnt1 = sum([v['common_eojeol_cnt'] for v in
                                      updated_output_dict[u][col].values()])  # 해당 위치(u, col)에서 표절로 검출된 어절의 수
                pyo_eojeol_cnt2 = sum([sum(v['short_common_eojeol_cnt']) for v in
                                      updated_output_dict[u][col].values()])  # 해당 위치(u, col)에서 표절로 검출된 어절의 수
                pyo_eojeol_cnt = pyo_eojeol_cnt1 + pyo_eojeol_cnt2
                #####
            else:
                pyo_eojeol_cnt = 0
                
            user_tot_pyojeol_sent_cnt += pyo_eojeol_cnt
            pyo_rates = pyo_eojeol_cnt / tot_words_length
            user_tot_sent_cnt += tot_words_length
            rate_dict[u]['docs_rates'][col] = round(pyo_rates, 4)
        rate_dict[u]['total_eojeol_cnt'] = user_tot_sent_cnt
        rate_dict[u]['total_pyojeol_eojeol_cnt'] = user_tot_pyojeol_sent_cnt
    return rate_dict



# 밑에 두 함수 안씀
# def user_change(u, df):
#     return df.loc[u, args.user_ind_colname]


# def exchange_dict(updated_output_dict):
#     coldict = {}
#     for i in range(len(dict_class.colname)):
#         coldict[i] = dict_class.colname[i]
#     newone = {}
#     for u, row in tqdm(updated_output_dict.items()):
#         # new_u = user_change(u, df)
#         if u not in newone:
#             newone[u] = {}
#         for col, docs in row.items():
#             new_col = coldict[col]
#             if new_col not in newone[u]:
#                 newone[u][new_col] = {}
#             for si, sent_val in docs.items():
#                 new_si = si + 1
#                 if new_si not in newone[u][new_col]:
#                     newone[u][new_col][new_si] = {}
#                 newone[u][new_col][new_si]['user_ind'] = user_change(sent_val['user_ind'], df)
#                 newone[u][new_col][new_si]['doc_ind'] = coldict[sent_val['doc_ind']]
#                 newone[u][new_col][new_si]['sent_ind'] = sent_val['sent_ind'] + 1
#                 newone[u][new_col][new_si]['common_sent'] = sent_val['common_sent']
#                 newone[u][new_col][new_si]['common_eojeol_cnt'] = sent_val['common_eojeol_cnt']
#     return newone

#@@@@@@@@@@@@@@@@@@
def reset_user_ind(updated_output_dict, origin_user_ind):
    for u, rows in tqdm(updated_output_dict.items()):
        for col, val in rows.items():
            for si, pyo_val in val.items():
                first_user_ind = pyo_val['user_ind']
                pyo_val['user_ind'] = origin_user_ind[first_user_ind]
    return updated_output_dict
#@@@@@@@@@@@@@@@@@@

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default='../data/killer.xlsx')
    parser.add_argument("--load_path2", type=str, default='../data/killer.xlsx')
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--columns", type=str, default='자기소개서1,자기소개서3,자기소개서4 ')
    parser.add_argument("--user_ind_colname", type=str, default='NO')
    parser.add_argument("--min_counts", type=int, default=2)
    parser.add_argument("--min_word_len", type=int, default=2)
    parser.add_argument("--slide_length", type=int, default=6)
    parser.add_argument("--percentile", type=int, default=80)
    parser.add_argument("--num_processes", type=int, default=40)
    args = parser.parse_args()
    start_time = time()
        
    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    args.df1_len = dict_class.df1_len
    
    ##### print("지원자 데이터 길이 : ", args.df1_len)
    df = dict_class.df
    filtered_vocab_dict, cnt_lst = create_vocab_dict(base_dict, args.min_counts, args.min_word_len)
    threshold = np.percentile(cnt_lst, args.percentile)
    vocab_dict = {word: data for word, data in filtered_vocab_dict.items() if data['count'] <= threshold}
    vocab_lsts = set()
    for word, data in tqdm(vocab_dict.items()):
        ##### for (p1, p2) in combinations(data['index'], 2):
        ##### 전체 조합
        # for (p1, p2) in combinations(data['index'], 2):
        for (p1, p2) in permutations(data['index'], 2):
        #####
            if p1[0] <= args.df1_len:
                vocab_lsts.add((p1, p2))
    result_dict = make_result_parallel(vocab_lsts, base_dict, num_processes=args.num_processes)
    updated_output_dict = make_seconds_output_dict(result_dict)
    
    #@@@@@@@@@@@@@@@@@@
    origin_user_ind = dict_class.origin_user_ind
    updated_output_dict = reset_user_ind(updated_output_dict, origin_user_ind)
    #@@@@@@@@@@@@@@@@@@
    
    ##### new_updated_output_dict = exchange_dict(updated_output_dict)
    # rate_dict_list = make_rate_dict(updated_output_dict, base_dict)
    rate_dict = make_rate_dict(updated_output_dict, base_dict)

    rating_df = pd.DataFrame.from_dict(rate_dict, orient='index')
    rating_df['tot_rates(%)'] = round(rating_df['total_pyojeol_eojeol_cnt'] / rating_df['total_eojeol_cnt'] * 100, 2)
    for i in range(len(dict_class.colname)):
        rating_df[f'Q{dict_class.colname[i][-1]}_rates(%)'] = rating_df['docs_rates'].apply(
            lambda x: round(x[i] * 100, 2))
    del rating_df['docs_rates']
    
    output_df = pd.DataFrame.from_dict(updated_output_dict, orient='index')
    output_df = output_df.reindex(df.index)
    output_df = output_df[sorted(output_df.columns)]
    
    ##### output_df.columns = [f'검출정보_{dict_class.colname[i][-1]}' for i in sorted(output_df.columns) if type(i) == int]
    ##### "검출정보_n" -> "검출n"
    output_df.columns = [f'검출{dict_class.colname[i][-1]}' for i in sorted(output_df.columns) if type(i) == int]
    #####
    
    output_df = output_df.fillna(0)
    for i in output_df.columns:
        output_df[i] = output_df[i].apply(lambda x: json.dumps(x, ensure_ascii=False))
    
    ##### 분석용 컬럼 -> 원래 컬럼으로 변환
    df.columns = dict_class.origin_colname
    #####
    
    finalDF = pd.concat([df, rating_df, output_df], axis=1)[:args.df1_len]
    filename = '_'.join(args.load_path.split('_')[-2:]).split('.')[0]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    formatted_datetime += f"_{args.slide_length}어절"
    #####
    print("#"*80)
    print(formatted_datetime)
    print("#"*80)
    #####
    end_time = time()
    print("@"*80)
    print(end_time - start_time)
    print("@"*80)
    os.makedirs(args.save_path, exist_ok=True)
    finalDF.to_csv(os.path.join(args.save_path, f'new_pyojeol_full_' + formatted_datetime + '.csv'),
                   encoding='cp949',
                #    encoding='utf-8',
                   errors='ignore', index=None)