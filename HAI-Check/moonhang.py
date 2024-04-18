import datetime
import os
from argparse import ArgumentParser
from difflib import SequenceMatcher

import nltk
import pandas as pd
from tqdm import tqdm
import json
from make_base_dict import MakeBaseDictionary


def cut_text_into_windows(long_text, window_size):
    sentences = long_text.split(". ")
    result = []

    for i in range(len(sentences) - window_size + 1):
        window = ". ".join(sentences[i:i + window_size])
        if len(window) != 0 and window[-1] != '.':
            window += '.'
        result.append(window)

    return result


def find_largest_value_in_columns(lst):
    num_rows = len(lst)
    num_columns = len(lst[0])
    largest_values = []

    for col in range(num_columns):
        max_value = float('-inf')
        max_pair = None

        for row in range(num_rows):
            current_pair = lst[row][col]
            current_value = current_pair[0]

            if current_value > max_value:
                max_value = current_value
                max_pair = current_pair

        largest_values.append(max_pair)

    return largest_values


def find_largest_value(whole_dict):
    fin_dict = {}
    for ui in whole_dict['Q0'].keys():
        lst = []
        for k, v in whole_dict.items():
            lst.append(v[ui])
        # print(ui, find_largest_value_in_columns(lst))
        fin_dict[ui] = find_largest_value_in_columns(lst)
    return fin_dict


def main(question, df):
    whole_dict = {}
    order = [i.strip('자기소개서') for i in dict_class.colname]

    for q in order:
        if q in question.keys():
            qval = question[q]
            query_dict = {}
            print(' ')
            print('=' * 50)
            print(qval)
            print('=' * 50)
            print(' ')
            for u in tqdm(df.index):
                query_dict[u] = []
                for col in dict_class.colname:
                    string = df.loc[u, col]
                    # q_sent_leng = len(dict_class.tokenize_sentences(qval))
                    if qval in string:
                        query_dict[u].append((100, string.count(qval), qval))
                    else:
                        max_ratio = 0
                        win_cnt = 0
                        cut_string = cut_text_into_windows(string, 1)
                        new_winind = 0
                        for winind, window in enumerate(cut_string):
                            ratio = SequenceMatcher(None, window, qval).ratio()
                            if len(window) / len(qval) > 0.7 and ratio >= 0.8:  # window in qval:
                                win_cnt += 1
                                if max_ratio <= ratio:
                                    max_ratio = ratio
                                    new_winind = winind
                        if win_cnt == 0:
                            query_dict[u].append((round(max_ratio * 100, 3), win_cnt, ''))
                        else:
                            query_dict[u].append((round(max_ratio * 100, 3), win_cnt, cut_string[new_winind]))
            whole_dict[f'Q{q}'] = query_dict
    return whole_dict


def combine_values(dictionary):
    combined_values = {}
    for key, sub_dict in dictionary.items():
        for sub_key, sublist in sub_dict.items():
            if sub_key not in combined_values:
                combined_values[sub_key] = []
            combined_values[sub_key].append(sublist)

    return combined_values


def make_output_dict(result):
    output = {}
    for i in range(len(dict_class.colname)):
        if i not in output:
            output[f'항목{i + 1} 검출'] = []
        for k, v in result.items():
            max_list = []
            sum_string = []
            for sublst in v:  # result.values():

                max_list.append(sublst[i][0])
                sum_string.append((sublst[i][0], sublst[i][2], sublst[i][1]))
            max_score = max(max_list)
            result1 = (max_score, sum_string)
            if max_score != 0:
                output[f'항목{i + 1} 검출'].append(result1)
            else:
                output[f'항목{i + 1} 검출'].append(0)
    return output


def trans_output(x):
    if x != 0:
        max_score, q_values_list = x
        q_values = {}
        for index, (rate, repeat_text, repeat_cnt) in enumerate(q_values_list, start=1):
            q_values[f'Q{index}'] = {
                'rate': rate,
                'repeat_text': repeat_text,
                'repeat_cnt': repeat_cnt
            }

        result = {
            'max_score': max_score,
            'Q_values': q_values
        }
        # print(result)
        return json.dumps(result, ensure_ascii=False)
    else:
        return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str,
                        default="../data/자기소개서_1028_20230624025957.xls")  # "../../data/sample.xlsx")
    # parser.add_argument("--load_path2", type=str, default='none')
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--columns", type=str, default='full')
    parser.add_argument("--question", type=dict, default={
        "1": '우리공사에 관심을 갖게 된 계기와 입사 후 이루고 싶은 목표는 무엇입니까? 본인이 제시한 목표를 달성하는 것이 공사의 미션 또는 비전 달성에 어떠한 형태로 기여할 수 있는지 기술해 주시기 바랍니다.',
        "2": '최근 5년 이내, 지원한 직무와 관련된 내용에 한하여 본인의 강점을 만들기 위해 노력한 경험을, 당시의 상황, 과정, 결과로 구분하여 구체적으로 기술해 주시기 바랍니다.',
        "3": "최근 5년 이내, 본인이 속한 조직에서 타인과의 의견차이로 인해 발생한 갈등을 원만하게 해결한 경험이 있다면, 갈등상황, 해결과정, 결과로 구분하여 구체적으로 기술해 주시기 바랍니다.",
        "4": "최근 5년 이내, 지원한 직무와 관련된 활동(학업, 동아리, 업무 등) 중 예상치 못한 문제가 발생하여 난감했던 상황을 효과적으로 극복한 사례가 있다면, 문제상황, 극복방법, 결과로 구분하여 구체적으로 기술해 주시기 바랍니다.",
        "5": "최근 5년 이내, 스스로 본인을 희생하면서 조직 구성원들의 이익을 위해 노력한 경험이 있다면, 구체적으로 기술해 주시기 바랍니다."})
    args = parser.parse_args()

    nltk.download('punkt')  # Download the tokenizer data
    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    whole_dict = main(args.question, dict_class.df)
    result_combine = combine_values(whole_dict)
    output_dict = make_output_dict(result_combine)
    outputDF = pd.DataFrame(output_dict)
    for i in range(len(dict_class.colname)):
        outputDF[f'항목{i + 1} 검출'] = outputDF[f'항목{i + 1} 검출'].apply(lambda x: trans_output(x))
    finaldf = pd.concat([dict_class.df, outputDF], axis=1)
    filename = '_'.join(args.load_path.split('_')[-2:]).split('.')[0]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    finaldf.to_csv(os.path.join(args.save_path, 'moonhang_' + formatted_datetime + '.csv'), encoding='utf-8',
                   errors='ignore')
