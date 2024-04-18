import datetime
import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from make_base_dict import MakeBaseDictionary


def split_into_slides(sentence, slide_length=6):
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

    # print(sum_line)
    if len(common_lines) == 0:
        return False, '', 0
    else:
        sum_line = find_longest_sentence(common_lines)
        return True, sum_line, len(sum_line.split(' '))


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
    longest_sentence = max(result, key=lambda x: len(x.split(' ')))
    return longest_sentence


def check_two_docs(user_num, i, j, tok_doc):
    reslist = []
    for ai, a in enumerate(tok_doc[i]):
        for bi, b in enumerate(tok_doc[j]):
            if compare2sent(a, b)[0]:
                # reslist.append((ai, bi, [compare2sent(a, b)[1], compare2sent(a, b)[2]]))
                reslist.append({i: {
                    ai: {'user_ind': user_num, 'doc_ind': j, 'sent_ind': bi, 'common_sent': compare2sent(a, b)[1],
                         'common_eojeol_cnt': compare2sent(a, b)[2]}}})
    return reslist


def make_result_dict(df):
    result_dict = {}
    for user_num in tqdm(range(len(df))):
        # user_num = 13953

        result_lst = []
        doc1 = df.loc[user_num][args.colname].values.tolist()
        tok_doc = [dict_class.tokenize_sentences(doc) for doc in doc1]
        for i in range(len(tok_doc) - 1):
            for j in range(i + 1, len(tok_doc)):
                # # 두 문서 비교
                res = check_two_docs(user_num, i, j, tok_doc)
                if res != []:
                    # pprint(res)
                    result_lst.extend(res)

        merged_dict = {}

        for divided_dict in result_lst:
            for key, sub_dict in divided_dict.items():
                for sub_key, value in sub_dict.items():
                    merged_key = (key, sub_key)
                    if merged_key in merged_dict:
                        if merged_dict[merged_key]['doc_ind'] == value['doc_ind']:
                            if merged_dict[merged_key]['common_eojeol_cnt'] < value['common_eojeol_cnt']:
                                merged_dict[merged_key] = value
                        else:
                            merged_dict[merged_key]['common_eojeol_cnt'] += value['common_eojeol_cnt']
                    else:
                        merged_dict[merged_key] = value

        merged_result = {}
        for key, value in merged_dict.items():
            main_key, sub_key = key
            if main_key in merged_result:
                merged_result[main_key][sub_key] = value
            else:
                merged_result[main_key] = {sub_key: value}
        if merged_result != {}:
            if user_num not in result_dict:
                result_dict[user_num] = merged_result
    return result_dict


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
                        output_dict2[target_ui][target_di][target_si]['common_eojeol_cnt'] = pyo_val[
                            'common_eojeol_cnt']
    return output_dict2


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
                pyo_eojeol_cnt = sum([v['common_eojeol_cnt'] for v in
                                      updated_output_dict[u][col].values()])  # 해당 위치(u, col)에서 표절로 검출된 어절의 수
            else:
                pyo_eojeol_cnt = 0
            user_tot_pyojeol_sent_cnt += pyo_eojeol_cnt
            pyo_rates = pyo_eojeol_cnt / tot_words_length
            user_tot_sent_cnt += tot_words_length
            rate_dict[u]['docs_rates'][col] = round(pyo_rates, 4)
        rate_dict[u]['total_eojeol_cnt'] = user_tot_sent_cnt
        rate_dict[u]['total_pyojeol_eojeol_cnt'] = user_tot_pyojeol_sent_cnt
    return rate_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default='../data/killer.xlsx')
    # parser.add_argument("--load_path2", type=str, default="None")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--columns", type=str, default="full")
    parser.add_argument("--slide_length", type=int, default=6)
    args = parser.parse_args()

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()  # input colname으로
    df = dict_class.df  # full df
    args.colname = dict_class.colname

    # base_dict는 인풋 컬럼 파싱으로 넣어야함.
    result_dict = make_result_dict(df)
    updated_output_dict = make_seconds_output_dict(result_dict)

    rate_dict = make_rate_dict(updated_output_dict, base_dict)

    rating_df = pd.DataFrame.from_dict(rate_dict, orient='index')
    rating_df['tot_rates(%)'] = round(rating_df['total_pyojeol_eojeol_cnt'] / rating_df['total_eojeol_cnt'] * 100, 2)
    for i in range(len(args.colname)):
        rating_df[f'Q{dict_class.colname[i][-1]}_rates(%)'] = rating_df['docs_rates'].apply(
            lambda x: round(x[i] * 100, 2))
    del rating_df['docs_rates']
    output_df = pd.DataFrame.from_dict(updated_output_dict, orient='index')
    output_df = output_df.reindex(df.index)
    output_df = output_df[sorted(output_df.columns)]
    output_df.columns = [f'검출정보_{dict_class.colname[i][-1]}' for i in sorted(output_df.columns) if type(i) == int]
    output_df = output_df.fillna(0)
    for i in output_df.columns:
        output_df[i] = output_df[i].apply(lambda x: json.dumps(x, ensure_ascii=False))
    finalDF = pd.concat([df, rating_df, output_df], axis=1)

    filename = '_'.join(args.load_path.split('_')[-2:]).split('.')[0]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    finalDF.to_csv(os.path.join(args.save_path, f'new_pyojeol_self_' + formatted_datetime + '.csv'),
                   encoding='utf-8',
                   errors='ignore', index=None)
