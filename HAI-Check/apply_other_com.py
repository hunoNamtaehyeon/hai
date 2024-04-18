import os
import datetime
import json
import os
from argparse import ArgumentParser

import pandas as pd
import torch
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, \
    AutoModelForSequenceClassification

from make_base_dict import MakeBaseDictionary


def no_corps_sent(sent, corps):
    for corp in corps:
        if corp in sent:
            return False
    return True


# def sent_in_corps(sent, corps, comdb):
#     for corp in corps:
#         if corp in sent:
#             return corp, True
#         else:
#             comlist = comdb['기업명'].values.tolist()
#             newcomlist = [x for x in comlist if x not in args.corps]
#             for newcom in newcomlist:
#                 if newcom in sent:
#                     return newcom, False


def is_tagiup(tokenizer, model, sent):
    inputs = tokenizer(sent, return_tensors='pt', truncation=True).to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs['logits']
    results = torch.softmax(logits, dim=-1)
    return int(torch.argmax(results)), float(max(results[0]))


def make_output_dict(tokenizer, model, base_dict, args):
    output_dict = {}
    comlist = args.comdb['기업명'].values.tolist()
    newcomlist = [x for x in comlist if x not in args.corps]
    for u, rows in tqdm(base_dict.items()):
        for col, docs in rows.items():
            for si, sent in docs.items():
                for newcom in newcomlist:
                    # 아래 "no_corps_sent(sent, args.corps):" 주석 해제 시 자기업, 타기업명 동시 등장 2가지 케이스 둘다 미검출
                    if newcom in sent:  # and no_corps_sent(sent, args.corps):
                        if newcom in args.banpair:
                            if no_corps_sent(sent, args.banpair[newcom]):
                                pred = is_tagiup(tokenizer, model, sent)  # 351
                                if pred[0] == 1:  # and pred[1]>0.9: #확실한 입사지원만
                                    if u not in output_dict:
                                        output_dict[u] = {}
                                    if f"검출{dict_class.colname[col][-1]}" not in output_dict[u]:
                                        output_dict[u][f"검출{dict_class.colname[col][-1]}"] = {}
                                    if si + 1 not in output_dict[u][f"검출{dict_class.colname[col][-1]}"]:
                                        output_dict[u][f"검출{dict_class.colname[col][-1]}"][si + 1] = {}
                                    output_dict[u][f"검출{dict_class.colname[col][-1]}"][si + 1] = json.dumps(
                                        {'타기업명': newcom, 'sent': sent}, ensure_ascii=False)
                        else:
                            pred = is_tagiup(tokenizer, model, sent)
                            if pred[0] == 1:  # and pred[1]>0.9: #확실한 입사지원만
                                if u not in output_dict:
                                    output_dict[u] = {}
                                if f"검출{dict_class.colname[col][-1]}" not in output_dict[u]:
                                    output_dict[u][f"검출{dict_class.colname[col][-1]}"] = {}
                                if si + 1 not in output_dict[u][f"검출{dict_class.colname[col][-1]}"]:
                                    output_dict[u][f"검출{dict_class.colname[col][-1]}"][si + 1] = {}
                                output_dict[u][f"검출{dict_class.colname[col][-1]}"][si + 1] = json.dumps(
                                    {'타기업명': newcom, 'sent': sent}, ensure_ascii=False)
    return output_dict


def infer(base_dict, args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    MODEL_NAME = "klue/roberta-large"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    model.classifier.out_proj.out_features = 4
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 4
    model.config = model_config
    model.load_state_dict(torch.load(args.best_ckpt))
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("START MAKING OUTPUT_DICT...")
    output_dict = make_output_dict(tokenizer, model, base_dict, args)
    return output_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/killer.xlsx")
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--best_ckpt", type=str, default="./models/other_com_best.pt")
    parser.add_argument("--comDB_path", type=str, default="./add_DB/other_com_comdb.xlsx")
    parser.add_argument("--banpairDB_path", type=str, default="./add_DB/other_com_banlist.xlsx")
    parser.add_argument("--columns", type=str, default='full')
    parser.add_argument("--corps_list", type=str, default='조폐공사,KOMSCO,한국조폐공사')
    args = parser.parse_args()
    args.corps = [item.strip() for item in args.corps_list.split(',')]

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df
    print(dict_class.colname)

    # ban 페어 리스트 로드
    banpair_df = pd.read_excel(args.banpairDB_path,
                               engine='openpyxl',
                               header=None,
                               names=['comname', 'banpair'])
    args.banpair = banpair_df.groupby('comname')['banpair'].apply(list).to_dict()

    # comdb 리스트 로드
    args.comdb = pd.read_excel(args.comDB_path, engine='openpyxl')[['기업명']]

    output_dict = infer(base_dict, args)
    output_df = pd.DataFrame.from_dict(output_dict, orient='index').reindex(df.index).fillna(0)

    final_df = pd.concat([df, output_df], axis=1)
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
    os.makedirs(args.save_path, exist_ok=True)
    final_df.to_csv(os.path.join(args.save_path, 'other_com_' + formatted_datetime + '.csv'),
                    encoding='utf-8',
                    errors='ignore',
                    index=None)
