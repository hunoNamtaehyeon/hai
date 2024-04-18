import re
from argparse import ArgumentParser

import nltk
import pandas as pd

nltk.download('punkt')  # Download the tokenizer data


class MakeBaseDictionary:
    def __init__(self, args):
        self.df, self.df1_len = self.df_loader(args.load_path)
        self.colname_whole = [col for col in self.df.columns if col.startswith('자기소개서')]
        if args.columns == 'full':
            self.colname = self.colname_whole
        else:
            self.input_col = [item.strip() for item in args.columns.split(',')]
            is_included = all(item in self.colname_whole for item in self.input_col)
            self.colname = self.input_col
            if not is_included:
                raise Exception("데이터에 존재하는 컬럼명만을 입력해주세요.")
        # df 리딩 - 전체 컬럼 파싱 - 인풋 컬럼 파싱 -
        if 'load_path2' in args and args.load_path2.lower() != 'none':
            # if args.load_path2.lower()!='none': #second path 존재한다면?
            self.df2, self.df2_len = self.df_loader(args.load_path2)
            self.df = pd.concat([self.df, self.df2], axis=0).reset_index(drop=True)
        print(len(self.df), self.colname)

    def preprocessing(self):
        print('start preprocessing')
        for col in self.colname_whole:
            self.df[col] = self.df[col].apply(lambda row: self.prepro_ser(row))

    def df_loader(self, load_path):
        formats = load_path.split('.')[-1]
        if formats == 'xlsx':
            loaded_df = pd.read_excel(load_path, engine='openpyxl').fillna(' ').astype(str)
            return loaded_df, len(loaded_df)
        elif formats == 'xls':
            loaded_df = pd.read_excel(load_path).fillna(' ').astype(str)
            return loaded_df, len(loaded_df)

    def prepro_ser(self, row):
        row = row.rstrip()
        if len(row) != 0:
            if row[-1] != '.':
                row = row + '.'
        else:
            row = row + '.'
        return row

    def tokenize_sentences(self, document):
        try:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', document)
        except:
            sentences = ['nan']
        return sentences

    def make_base_dict(self):
        user_num = len(self.df)
        capture_dict = {}
        for user in range(user_num):
            capture_dict[user] = {}
            for col_num, col in enumerate(self.colname):  # 인풋컬럼 넣어야함
                capture_dict[user][col_num] = {}
                document = self.df.loc[user, self.colname[col_num]]
                sents = self.tokenize_sentences(document)
                for sent_num, sent in enumerate(sents):
                    capture_dict[user][col_num][sent_num] = sent
        return capture_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="../data/killer.xlsx")
    parser.add_argument("--load_path2", type=str, default="../data/killer.xlsx")
    parser.add_argument("--save_path", type=str, default="./")
    # parser.add_argument("--columns", type=str, default="자기소개서3,자기소개서4")
    parser.add_argument("--columns", type=str, default="full")
    args = parser.parse_args()

    dict_class = MakeBaseDictionary(args)
    dict_class.preprocessing()
    base_dict = dict_class.make_base_dict()
    df = dict_class.df
    df1_len = dict_class.df1_len
