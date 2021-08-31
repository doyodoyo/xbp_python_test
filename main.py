# コードについて
# https://blog.uni-3.app/2020/02/05/python-co-network
# spacy.loadのエラー解消について
# https://cool-and-tough.hatenablog.com/entry/2019/11/04/233802
# pip install ginza と管理者権限でPyCharmの実行を行った
# 下のURLによるとpip install -U ginza これだけでspacyもインストールされるみたい
# https://www.koi.mashykom.com/spacy_ginza.html
import os
import spacy
import pandas as pd
import utils

# dataset
f = open('myfile.txt', 'r')
data = f.read()
f.close()

df = pd.DataFrame({'文書': [data]})

# exec spacy
nlp = spacy.load('ja_ginza')
stop_words = nlp.Defaults.stop_words
for sw in ['こと', 'とき', 'もの', 'ところ', 'いう', 'お', 'しれ', '方', '〓']:
    stop_words.add(sw)

docs = [nlp(s) for s in df['文書']]

# 除外する品詞の指定
# ADJ: 形容詞, ADP: 設置詞, ADV: 副詞, AUX: 助動詞, CCONJ: 接続詞,
# DET: 限定詞, INTJ: 間投詞, NOUN: 名詞, NUM: 数詞,
# PART: 助詞, PRON: 代名詞, PROPN: 固有名詞, PUNCT: 句読点,
# SCONJ: 連結詞, SYM: シンボル, VERB: 動詞, X: その他
extract_pos = ['ADP', 'AUX', 'CCONJ', 'INTJ', 'NUM', 'PART', 'PRON', 'PUNCT', 'SCONJ', 'SYM', 'X', 'SPACE']

# 出現頻度取得
df_count = pd.concat([utils.get_count_df(d) for d in docs])

df_ex_count = df_count[(~df_count['word_pos'].isin(extract_pos))
                       & (~df_count['word'].isin(stop_words))]

utils.plot_word_dist(df_ex_count, topk=100)

# 共起語関係取得
df_co_word_count = pd.concat([utils.get_co_df(d) for d in docs]).reset_index(drop=True)

df_ex_word_count = df_co_word_count[(~df_co_word_count['word1_pos'].isin(extract_pos))
                                    & (~df_co_word_count['word2_pos'].isin(extract_pos))
                                    & (~df_co_word_count['word1'].isin(stop_words))
                                    & (~df_co_word_count['word2'].isin(stop_words))]

# 共起回数を全文書でまとめておく
df_net = df_ex_word_count.groupby(
    ['word1', 'word2', 'word1_pos', 'word2_pos']).sum()['count'].sort_values(
    ascending=False).reset_index()

utils.plot_draw_networkx(df_net, edge_threshold=10, k=0.5)
# utils.plot_draw_networkx(df_net, word='災害')
