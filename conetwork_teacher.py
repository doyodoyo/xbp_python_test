import os
import glob
import codecs
import spacy
import pandas as pd
import utils


department_dirs = glob.glob("./text/teacher/*")

for department_dir in department_dirs:
    file_name = os.path.basename(department_dir)

    teacher_files = glob.glob("{}/*.txt".format(department_dir))
    text_data = ""
    for teacher_file in teacher_files:
        f = open(teacher_file, 'r', encoding="utf-8_sig")
        text_data += f.read()
        f.close()

    df = pd.DataFrame({'文書': [text_data]})

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

    utils.save_word_dist(df_ex_count, file_name, topk=30)

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

    utils.save_draw_networkx(df_net, file_name, edge_threshold=10, k=0.5)
