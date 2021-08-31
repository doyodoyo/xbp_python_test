import collections
import itertools
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.cm as cm

from scipy import sparse
import math


def get_count_df(doc):
    """
    ginzaのdocを受け取って、1文ごとに共起語の組み合わせをカウントする
    """

    sentences = list(doc.sents)
    sentence_combinations = [list(itertools.combinations(sentence, 2)) for sentence in sentences]

    # listをflatにする
    tc = []
    for sentence in sentences:
        tc.extend(sentence)

    # (word, pos)の組み合わせでカウント
    tc_set = [(t.text, t.pos_) for t in tc]

    # 出現回数
    ct = collections.Counter(tc_set)
    # ct.most_common()[:10]

    #  単語の組合せと出現回数のデータフレームを作る
    word_combines = []
    for key, value in ct.items():
        word_combines.append([key[0], key[1], value])

    df = pd.DataFrame([{
        'word': i[0][0], 'count': i[1], 'word_pos': i[0][1]
    } for i in ct.most_common()])

    return df


def plot_word_dist(df_word, topk, title="出現頻度", color="darkcyan"):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=180)
    fp = FontProperties(fname='./fonts/ipaexg.ttf', size=8)
    cmap = cm.get_cmap('Set3')
    color = 2

    left = np.array(list(range(topk)))
    count = df_word['count'][:topk]
    plt.bar(left, count, color=cmap(color))

    ax.legend()
    ax.set_xticks(left)
    ax.set_xticklabels(df_word['word'][:topk], rotation=290, fontproperties=fp)
    ax.set_ylabel("回数", fontproperties=fp)
    ax.set_title(title, fontproperties=fp)
    ax.tick_params(labelsize=6)
    ax.legend(prop={'size': 6, "family": "IPAexGothic"})
    fig.subplots_adjust(bottom=0.20)
    plt.show()
    # fig.savefig("./result/frequency.png")
    # plt.clf()
    # plt.close()


def save_word_dist(df_word, file_name, topk, title="出現頻度", color="darkcyan"):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=180)
    fp = FontProperties(fname='./fonts/ipaexg.ttf', size=8)
    cmap = cm.get_cmap('Set3')
    color = 2

    left = np.array(list(range(topk)))
    count = df_word['count'][:topk]
    plt.bar(left, count, color=cmap(color))

    ax.legend()
    ax.set_xticks(left)
    ax.set_xticklabels(df_word['word'][:topk], rotation=290, fontproperties=fp)
    ax.set_ylabel("回数", fontproperties=fp)
    ax.set_title(title, fontproperties=fp)
    ax.tick_params(labelsize=6)
    ax.legend(prop={'size': 6, "family": "IPAexGothic"})
    fig.subplots_adjust(bottom=0.20)
    fig.savefig("./result/frequency_{}.png".format(file_name))
    plt.clf()
    plt.close()


def get_co_df(doc):
    """
    ginzaのdocを受け取って、1文ごとに共起語の組み合わせをカウントする
    """

    sentences = list(doc.sents)
    # 各文の2-gramの組み合わせ
    # sentence_combination_kouho = []
    # for sentence in sentences:
    #     for s in sentence:
    #         if not s.is_stop:
    #             sentence_combination_kouho.append(s)
    # sentence_combinations = [list(itertools.combinations(sentence_combination_kouho, 2))]
    sentence_combinations = [list(itertools.combinations(sentence, 2)) for sentence in sentences]

    # listをflatにする
    tc = []
    for sentence in sentence_combinations:
        tc.extend(sentence)

    # (word, pos)の組み合わせで共起語をカウント
    tc_set = [((t[0].text, t[0].pos_), (t[1].text, t[1].pos_)) for t in tc]

    # 出現回数
    ct = collections.Counter(tc_set)
    # ct.most_common()[:10]

    # sparce matrix
    # {単語, インデックス}の辞書作成
    tc_set_0 = [(t[0].text, t[0].pos_) for t in tc]
    tc_set_1 = [(t[1].text, t[1].pos_) for t in tc]

    ct_0 = collections.Counter(tc_set_0)
    ct_1 = collections.Counter(tc_set_1)

    dict_index_ct_0 = collections.OrderedDict((key[0], i) for i, key in enumerate(ct_0.keys()))
    dict_index_ct_1 = collections.OrderedDict((key[0], i) for i, key in enumerate(ct_1.keys()))
    dict_index_ct = collections.OrderedDict((key[0], i) for i, key in enumerate(ct.keys()))
    # print(dict_index_ct_0)

    #  単語の組合せと出現回数のデータフレームを作る
    word_combines = []
    for key, value in ct.items():
        word_combines.append([key[0][0], key[1][1], value, key[0][1], key[1][1]])

    df = pd.DataFrame([{
        'word1': i[0][0][0], 'word2': i[0][1][0], 'count': i[1]
        , 'word1_pos': i[0][0][1], 'word2_pos': i[0][1][1]
    } for i in ct.most_common()])

    return df


def get_cmap(df: pd.DataFrame):
    """
    Args:
      df(dataframe): 'word1', 'word2', 'count', 'word1_pos', 'word2_pos'

    Returns:
      {'ADP': 1, ...}

    """
    # 単語のposを抽出 indexで結合
    df_word_pos = pd.merge(pd.melt(df, id_vars=[], value_vars=['word1', 'word2'], value_name='word')
                           , pd.melt(df, id_vars=[], value_vars=['word1_pos', 'word2_pos'], value_name='pos')
                           , right_index=True, left_index=True).drop_duplicates(subset=['word', 'pos'])[['word', 'pos']]

    # posごとに色を付けたい
    cmap = set(df_word_pos['pos'].tolist())
    cmap = {k: v for v, k in enumerate(cmap)}

    return df_word_pos, cmap


def get_co_word(df: pd.DataFrame, word: str):
    """
    Args:
        df(pd.DataFrame):

    Returns:
        df_ex_co_word: 関連する単語のみを抽出する

    """

    # 特定のwordのみ抽出
    df_word = pd.concat([df[df['word1'] == word], df[df['word2'] == word]])

    # 単語のposを抽出 indexで結合
    df_word_pos = pd.merge(pd.melt(df_word, id_vars=[], value_vars=['word1', 'word2'], value_name='word')
                           , pd.melt(df_word, id_vars=[], value_vars=['word1_pos', 'word2_pos'], value_name='pos')
                           , right_index=True, left_index=True).drop_duplicates(subset=['word', 'pos'])[['word', 'pos']]

    # 特定の単語と関連する単語群の繋がり関係のみ抽出
    # 関連ワードがword1 or word2にある行を抽出
    df_ex_co_word = df[df[['word1', 'word2']].isin(list(df_word_pos['word'])).any(axis=1)]

    return df_ex_co_word


def get_network(df, edge_threshold=20):
    """
    df
    'word1', 'word2', 'count', 'word1_pos', 'word2_pos'
    """

    df_net = df.copy()

    # networkの定義
    nodes = list(set(df_net['word1'].tolist() + df_net['word2'].tolist()))

    graph = nx.Graph()
    #  頂点の追加
    graph.add_nodes_from(nodes)

    #  辺の追加
    #  edge_thresholdで枝の重みの下限を定めている
    df_graph = pd.DataFrame(columns=['word1', 'word2', 'count'])
    for i in range(len(df_net)):
        row = df_net.iloc[i]
        if row['word2_pos'] != "SPACE":
            if row['word1'] != row['word2']:
                if row['count'] >= edge_threshold:
                    add_row = pd.Series([row['word1'], row['word2'], row['count']], index=['word1', 'word2', 'count'])
                    df_graph.append(add_row, ignore_index=True)
                    graph.add_edge(row['word1'], row['word2'], weight=row['count'])

    # 孤立したnodeを削除
    isolated = [n for n in graph.nodes if len([i for i in nx.all_neighbors(graph, n)]) == 0]
    graph.remove_nodes_from(isolated)

    return graph


def plot_draw_networkx(df, word=None, figsize=(8, 8), edge_threshold=20, k=0.7):
    """
    wordを指定していれば、wordとそれにつながるnodeを描画する
    """
    G = get_network(df, edge_threshold)

    plt.figure(figsize=figsize)

    # k = node間反発係数 weightが太いほど近い
    # k = 2.0 / math.sqrt(len(G.nodes()))
    pos = nx.spring_layout(G, k=k)
    pr = nx.pagerank(G)

    pr_values = np.array([pr[node] for node in G.nodes()])

    # nodeの大きさ
    # posごとに色を付けたい
    df_word_pos, c = get_cmap(df)

    cname = ['aquamarine', 'navy', 'tomato', 'yellow', 'yellowgreen',
             'lightblue', 'limegreen', 'gold',
             'red', 'lightseagreen', 'lime', 'olive', 'gray',
             'purple', 'brown' 'pink', 'orange']

    # cnameで指定する。品詞と数値の対応から、nodeの単語の色が突合できる
    cmap_all = [cname[c.get(df_word_pos[df_word_pos['word'] == node]['pos'].values[0])] for node in G.nodes()]

    # 出力する単語とつながりのある単語のみ抽出、描画
    words = []
    edges = []
    if word is not None:
        df_word = pd.concat([df[df['word1'] == word], df[df['word2'] == word]])

        words = list(pd.merge(pd.melt(df_word, id_vars=[], value_vars=['word1', 'word2'], value_name='word')
                              , pd.melt(df_word, id_vars=[], value_vars=['word1_pos', 'word2_pos'], value_name='pos')
                              , right_index=True, left_index=True).drop_duplicates(subset=['word', 'pos'])[
                         ['word', 'pos']]['word'])

        edges = list(df_word[['word1', 'word2']].apply(tuple, axis=1))

    cmap = [cname[c.get(df_word_pos[df_word_pos['word'] == node]['pos'].values[0])] for node in words]

    nx.draw_networkx_nodes(G, pos
                           , node_color=cmap if word is not None else cmap_all
                           , cmap=plt.cm.Reds
                           , alpha=0.3
                           , node_size=pr_values * 30000
                           , nodelist=words if word is not None else G.nodes()  # 描画するnode
                           )
    # 日本語ラベル
    labels = {}
    for w in words:
        labels[w] = w
    nx.draw_networkx_labels(G, pos
                            , labels=labels if word is not None else None
                            , font_family='IPAexGothic'
                            , font_weight="normal"
                            )

    # 隣あう単語同士のweight

    edge_width = [G[edge[0]][edge[1]]['weight'] * 1.5 for edge in edges]
    nx.draw_networkx_edges(G, pos
                           , edgelist=edges if word is not None else G.edges()
                           , alpha=0.5
                           , edge_color="darkgrey"
                           , width=edge_width if word is not None else edge_width
                           )

    plt.axis('off')
    plt.show()



def save_draw_networkx(df, file_name, word=None, figsize=(8, 8), edge_threshold=20, k=0.7):
    """
    wordを指定していれば、wordとそれにつながるnodeを描画する
    """
    G = get_network(df, edge_threshold)

    plt.figure(figsize=figsize)

    # k = node間反発係数 weightが太いほど近い
    # k = 2.0 / math.sqrt(len(G.nodes()))
    pos = nx.spring_layout(G, k=k)
    pr = nx.pagerank(G)

    pr_values = np.array([pr[node] for node in G.nodes()])

    # nodeの大きさ
    # posごとに色を付けたい
    df_word_pos, c = get_cmap(df)

    cname = ['aquamarine', 'navy', 'tomato', 'yellow', 'yellowgreen',
             'lightblue', 'limegreen', 'gold',
             'red', 'lightseagreen', 'lime', 'olive', 'gray',
             'purple', 'brown' 'pink', 'orange']

    # cnameで指定する。品詞と数値の対応から、nodeの単語の色が突合できる
    cmap_all = [cname[c.get(df_word_pos[df_word_pos['word'] == node]['pos'].values[0])] for node in G.nodes()]

    # 出力する単語とつながりのある単語のみ抽出、描画
    words = []
    edges = []
    if word is not None:
        df_word = pd.concat([df[df['word1'] == word], df[df['word2'] == word]])

        words = list(pd.merge(pd.melt(df_word, id_vars=[], value_vars=['word1', 'word2'], value_name='word')
                              , pd.melt(df_word, id_vars=[], value_vars=['word1_pos', 'word2_pos'], value_name='pos')
                              , right_index=True, left_index=True).drop_duplicates(subset=['word', 'pos'])[
                         ['word', 'pos']]['word'])

        edges = list(df_word[['word1', 'word2']].apply(tuple, axis=1))

    cmap = [cname[c.get(df_word_pos[df_word_pos['word'] == node]['pos'].values[0])] for node in words]

    nx.draw_networkx_nodes(G, pos
                           , node_color=cmap if word is not None else cmap_all
                           , cmap=plt.cm.Reds
                           , alpha=0.3
                           , node_size=pr_values * 30000
                           , nodelist=words if word is not None else G.nodes()  # 描画するnode
                           )
    # 日本語ラベル
    labels = {}
    for w in words:
        labels[w] = w
    nx.draw_networkx_labels(G, pos
                            , labels=labels if word is not None else None
                            , font_family='IPAexGothic'
                            , font_weight="normal"
                            )

    # 隣あう単語同士のweight

    edge_width = [G[edge[0]][edge[1]]['weight'] * 1.5 for edge in edges]
    nx.draw_networkx_edges(G, pos
                           , edgelist=edges if word is not None else G.edges()
                           , alpha=0.5
                           , edge_color="darkgrey"
                           , width=edge_width if word is not None else edge_width
                           )

    plt.axis('off')
    plt.savefig("./result/conetwork_{}_{}_{}.png".format(file_name, edge_threshold, k))
