
import nlplot
import pandas as pd
from janome.tokenizer import Tokenizer

# figure
width = 800
height = 600

# dataset
f = open('./myfile.txt', 'r', encoding="utf-8")
data = f.read()
f.close()

df = pd.DataFrame({'text': data.split("。")})

def subtract_words(text):
    # tokenizer = Tokenizer()
    tokenizer = Tokenizer('./user_dict.csv', udic_enc='utf8')
    tokens = tokenizer.tokenize(text)
    
    #形態素解析した結果を格納するリスト
    wordlist = []
    
    for token in tokens:
        hinshi_00 = token.part_of_speech.split(',')[0]
        hinshi_01 = token.part_of_speech.split(',')[1]
        base_word = token.base_form
        
        if hinshi_01 != "非自立":
            if hinshi_00 == "名詞":
                if hinshi_01 != "数":
                    wordlist.append(base_word)
            elif hinshi_00 == "形容詞":
                wordlist.append(base_word)
            elif hinshi_00 == "動詞":
                wordlist.append(base_word)

    return wordlist

#形態素結果をリスト化し、データフレームdfに結果を列追加する
df['words'] = df['text'].apply(subtract_words)

npt = nlplot.NLPlot(df, target_col='words')

# top_nで指定する頻出上位の単語, min_freqで指定する頻出回数以下の単語を指定できる
stopwords = npt.get_stopword(top_n=2, min_freq=0)
stopwords += ["ある", "られる", "いく", "とき", "の"]

# https://boxcode.jp/nlplot%E3%81%8C%E5%87%84%E3%81%84%EF%BC%81%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E3%82%92%E5%8F%AF%E8%A6%96%E5%8C%96%E3%83%BB%E5%88%86%E6%9E%90%E3%81%A7%E3%81%8D%E3%82%8Bpython%E3%83%A9%E3%82%A4

# 頻出ランキンググラフ
npt.bar_ngram(
    title='uni-gram',
    xaxis_label='word_count',
    yaxis_label='word',
    ngram=1,
    top_n=20,
    stopwords=stopwords,
    width=width,
    height=height,
    save=True
)

# ツリーマップ
npt.treemap(
    title='Tree of Most Common Words',
    ngram=1,
    top_n=30,
    stopwords=stopwords,    
    width=width,
    height=height,
    save=True
)

# 単語数の分布
npt.word_distribution(
    title='number of words distribution',
    xaxis_label='count',
    width=width,
    height=height,
    save=True
)

# ワードクラウド
npt.wordcloud(
    max_words=100,
    max_font_size=100,
    colormap='tab20_r',
    stopwords=stopwords,
    width=width,
    height=height,
    save=True
)

npt.build_graph(stopwords=stopwords, min_edge_frequency=1)

# 共起ネットワーク
npt.co_network(
    title='Co-occurrence network', 
    width=width,
    height=height,
    save=True
)

# サンバースト
npt.sunburst(
    title='All sentiment sunburst chart',
    colorscale=True,
    color_continuous_scale='Oryel',
    width=width,
    height=height,
    save=True
)


