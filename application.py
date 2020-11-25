from flask import Flask, render_template, request
import numpy as np
from gensim.models import Word2Vec
import jieba
import heapq
import collections
import re
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

input_news = ""
processed_data_model = 'processed_data_model_simplified.model'
model = Word2Vec.load(processed_data_model)
total_word_count = sum([model.wv.vocab[word].count for word in model.wv.vocab])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarise', methods=["POST"])
def summarise():
    title = request.form.get("title")
    article = request.form.get("inputText")
    summary = _summarise(article, title)
    return render_template('summary.html', summary=summary, title=title, article=article)

@app.route('/reset', methods=["POST"])
def reset():
    return render_template('index.html')

def _summarise(content, title, top_n=3):
    a = 0.001
    v_c = sentence_to_vec(model, content, a)
    v_t = sentence_to_vec(model, title, a)

    words_index_table, words_weight_table, words_vector, valid_lines, all_words = get_all_words(model, title, content,
                                                                                                a)
    words_index_array = get_words_index_array(words_index_table, len(all_words))
    words_weight_array = get_words_weight_array(words_weight_table, len(all_words))
    words_vector_array = get_words_vector_array(all_words)

    v_s = SIF_embedding(words_vector_array, words_index_array, words_weight_array)
    # v_s = vectorise_article_content(model, content, a)
    # v_s = v_s.T
    sentence_indexs = top_n_sentence_index(v_s, v_c, v_t, top_n, valid_lines)
    res = []
    for score, index in sentence_indexs:
        #         print(index)
        res.append(valid_lines[index])
    return '，'.join(res) + '。'

def get_stopwords():
    stopwords = []
    with open('stop_words.txt', encoding = 'gbk') as f:
        for line in f:
            stopwords.append(line.strip())
    return stopwords

# special characters elimation elimination
def filter_special_characters(line):
    special_characters = re.compile("\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]")
    clean_line = special_characters.sub('', line)
    return clean_line

# cut words
def cut_words(line):
    return list(jieba.cut(line))

stopwords = get_stopwords()
def remove_stop_words(words):
    clean_words = []
    for word in words:
        if word not in stopwords:
            clean_words.append(word)
    return clean_words

def clean_sentence(sentence):
    sentence = filter_special_characters(sentence)
    cutted_words = remove_stop_words(cut_words(sentence))
    return cutted_words

def get_all_words(model, title, content, a=1e-3):
    all_words, valid_lines = [], []
    lines = re.findall('\w+', content)
    words_index_table = collections.defaultdict(list)
    words_weight_table = collections.defaultdict(list)
    words_vector = collections.defaultdict()
    index = 0
    for line in lines:
        # add word index for each sentence
        cutted_words = clean_sentence(line)
        for word in cutted_words:
            if word in model.wv.vocab:
                if word not in all_words:
                    all_words.append(word)
                words_index_table[index].append(all_words.index(word))
                p_w = a / (a + model.wv.vocab[word].count / total_word_count)
                words_weight_table[index].append((all_words.index(word), p_w))
        # add valid lines
        tmp = sentence_to_vec(model, line)
        if type(tmp) is np.ndarray:
            valid_lines.append(line)
            index += 1
    # add word vector
    for i,v in enumerate(all_words):
        words_vector[i] = model.wv[v]
    return words_index_table, words_weight_table, words_vector, valid_lines, all_words

def get_words_index_array(words_index_table, num_cols):
    num_rows = len(words_index_table.keys())
    words_index_array = np.zeros([num_rows,num_cols])
    for sentence_index,word_indexs in words_index_table.items():
        for word_index in word_indexs:
            words_index_array[sentence_index, word_index] = word_index
    return words_index_array

def get_words_weight_array(words_weight_table, num_cols):
    num_rows = model.wv.vectors.shape[1]
    words_weight_array = np.zeros([1,num_cols])
    for sentence_index,word_weight_vector in words_weight_table.items():
        tmp = np.zeros([1,num_cols])
        for word_index,word_weight in word_weight_vector:
            tmp[0,word_index] = word_weight
        words_weight_array = np.concatenate((words_weight_array, tmp),axis=0)
    return words_weight_array[1:,:]

def get_words_vector_array(all_words):
    num_cols = model.wv.vectors.shape[1]
    words_vector_array = np.zeros([1,num_cols])
    for word in all_words:
        word_vector = np.reshape(model.wv[word], (1,num_cols))
        words_vector_array = np.concatenate((words_vector_array,word_vector),axis=0)
    return words_vector_array[1:,:]

def get_weighted_average_vector(words_vector_array, words_index_array, words_weight_array):
    # return weighted_average_vector[i,:] for i-th sentence
    num_words = words_index_array.shape[0]
    weighted_average_vector = np.zeros((num_words, model.wv.vectors.shape[1]))
    for i in range(num_words):
        word_indexs = words_index_array[i,:].astype(int)
        weighted_average_vector[i,:] = words_weight_array[i,:].dot(words_vector_array[word_indexs,:])/np.count_nonzero(word_indexs)
    return weighted_average_vector

def compute_pc(weighted_average_vector, num_pc=1):
    # calculate (first) singular vector/ principal components
    svd = TruncatedSVD(n_components=num_pc, n_iter=7, random_state=0)
    svd.fit(weighted_average_vector)
    return svd.components_


def remove_pc(weighted_average_vector, num_pc=1):
    pc = compute_pc(weighted_average_vector, num_pc)
    if num_pc == 1:
        res = weighted_average_vector - weighted_average_vector.dot(pc.transpose()) * pc
    else:
        res = weighted_average_vector - weighted_average_vector.dot(pc.transpose()).dot(pc)
    return res

def SIF_embedding(words_vector_array, words_index_array, words_weight_array, num_pc=1):
    # words_index_array[i,:] is the vector for words indices in i-th sentence
    # words_weight_array[i,:] is the vector for weights of the words in i-th sentence
    # words_vector_array[i:] is the vector for word i
    embedding = get_weighted_average_vector(words_vector_array, words_index_array, words_weight_array)
    if num_pc > 0:
        embedding = remove_pc(embedding)
    return embedding

# v_si, each sentence vector in the article
# v_t, title vector
# v_c, whole article vector
def calculate_similarity(v_si, v_c, v_t):
    sentence_to_title = 1 - distance.cosine(v_si, v_t)
    sentence_to_article = 1 - distance.cosine(v_si, v_c)
#     sentence_to_title = cosine_similarity(v_si.reshape(1,-1), v_t.reshape(1,-1))
#     sentence_to_article = cosine_similarity(v_si.reshape(1,-1), v_c.reshape(1,-1))
    return sentence_to_title+sentence_to_article

def knn_smoothed(scores, mid=3, left=1, right=1):
    res = [0]*len(scores)
    for i in range(1,len(scores)-1):
        res[i] = (scores[i-1]*left + scores[i]*mid + scores[i+1]*right) / (mid + left + right)
    res[0] = (scores[0] * mid + scores[1] * right) / (mid + right)
    res[-1] = (scores[-1] * mid + scores[-2] * left) / (mid + left)
    return res

def top_n_sentence_index(v_s, v_c, v_t, top_n, valid_lines):
    sentence_indexs = []
    scores = []
    for i in range(len(v_s)):
        score = calculate_similarity(v_s[i,:], v_c, v_t)
        scores.append(score)
    smoothed_scores = knn_smoothed(scores,mid=3,left=1,right=1)
    for i in range(len(scores)):
        heapq.heappush(sentence_indexs, (smoothed_scores[i], i))
    return heapq.nlargest(top_n, sentence_indexs)

def sentence_to_vec(model, line, a=1e-3):
    v_s = 0
    cutted_words = clean_sentence(line)
    for word in cutted_words:
        if word in model.wv.vocab:
            word_count = model.wv.vocab[word].count
            p_w = word_count / model.wv.vectors.shape[0]
            v_s += a / (a + p_w) * model.wv[word]
    if len(cutted_words) != 0:
        v_s = v_s / len(cutted_words)
    return v_s

if __name__ == '__main__':
    app.run(debug=True)