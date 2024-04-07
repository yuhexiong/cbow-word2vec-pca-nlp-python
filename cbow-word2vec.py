import re
import jieba
from gensim.models import word2vec
from pprint import pprint

# read article
article = ""

infile_m = open(r'article/article1.txt', 'r', encoding='utf-8')
article += infile_m.read()
infile_m.close()

# clean data
regex1 = r'https?://[a-zA-Z0-9./_%-@=]+|[a-zA-Z0-9]+@[a-zA-Z0-9.]+|@[a-zA-Z0-9.]+|\d\.|\(\d\)'
regex2 = r'，|,|。|…|！|!|？|\?|、|；|【|】|〖|〗|「|」|:|：|;|；|＆|&|\(|（|\)|）|・|．|<|>|-|–|╴|\[|]|［|］|\*|~|～|\'|\"|{|}|\^|⌃|\$|\||●|•|★|✦|❖|■|※|✓|⇧|⌘|＝|=|@|\r|✨|📌|💡|🔒|👋|🌵|💛|_|👷|👨|💼|💻|📚|⭐|📖|🍌|🔥|😭|😆|😅|👏|😄|😎|😍|🎉|🙂|✅|🤪|🤔|👊|😏|😕|🙌|👌|🚩|😂|👉|📍|😣|🤩|🎵|🌈|🥺|🤘|⚡|📮|💁|🙏|👇|🌊|❓|🔮|👍|😵|💃|😱|🎊|✌|🎨|👨|😛|😳|😲|😊'

contentV1 = []

for i in range(len(article)):
    c00 = re.sub(regex1, ' ', article[i])
    c01 = re.sub(regex2, ' ', c00)
    c02 = re.sub(r'^\s+', '', c01)
    if c02 != "" and c02[:8]!= "ERROR410":
        contentV1.append(c02)

# stop word
stopWord = r'的|你|妳|我|他|們|您|這個|那個|與|和|或|及|之|等|不只|是|以及|每次|提供|例如|並|者|又|至|最|於|需|需要|而且|且|目前|尋找|想找|尤佳|更|佳|希望|職缺|公司|工作|內容|職務|面試|面談|人才|履歷|職位|包含|一間|具備|熟悉|熟練|成立|主要|單位|眾多|負責|歡迎|加入|至少|使用|相關|擔任|回覆|對於|協助|辦理|事務|執行|關於|以下|其他|致力|身為|簡介|企業|另行|懂|非常|謝謝|帶來|若|如果|如下|未填寫'
enStopWord = r'\ba\b|\bi\b|\byou\b|\bwe\b|\bis\b|\bare\b|\bwas\b|\bwere\b|\band\b|\bthe\b|\bin\b|\bfor\b|\bon\b|\bto\b|\bat\b|\bof\b'
regex3 = r'(?=import)|(?=matplotlib)|(?=python)|(?=jupyter)|(?=pip)|(?=tensorflow)|(?=anaconda)|(?=linux)|(?=firewall)|(?=command)'

contentV2 = []

for i in range(len(contentV1)):
    c10 = re.sub(stopWord, ' ', contentV1[i])
    c11 = c10.lower() # lower case
    c12 = re.sub(enStopWord, ' ', c11)
    c13 = re.sub(regex3, ' ', c12)
    c14 = re.sub(r'\s+', ' ', c13)
    contentV2.append(c14)

# train jieba with words(by web crawling)
path_dic = [r'dictionary\dic1.txt',
            r'dictionary\dic2.txt',
            r'dictionary\dic3.txt']
for dic in path_dic:
    with open(dic, 'r', encoding='utf-8') as infile:
        d = infile.read()
    for word in d:
        jieba.add_word(word+' 1000')

# Default Mode
contentV3 = []
seg_list = []
for i in range(len(contentV2)):
    seg_list = jieba.cut(contentV2[i], cut_all=False)
    contentV3.append(" ".join(seg_list))

# split as list
regex4 = r'\b\d+\b'
contentV4 = []

for i in range(len(contentV3)):
    c20 = re.sub(regex4, ' ', contentV3[i])
    c21 = c20.split()
    c_list = []
    i = len(c21)-1
    while i>=0:
        if c21[i] == '.' or c21[i] == '/' or c21[i] == '//' or c21[i] == '\\' or c21[i] == '\\\\' or (not c21[i].isprintable()):
            i -= 1
            continue
        c_list.append(c21[i])
        i -= 1
    contentV4.append(c_list[::-1])

# train word2vec
seed = 666
sg = 1
window_size = 10
vector_size = 100
min_count = 15
workers = 8
epochs = 5
batch_words = 10000

train_data = contentV4
model = word2vec.Word2Vec(
    train_data,
    min_count=min_count,
    vector_size=vector_size,
    workers=workers,
    epochs=epochs,
    window=window_size,
    sg=sg,
    seed=seed,
    batch_words=batch_words
)

model.save('word2vec.model')