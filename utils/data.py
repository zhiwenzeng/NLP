from utils.const import _const as const
from utils.word2vec import w2v
from tensorflow.python import keras
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def text2data(texts):
    # 使用预处理好的word2vec模型
    vocab = w2v.get_model(const.w2v.model_path).wv.vocab
    res = []
    # 进行序词语索引
    for sentence in texts:
        idxs = []
        for word in sentence:
            try:
                idxs.append(vocab[str(word)].index + 1)
            except Exception as e:
                idxs.append(0)
        res.append(idxs)

    # 将文本变为pad_sequences
    res = keras.preprocessing.sequence.pad_sequences(res, maxlen=const.w2v.vector_size)
    return res