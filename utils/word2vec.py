from utils.const import _const as const
import warnings
import os
import logging
from gensim.models import word2vec
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')


class w2v:

    model = None

    # 自定义错误类型
    class w2vError(TypeError):
        pass

    def __init__(self):
        pass

    '''获取word2vec模型'''
    @classmethod
    def get_model(cls, model_path):
        if not os.path.exists(model_path):
            raise cls.w2vError("not found %s", model_path)
        if cls.model == None:
            cls.model = word2vec.Word2Vec.load(model_path)
        return cls.model

    '''
        增量训练word2vec模型
        sentences: 分词后的文本序列,例如: [['体育', '姚明', ..], ['娱乐', '倪妮', '气质', ..], ..]
    '''
    @classmethod
    def train(cls, sentences, model_path):
        if not os.path.exists(model_path):
            model = word2vec.Word2Vec(sentences, size=const.w2v.vector_size, window=5, min_count=5, workers=3, sg=1)
        else:
            model = word2vec.Word2Vec.load(model_path)
            model.build_vocab(sentences, update=True)
            model.train(sentences, total_examples=len(sentences), epochs=model.epochs)

        model.save(model_path)