class _const:

    class w2v:
        model_path = None
        vector_size = None

    class cnn:
        input_size = None
        filters = None
        kernel_size = None
        pool_size = None
        output_size = None
        model_path = None
        pictures = None
        labels = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
        to_labels = {}
        to_hots = {}
        for i in range(len(labels)):
            to_labels[i] = labels[i]
            to_hots[labels[i]] = i

    class jieba:
        stopwords = None


_const.w2v.model_path = r'../w2v.model'
_const.w2v.vector_size = 100
_const.cnn.input_size = _const.w2v.vector_size
_const.cnn.filters = 256
_const.cnn.kernel_size = [3, 4, 5]
_const.cnn.pool_size = [25, 24, 23]
_const.cnn.output_size = 14
_const.cnn.model_path = r'../textcnn_model.h5'
_const.jieba.stopwords = r'../stopwords.txt'