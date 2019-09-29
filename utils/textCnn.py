from utils.word2vec import w2v
from utils.const import _const as const
import os
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import layers, models

class textCnn:

    """
        TextCnn构造
            参数:
                train_data: 训练数据
                test_data: 测试数据
                w2v_model: word2vec模型
    """
    def __init__(self, train_data=None, test_data=None, w2v_model=w2v.get_model(const.w2v.model_path)):
        self.train_data = train_data
        self.test_data = test_data
        self.model = None

        '''
            参数:
                model: word2vec模型
            返回:
                vocab: 词汇
                embedding_matrix: 每个词词向量构成的矩阵
        '''
        def _(model):
            vocab = model.wv.vocab
            embedding_matrix = np.zeros((len(vocab) + 1, const.w2v.vector_size))
            for word, i in vocab.items():
                try:
                    embedding_vector = model[str(word)]
                    embedding_matrix[i.index+1] = embedding_vector
                except KeyError:
                    continue
            return vocab, embedding_matrix

        if w2v_model is not None:
            self.vocab, self.embedding_matrix = _(w2v_model)

        '''模型存在则直接加载模型,模型不存在就创建模型'''
        if os.path.exists(const.cnn.model_path):
            self.model = models.load_model(const.cnn.model_path)
        else:
            self.model = self.__get_model()

    '''
        创建TextCnn模型
    '''
    def __get_model(self):
        # 输入x
        inputs = layers.Input(shape=(const.cnn.input_size,))
        # 向量嵌入层
        emb = layers.Embedding(input_dim=len(self.vocab)+1, output_dim=const.w2v.vector_size, input_length=const.cnn.input_size,
                               weights=[self.embedding_matrix], trainable=False)
        emb = emb(inputs)
        # 卷积和池化层
        pool_output = []
        for kernel_size, pool_size in zip(const.cnn.kernel_size, const.cnn.pool_size):
            cnv = layers.Conv1D(filters=const.cnn.filters, kernel_size=kernel_size, strides=1, padding='same',
                                activation='relu')
            pool = layers.MaxPooling1D(pool_size=pool_size)
            pool_output.append(pool(cnv(emb)))
        pool_output = layers.concatenate(pool_output, axis=-1)
        # 输入层(数据扁平层)
        flat = layers.Flatten()
        flat = flat(pool_output)
        # 解决过拟合
        drop = layers.Dropout(0.2)
        drop = drop(flat)
        # 输出层
        outputs = layers.Dense(const.cnn.output_size, activation='softmax')
        outputs = outputs(drop)
        # 创建模型
        model = keras.Model(inputs=inputs, outputs=outputs)
        # 配置学习过程
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    '''
        保存TextCnn模型
    '''
    def save(self):
        self.model.save(const.cnn.model_path)

    '''
        增量训练
    '''
    def fit(self):
        return self.__train()

    def __train(self):
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        # 独热编码
        y_train = keras.utils.to_categorical(y_train, num_classes=const.cnn.output_size)
        y_test = keras.utils.to_categorical(y_test, num_classes=const.cnn.output_size)
        # 训练
        history = self.model.fit(x_train, y_train, batch_size=50, epochs=5, validation_data=(x_test, y_test))
        # 评测
        test_scores = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', test_scores[0])
        print('Test accuracy:', test_scores[1])
        return history
