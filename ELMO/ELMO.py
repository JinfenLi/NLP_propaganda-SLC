import os
import csv
import time
import datetime
import random

from collections import Counter
from math import sqrt

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score,f1_score

from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings, Batcher


# 配置参数

class TrainingConfig(object):
    epoches = 20
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.01


class ModelConfig(object):
    embeddingSize = 256  # 这个值和ELMo的词向量大小一致

    hiddenSizes = [128]  # LSTM结构的神经元个数

    dropoutKeepProb = 0.5
    l2RegLambda = 0.0


class Config(object):
    sequenceLength = 70  # 取了所有序列长度的均值
    batchSize = 128

    dataSource = "labeledTrain.tsv"

    stopWordSource = "data\english"

    optionFile = "modelParams\elmo_options.json"
    weightFile = "modelParams\elmo_weights.hdf5"
    vocabFile = "modelParams\\vocab.txt"
    tokenEmbeddingFile = 'modelParams\elmo_token_embeddings.hdf5'

    numClasses = 2

    rate = 0.9  # 训练集的比例

    training = TrainingConfig()

    model = ModelConfig()


# 实例化配置参数对象
config = Config()


# 数据预处理的类，生成训练集和测试集

class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource
        self._optionFile = config.optionFile
        self._weightFile = config.weightFile
        self._vocabFile = config.vocabFile
        self._tokenEmbeddingFile = config.tokenEmbeddingFile

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.review_test = []
        self.labels_test = []

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """
        import csv
        with tf.gfile.Open('labeledTrain.tsv', "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
        review = []
        labels = []
        for line in reader:
            review.append(line[1])
            labels.append(int(line[0]))
        reviews = [line.strip().split() for line in review]

        with tf.gfile.Open('test.tsv', "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
        review_test = []
        labels_test = []
        for line in reader:
            review_test.append(line[1])
            labels_test.append(int(line[0]))
        review_test = [line.strip().split() for line in review_test]

        return reviews, labels,review_test,labels_test

    def _genVocabFile(self, reviews):
        """
        用我们的训练数据生成一个词汇文件，并加入三个特殊字符
        """
        allWords = [word for review in reviews for word in review]
        wordCount = Counter(allWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        words = [item[0] for item in sortWordCount.items()]
        allTokens = ['<S>', '</S>', '<UNK>'] + words
        with open(self._vocabFile, 'w') as fout:
            fout.write('\n'.join(allTokens))

    def _fixedSeq(self, reviews):
        """
        将长度超过200的截断为200的长度
        """
        return [(review+['##']*30)[:self._sequenceLength] for review in reviews]

    def _genElmoEmbedding(self):
        """
        调用ELMO源码中的dump_token_embeddings方法，基于字符的表示生成词的向量表示。并保存成hdf5文件，文件中的"embedding"键对应的value就是
        词汇表文件中各词汇的向量表示，这些词汇的向量表示之后会作为BiLM的初始化输入。
        """
        dump_token_embeddings(
            self._vocabFile, self._optionFile, self._weightFile, self._tokenEmbeddingFile)

    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """
        y = [[item] for item in y]
        trainIndex = int(len(x) * rate)

        trainReviews = x[:trainIndex]
        trainLabels = y[:trainIndex]

        evalReviews = x[trainIndex:]
        evalLabels = y[trainIndex:]

        return trainReviews, trainLabels, evalReviews, evalLabels

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化数据集
        reviews, labels,review_test,labels_test = self._readData(self._dataSource)

        #         self._genVocabFile(reviews) # 生成vocabFile
        #         self._genElmoEmbedding()  # 生成elmo_token_embedding

        reviews = self._fixedSeq(reviews)
        review_test = self._fixedSeq(review_test)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

        self.review_test = review_test
        self.labels_test = [[item] for item in labels_test]


data = Dataset(config)
data.dataGen()

print("train data shape: {}".format(len(data.trainReviews)))
print("train label shape: {}".format(len(data.trainLabels)))
print("eval data shape: {}".format(len(data.evalReviews)))

# 输出batch数据集
def nextBatch(x, y, batchSize):
        """
        生成batch数据集，用生成器的方式输出
        """
        # 每一个epoch时，都要打乱数据集
        midVal = list(zip(x, y))
        random.shuffle(midVal)
        x, y = zip(*midVal)
        x = list(x)
        y = list(y)

        numBatches = len(x) // batchSize

        for i in range(numBatches):
            start = i * batchSize
            end = start + batchSize
            batchX =x[start: end]
            batchY = y[start: end]

            yield batchX, batchY


def testnextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """
    # 每一个epoch时，都要打乱数据集
    #midVal = list(zip(x, y))
    #random.shuffle(midVal)
    #x, y = zip(*midVal)
    x = list(x)
    y = list(y)

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = x[start: end]
        batchY = y[start: end]

        yield batchX, batchY


# 构建模型
class BiLSTMAttention(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config):
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.float32, [None, config.sequenceLength, config.model.embeddingSize],
                                     name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            embeddingW = tf.get_variable(
                "embeddingW",
                shape=[config.model.embeddingSize, config.model.embeddingSize],
                initializer=tf.contrib.layers.xavier_initializer())

            reshapeInputX = tf.reshape(self.inputX, shape=[-1, config.model.embeddingSize])

            self.embeddedWords = tf.reshape(tf.matmul(reshapeInputX, embeddingW),
                                            shape=[-1, config.sequenceLength, config.model.embeddingSize])
            self.embeddedWords = tf.nn.dropout(self.embeddedWords, self.dropoutKeepProb)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   self.embeddedWords, dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                    self.embeddedWords = tf.concat(outputs_, 2)

        outputs = tf.split(self.embeddedWords, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self._attention(H)
            outputSize = config.model.hiddenSizes[-1]

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, 1],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.0), tf.float32, name="binaryPreds")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = config.model.hiddenSizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, config.sequenceLength])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, config.sequenceLength, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output


# 定义性能指标函数

def mean(item):
    return sum(item) / len(item)

from sklearn.metrics import f1_score
def genMetrics(trueY, predY, binaryPredY):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    f1 = f1_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)

    return round(accuracy, 4), round(auc, 4), round(precision, 4),round(f1, 4), round(recall, 4)


# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

testReviews = data.review_test
testLabels = data.labels_test

# 定义计算图

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():
        cnn = BiLSTMAttention(config)

        # 实例化BiLM对象，这个必须放置在全局下，不能在elmo函数中定义，否则会出现重复生成tensorflow节点。
        with tf.variable_scope("bilm", reuse=True):
            bilm = BidirectionalLanguageModel(
                config.optionFile,
                config.weightFile,
                use_character_inputs=False,
                embedding_weight_file=config.tokenEmbeddingFile
            )
        inputData = tf.placeholder('int32', shape=(None, None))

        # 调用bilm中的__call__方法生成op对象
        inputEmbeddingsOp = bilm(inputData)

        # 计算ELMo向量表示
        elmoInput = weight_layers('input', inputEmbeddingsOp, l2_coef=0.0)

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", cnn.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        #         builder = tf.saved_model.builder.SavedModelBuilder("../model/textCNN/savedModel")
        sess.run(tf.global_variables_initializer())


        def elmo(reviews):
            """
            对每一个输入的batch都动态的生成词向量表示
            """

            #           tf.reset_default_graph()
            # TokenBatcher是生成词表示的batch类
            batcher = TokenBatcher(config.vocabFile)

            # 生成batch数据
            inputDataIndex = batcher.batch_sentences(reviews)

            # 计算ELMo的向量表示
            elmoInputVec = sess.run(
                [elmoInput['weighted_op']],
                feed_dict={inputData: inputDataIndex}
            )

            return elmoInputVec


        def trainStep(batchX, batchY):
            """
            训练函数
            """

            feed_dict = {
                cnn.inputX: elmo(batchX)[0],  # inputX直接用动态生成的ELMo向量表示代入
                cnn.inputY: np.array(batchY, dtype="float32"),
                cnn.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions, binaryPreds = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()
            acc, auc, precision,F1, recall = genMetrics(batchY, predictions, binaryPreds)
            print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step, loss, acc,
                                                                                               auc, precision, recall))
            trainSummaryWriter.add_summary(summary, step)


        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                cnn.inputX: elmo(batchX)[0],
                cnn.inputY: np.array(batchY, dtype="float32"),
                cnn.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions, binaryPreds = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)

            acc, auc, precision,F1, recall = genMetrics(batchY, predictions, binaryPreds)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc,F1, auc, precision, recall


        def testStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                cnn.inputX: elmo(batchX)[0],
                cnn.inputY: np.array(batchY, dtype="float32"),
                cnn.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions, binaryPreds = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)

            return binaryPreds[:,0]


        for i in range(config.training.epoches):
            # 训练模型
            print("start training model,epoch:"+str(i))
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")

                    losses = []
                    accs = []
                    aucs = []
                    list_F1=[]
                    precisions = []
                    recalls = []

                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc,F1,auc, precision, recall = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        aucs.append(auc)
                        list_F1.append(F1)
                        precisions.append(precision)
                        recalls.append(recall)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {}, auc: {}, F1: {}, precision: {}, recall: {}".format(time_str,
                                                                                                       currentStep,
                                                                                                       mean(losses),
                                                                                                       mean(accs),
                                                                                                       mean(aucs),
                                                                                                       mean(list_F1),
                                                                                                       mean(precisions),
                                                                                                       mean(recalls)))
        predictions = []
        for batchtest in testnextBatch(testReviews, testLabels, config.batchSize):
            predictions=predictions+list(testStep(batchtest[0], batchtest[1]))



        #predictions.append(testStep(testReviews, testLabels))
        submission = pd.DataFrame()
        submission['prediction'] = predictions
        submission.to_csv('submission_ELMO.csv', index=False)

#
#
# if currentStep % config.training.checkpointEvery == 0:
#                     # 保存模型的另一种方法，保存checkpoint文件
#                     path = saver.save(sess, "../model/textCNN/model/my-model", global_step=currentStep)
#                     print("Saved model checkpoint to {}\n".format(path))

#         inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
#                   "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}

#         outputs = {"binaryPreds": tf.saved_model.utils.build_tensor_info(cnn.binaryPreds)}

#         prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
#                                                                                       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
#         legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
#         builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
#                                             signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

#         builder.save()