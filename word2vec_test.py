import tensorflow as tf
import math

vocabulary_size = 10000
embedding_size = 128
examples = [3,3,3,3,10,10,10,10]
labels = [2,1,3,5,3,5,6,82]
batch_size = 8
num_samples = 8   #num_samples 为采样个数




###构建计算流图
# 首先定义词向量矩阵，也称为 embedding matrix，这个是我们需要通过训练得到的词向量，其中vocabulary_size表示词典大小，
# embedding_size表示词向量的维度，那么词向量矩阵为 vocabulary_size × embedding_size，利用均匀分布对它进行随机初始化：
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

#定义权值矩阵和偏置向量，并初始化为0：
weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
biases = tf.Variable(tf.zeros([vocabulary_size]))


#给定一个batch的输入，从词向量矩阵中找到对应的向量表示，以及从权值矩阵和偏置向量中找到对应正确输出的参数，
# 其中examples是输入词，labels为对应的正确输出，一维向量表示，每个元素为词在字典中编号：
# Embeddings for examples: [batch_size, embedding_size]
example_emb = tf.nn.embedding_lookup(embeddings, examples)
# Weights for labels: [batch_size, embedding_size]
true_w = tf.nn.embedding_lookup(weights, labels)
# Biases for labels: [batch_size, 1]
true_b = tf.nn.embedding_lookup(biases, labels)


#负采样得到若干非正确的输出，其中labels_matrix为正确的输出词，采样的时候会跳过这些词，num_samples为采样个数，
# distortion即为公式(3-4)中的幂指数：
labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64),[batch_size, 1])
# Negative sampling.
#详情：https://blog.csdn.net/u011026968/article/details/88537939
sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
    true_classes=labels_matrix,
    num_true=1,
    num_sampled=num_samples,
    unique=True,
    range_max=vocabulary_size,
    distortion=0.75,
    unigrams=[1]*10000)


#找到采样样本对应的权值和偏置参数：
# Weights for sampled ids: [num_samples, embedding_size]
sampled_w = tf.nn.embedding_lookup(weights, sampled_ids)
# Biases for sampled ids: [num_samples, 1]
sampled_b = tf.nn.embedding_lookup(biases, sampled_ids)


#分别计算正确输出和非正确输出的logit值，即计算 WX+b，并通过交叉熵得到目标函数(3-3)：
# True logits: [batch_size, 1]
true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b
# Sampled logits: [batch_size, num_sampled]
# We replicate sampled noise lables for all examples in the batch
# using the matmul.
sampled_b_vec = tf.reshape(sampled_b, [num_samples])
sampled_logits = tf.reduce_sum(tf.multiply(example_emb, sampled_w), 1) + sampled_b_vec
# cross-entropy(logits, labels)
true_xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=true_logits,labels= tf.ones_like(true_logits))
sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=sampled_logits,labels= tf.zeros_like(sampled_logits))
# NCE-loss is the sum of the true and noise (sampled words)
# contributions, averaged over the batch.
loss = (tf.reduce_sum(true_xent) +tf.reduce_sum(sampled_xent)) / batch_size

###训练模型
#计算流图构建完毕后，我们需要去优化目标函数。采用梯度下降逐步更新参数，首先需要确定学习步长，随着迭代进行，
# 逐步减少学习步长，其中trained_words为已训练的词数量，words_to_train为所有待训练的词数量：
lr = init_learning_rate * tf.maximum(0.0001, 1.0 - tf.cast(trained_words, tf.float32) / words_to_train)

#定义优化算子，使用梯度下降训练模型：
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss,
                           global_step=global_step,
                           gate_gradients=optimizer.GATE_NONE)
session.run(train)