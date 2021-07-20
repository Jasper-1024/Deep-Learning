import tensorflow as tf
import numpy as np


class NaiveDense:  #一个基本的Dense层
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        w_shape = (input_size, output_size
                   )  #创建矩阵 w 尺寸为 (input_size, output_size) 随机初始化
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        b_shape = (output_size, )  #创建矩阵 b 尺寸为 (output_size) 随机初始化
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)  # 定义激活函数

    @property
    def weights(self):  #获取权重
        return [self.W, self.b]


class NaiveSequential:  #连接层
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):  #自上而下调用层
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):  #获取权重
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])

assert len(model.weights) == 4


class BatchGenerator:  #生成器 迭代处理 mnist 数据
    def __init__(self, images, labels, batch_size=128):
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def next(self):
        images = self.images[self.index:self.index + self.batch_size]
        labels = self.labels[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


def one_training_step(model, images_batch, labels_batch):#一次训练
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


# learning_rate = 1e-3

# def update_weights(gradients, weights):
#     for g, w in zip(gradients, model.weights):
#         w.assign_sub(w * learning_rate)

from tensorflow.keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)


def update_weights(gradients, weights):#更新权重
    optimizer.apply_gradients(zip(gradients, weights))


def fit(model, images, labels, epochs, batch_size=128):#训练
    for epoch_counter in range(epochs):
        print('Epoch %d' % epoch_counter)
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(len(images) // batch_size):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print('loss at batch %d: %.2f' % (batch_counter, loss))


from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)

predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")
