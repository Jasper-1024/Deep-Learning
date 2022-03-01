import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

num_samples_per_classs = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0,3],cov=[[1,0.5],[0.5,1]],size=num_samples_per_classs)
positive_samples = np.random.multivariate_normal(
    mean=[3,0],cov=[[1,0.5],[0.5,1]],size=num_samples_per_classs)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples_per_classs, 1), dtype="float32"),
                     np.ones((num_samples_per_classs, 1), dtype="float32")))


model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer="rmsprop",
              loss="mean_squared_error",
              metrics=["accuracy"])



# model.compile(optimizer=keras.optimizers.RMSprop(),
#               loss=keras.losses.MeanSquaredError())


history = model.fit(
    inputs,
    targets,
    epochs=100,
    batch_size=128
)

print(history.history)


# input_dim = 2 # 输入数据阶数
# output_dim = 1 # 输出数据阶数

# W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
# b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

# def model(inputs):
#     return tf.matmul(inputs, W) + b # prediction = W * P + b

# def square_loss(targets, predictions):
#     per_sample_loss = tf.square(
#         targets - predictions)  #per_sample_loss =  (targets - predictions)^2
#     return tf.reduce_mean(per_sample_loss)  # per_sample_loss 平均值

# learning_rate = 0.1  # 学习率


# def training_step(inputs, targets):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs)  #预测值
#         loss = square_loss(targets, predictions)  #计算损失
#     grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])  #计算梯度
#     W.assign_sub(
#         grad_loss_wrt_W *
#         learning_rate)  #更新 W 值 W = W + grad_loss_wrt_W * learning_rate
#     b.assign_sub(
#         grad_loss_wrt_b *
#         learning_rate)  #更新 b 值 b = b + grad_loss_wrt_b * learning_rate
#     return loss

# for step in range(40):
#     loss = training_step(inputs,targets)
#     print(f"Loss at step {step}:{loss}")

# predictions = model(inputs)
# plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
# plt.show()
