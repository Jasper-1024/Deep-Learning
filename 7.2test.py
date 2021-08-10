import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers

vocabulary_size = 10000  #文本编码长度
num_tags = 100  # one-hot 编码长度
num_departments = 4  # 部门几个
# 输入数据
title = keras.Input(shape=(vocabulary_size, ), name="title")
text_body = keras.Input(shape=(vocabulary_size, ), name="text_body")
tags = keras.Input(shape=(num_tags, ), name="tags")
# 中间 layer
features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)
# 输出数据
priority = layers.Dense(1, activation="sigmoid",
                        name="priority")(features)  #优先级
department = layers.Dense(num_departments,
                          activation="softmax",
                          name="department")(features)  #部门

model = keras.Model(inputs=[title, text_body, tags],
                    outputs=[priority, department])

num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))  #标题
text_body_data = np.random.randint(0, 2,
                                   size=(num_samples, vocabulary_size))  #正文
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))  #标签

priority_data = np.random.random(size=(num_samples, 1))  #优先级
department_data = np.random.randint(0, 2,
                                    size=(num_samples, num_departments))  #部门

model.compile(
    optimizer="adam",  #优化器
    loss=["mean_squared_error", "categorical_crossentropy"],  #损失函数
    metrics=[["mean_absolute_error"], ["accuracy"]])  #指标

model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data],
          epochs=1)  #训练模型,传入数据要和 model 声明一致

# plot_model(model, "./ticket_classifier.png")

keras.utils.plot_model(model, "./ticket_classifier.png")