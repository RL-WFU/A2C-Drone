import tensorflow as tf
import numpy as np
from env_2 import *
import argparse
from tensorflow import keras
from keras import models
from keras import layers
import matplotlib.pyplot as plt






def produceSections(env, env_dims=25, sight_dim=2):
    sim = env.sim
    sections = []
    for i in range(env_dims - sight_dim * 2):
        for j in range(env_dims - sight_dim * 2):
            row_pos = sight_dim + i
            col_pos = sight_dim + j
            section = sim.getClassifiedDroneImageAt(row_pos, col_pos)
            sections.append(section)

    sections = np.asarray(sections)

    sections = np.reshape(sections, [-1, (sight_dim * 2 + 1) * (sight_dim * 2 + 1), 3])

    return sections

def produceLabels(sections, sight_dim):
    labels = np.zeros(len(sections))
    for i in range(len(sections)):
        for j in range((sight_dim * 2 + 1) * (sight_dim * 2 + 1)):
            if np.max(sections[i, j]) == sections[i, j, 0]:
                labels[i] = 1

    return labels

def shuffleData(sections, labels):
    a = []
    b = []
    indices = np.arange(len(labels))

    for i in range(len(labels)):
        num = np.random.randint(0, len(indices), 1)
        index = indices[num]
        a.append(sections[index, :, :])
        b.append(labels[index])
        indices = np.delete(indices, num)

    a = np.asarray(a)
    b = np.asarray(b)

    return a, b


def prepareData(args):
    environments = []
    img_dir = 'Train Images'
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".TIF") or filename.endswith(".JPG"):
            env = Env(args, os.path.join(img_dir, filename))
            environments.append(env)
        else:
            continue

    x = produceSections(environments[0], args.env_dims, args.sight_dim)
    for i in range(len(environments) - 1):
        x = np.concatenate([x, produceSections(environments[i + 1], args.env_dims, args.sight_dim)], axis=0)

    x = np.asarray(x)

    y = produceLabels(x, args.sight_dim)

    y = np.asarray(y)

    x, y = shuffleData(x, y)
    x = np.squeeze(x)
    y = np.squeeze(y)


    eighty = int(round(len(x) * .8))

    train_split_x = x[:eighty, :, :]
    train_split_y = y[:eighty]

    test_x = x[eighty:, :, :]
    test_y = y[eighty:]


    eighty = int(round(len(train_split_x) * .8))

    training_x= train_split_x[:eighty, :, :]
    validation_x = train_split_x[eighty:, :, :]

    training_y = train_split_y[:eighty]
    validation_y = train_split_y[eighty:]

    training_x = np.reshape(training_x, [-1, ((args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)) * 3])
    validation_x = np.reshape(validation_x, [-1, ((args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)) * 3])
    test_x = np.reshape(test_x, [-1, ((args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1)) * 3])


    return training_x, validation_x, test_x, training_y, validation_y, test_y


def train_model(args):
    train_x, val_x, test_x, train_y, val_y, test_y = prepareData(args)
    print(train_x.shape, val_x.shape, test_x.shape)

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=[75,]))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y, epochs=300, batch_size=128, validation_data=(val_x, val_y))

    results = model.evaluate(test_x, test_y)
    print(results)

    return model


#model = train_model(args) #This model is pre-trained

