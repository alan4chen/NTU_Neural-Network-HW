

from __future__ import print_function



import numpy as np
import _pickle as cPickle


def load_hw1_data():
    train_x = []
    train_y = []
    with open('./hw1data.dat') as f:
        line = f.readline()
        lst = line.split(" ")
        data_size = int(lst[0])
        input_num = int(lst[1])
        output_num = int(lst[2])

        for x in range(data_size):
            lst = f.readline().split("\t")
            train_x.append([float(lst[0]), float(lst[1])])
            train_y.append([1 if float(lst[2]) == 1 else -1])
    return np.array(train_x), np.array(train_y)

def train():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import rmsprop

    train_x, train_y = load_hw1_data()
    # print(train_x)
    print(train_x.shape)
    # print(train_y)
    print(train_y.shape)

    num_classes = 1
    epochs = 10000

    model = Sequential()
    model.add(Dense(output_dim=8, input_dim=2, activation='tanh'))
    # model.add()
    model.add(Dense(output_dim=6, activation='tanh'))
    # model.add(Activation('tanh'), )
    model.add(Dense(output_dim=1, activation='tanh'))

    model.summary()

    model.compile(loss='mse',
                  optimizer=rmsprop(),
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y,
                        epochs=epochs, batch_size=1000)
    score = model.evaluate(train_x, train_y, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    predict_y = model.predict(train_x)
    print(predict_y)

    # print(history.history.keys())
    # print(history.history['loss'])

    return model, history.history


if __name__ == "__main__":

    # model, history = train()
    # model.save("hw1_3_1.model")
    # fw_h = open("hw1_3_1.history", 'wb')
    # cPickle.dump(history, fw_h)
    # fw_h.close()

    from keras.models import load_model
    model = load_model('hw1_3_1.model')
    fr_h = open("hw1_3_1.history", 'rb')
    history = cPickle.load(fr_h)
    fr_h.close()

    print(history)
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')



