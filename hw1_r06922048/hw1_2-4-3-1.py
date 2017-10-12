

from __future__ import print_function



import numpy as np

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
    model.add(Dense(output_dim=4, input_dim=2, activation='tanh'))
    # model.add()
    model.add(Dense(output_dim=3, activation='tanh'))
    # model.add(Activation('tanh'), )
    model.add(Dense(output_dim=1, activation='tanh'))

    model.summary()

    model.compile(loss='mse',
                  optimizer=rmsprop(),
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y,
                        epochs=epochs, batch_size=1000)
    score = model.evaluate(train_x, train_y, verbose=1)

    print('Train loss:', score[0])
    print('Train accuracy:', score[1])

    predict_y = model.predict(train_x)
    print(predict_y)

    # print(history.history.keys())
    # print(history.history['loss'])

    return model, history.history


if __name__ == "__main__":

    model, history = train()


