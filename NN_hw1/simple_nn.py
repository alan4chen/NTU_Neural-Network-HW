
# Back-Propagation Neural Networks
#
import math
import random

random.seed(10)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh1, nh2, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh1 = nh1 + 1
        self.nh2 = nh2 + 1
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah1 = [1.0]*self.nh1
        self.ah2 = [1.0]*self.nh2
        self.ao = [1.0]*self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh1-1)
        self.wh = makeMatrix(self.nh1, self.nh2-1)
        self.wo = makeMatrix(self.nh2, self.no)
        # set them to random vaules
        # for i in range(self.ni):
        #     for j in range(self.nh1-1):
        #         self.wi[i][j] = rand(-0.1, -0.1)
        # for i in range(self.nh1):
        #     for j in range(self.nh2-1):
        #         self.wh[i][j] = rand(-0.1, -0.1)
        # for j in range(self.nh2):
        #     for k in range(self.no):
        #         self.wo[j][k] = rand(-0.1, -0.1)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh1-1):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah1[j] = sigmoid(sum)

        # hidden activations
        for j in range(self.nh2 - 1):
            sum = 0.0
            for i in range(self.nh1):
                sum = sum + self.ah1[i] * self.wh[i][j]
            self.ah2[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh2):
                sum = sum + self.ah2[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            output_deltas[k] = (targets[k]-self.ao[k]) * dsigmoid(self.ao[k])

        # calculate error terms for hidden2
        hidden2_deltas = [0.0] * self.nh2
        for j in range(self.nh2):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k]*self.wo[j][k]
            hidden2_deltas[j] = dsigmoid(self.ah2[j]) * error

        # calculate error terms for hidden1
        hidden1_deltas = [0.0] * self.nh1
        for j in range(self.nh1):
            error = 0.0
            for k in range(self.nh2-1):
                error += hidden2_deltas[k] * self.wh[j][k]
            hidden1_deltas[j] = dsigmoid(self.ah1[j]) * error

        # update output weights
        for j in range(self.nh2):
            for k in range(self.no):
                change = output_deltas[k]*self.ah2[j]
                self.wo[j][k] += N*change
                # self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update hidden weights
        for j in range(self.nh1):
            for k in range(self.nh2-1):
                change = hidden2_deltas[k]*self.ah1[j]
                self.wh[j][k] += N*change
                # self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh1-1):
                change = hidden1_deltas[j]*self.ai[i]
                self.wi[i][j] += N*change
                # self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + (targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Hidden weights:')
        for j in range(self.nh1):
            print(self.wh[j])
        print()
        print('Output weights:')
        for k in range(self.nh2):
            print(self.wo[k])

    def train(self, patterns, iterations=1000, N=1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N)
            if i % 100 == 0:
                print('error %-.5f' % error)


def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1, 1)
    # train it with some patterns
    n.train(pat)

    n.weights()
    # test it
    n.test(pat)

def hw1():
    pat = []
    with open('./hw1data.dat') as f:
        line = f.readline()
        lst = line.split(" ")
        data_size = int(lst[0])
        input_num = int(lst[1])
        output_num = int(lst[2])
        for x in range(data_size):
            lst = f.readline().split("\t")
            x = [float(lst[0]), float(lst[1])]
            y = [1 if float(lst[2]) == 1 else 0]
            pat.append([x, y])
    print(pat)

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 4, 3, 1)
    # train itf with some patterns
    n.train(pat)
    n.weights()
    n.test(pat)

if __name__ == '__main__':
    # demo()
    hw1()