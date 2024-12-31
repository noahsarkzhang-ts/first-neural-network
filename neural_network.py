import numpy
import scipy.special

# neural network class definitioin


class neuralNetork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) -> None:

        # set number os nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learing rate
        self.lr = learningrate

        # link weight matrices, wih and who
        self.wih = numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, input_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final opuput layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nedes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (
            1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layer
        self.wih += self.lr * \
            numpy.dot((hidden_errors * hidden_outputs *
                      (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    # query the neural network
    def query(self, input_list):
        # convert inputs list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final opuput layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

        pass

    pass


# number of input, hidden and output nodes
inodes = 784
hnodes = 300
onodes = 10

# learning rate is 0.3
learning_rate = 0.2

# create instance of neural network
n = neuralNetork(inodes, hnodes, onodes, learning_rate)

# load the mnist training data CSV file into a list
# training_data_file = open('mnist_dataset/mnist_train_100.csv', 'r')
training_data_file = open('mnist_dataset/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural
train_num = 1
# go through all records in the training data set
for record in training_data_list:
    print("Train:",train_num)
    # split the record by the ',' commas
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01,except the desired label which is 0.99)
    targets = numpy.zeros(onodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

    train_num += 1
    pass

# load the mnist test data CSV file into a list
# test_data_file = open('mnist_dataset/mnist_test_10.csv', 'r')
test_data_file = open('mnist_dataset/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []
test_num = 1
# go through all the records in the test data set
for record in test_data_list:
    print("Test", test_num)
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # print(correct_label, "correct label")
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    test_num += 1
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # print(label, "network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass

# print(scorecard)

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asanyarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
