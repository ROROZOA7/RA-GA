import copy
import numpy as np
from tensorflow.keras.utils import to_categorical


class NeuralNetwork2(object):
    '''
    Main class to build Neural Network - pure python version
    '''
    
    
    def __init__(self, learning_rate=0.1, momentum=0.9, train_err_threshold=0.003, val_err_threshold=0,
                 num_partial_training=40, random_seed=None):
        '''
        :param learning_rate: learning rate 
        :param momentum: value ranges from [0,1]. Use a low learning rate with high momentum and vice versa.
        :param train_err_threshold: constant values thresh hold of train error
        :param val_err_threshold: constant values thresh hold of validation error
        :param num_partial_training: The number of epochs each times model train call is a strip
        :param verbose: If True print out the error for each epoch.
        :param random_seed: random seed for regeneration of results.
        '''
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.train_err_threshold = train_err_threshold
        self.val_err_threshold = val_err_threshold
        self.num_partial_trianing = num_partial_training
        np.random.seed(random_seed)



    def create_X_y_data(self, dataset, target_col, sub_features):
        """
        Method create X, y dataframe from dataset by select features in sub_features.

        :param dataset: dataframe of dataset
        :param target_col: Column label name in dataframe dataset
        :param  sub_features: a subset feature solution
        :return X.T = (num_of_features, num_of_samples), y.T = (num_of_classes, num_of_sample)        
        """
        y = dataset[target_col].values
        y = to_categorical(y)
        features = dataset.columns.tolist()
        dropped_col = [features[i] for i in range(len(sub_features)) if sub_features[i] == 0]
        dropped_col.append(target_col)
        X = dataset.drop(columns=dropped_col).to_numpy()
        return X.T, y.T

    # build model
    def define_structure(self, X, y, n_hidden_unit):
        """
        Method define construct of model.

        :param X = (num_of_features, num_of_samples)
        :param y = (num_of_classes, num_of_sample)   
        :param n_hidden_unit: number of hidden unit in hidden layer
        :return (input_unit, hidden_unit, output_unit)    
        """
        input_unit = X.shape[0]  # size of input layer
        hidden_unit = n_hidden_unit 
        output_unit = y.shape[0]  # size of output layer
        return (input_unit, hidden_unit, output_unit)

    def add_hidden_unit(self, parameters, input_unit, output_unit):
        """
        Method add one unit to hidden layers.

        :param parameters = {"W1": W1,
                              "b1": b1,
                              "W2": W2,
                              "b2": b2}
                            dictionary parameters of model.
        :param input_unit : size of input layer.   
        :param output_unit: size of output layer.
        :return parameters.   
        """
        
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # add hidden units
        tp = np.random.rand(1, input_unit) * 2 - 1
        W1 = np.append(W1, tp, axis=0)
        b1 = np.append(b1, np.zeros((1, 1)), axis=0)
        tp = np.random.rand(output_unit, 1) * 2 - 1
        W2 = np.append(W2, tp, axis=1)

        # update parameters
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def del_hidden_unit(self, parameters, input_unit, output_unit):
        """
       Method delete one unit to hidden layers.

       :param parameters = {"W1": W1,
                             "b1": b1,
                             "W2": W2,
                             "b2": b2}
                           dictionary parameters of model.
       :param input_unit : size of input layer.   
       :param output_unit: size of output layer.
       :return parameters.   
       """
        
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # delete hidden units
        W1 = np.delete(W1, -1, axis=0)
        b1 = np.delete(b1, -1, axis=0)
        W2 = np.delete(W2, -1, axis=1)

        # update parameters
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def add_hidden_unit_momentum(self, parameters, input_unit, output_unit):
        """
       Method add one unit to hidden layers of model using momentum.

       :param parameters = {"W1": W1,
                             "b1": b1,
                             "W2": W2,
                             "b2": b2,
                             'theta': theta}
                           dictionary parameters of model.
       :param input_unit : size of input layer.   
       :param output_unit: size of output layer.
       :return parameters.   
       """
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        theta = parameters['theta']

        # add hidden units
        tp = np.random.rand(1, input_unit) * 2 - 1
        W1 = np.append(W1, tp, axis=0)
        b1 = np.append(b1, np.zeros((1, 1)), axis=0)
        tp = np.random.rand(output_unit, 1) * 2 - 1
        W2 = np.append(W2, tp, axis=1)

        # update theta
        theta['theta_w1'] = np.append(theta['theta_w1'], np.zeros((1, input_unit)), axis=0)
        theta['theta_b1'] = np.append(theta['theta_b1'], np.zeros((1, 1)), axis=0)
        theta['theta_w2'] = np.append(theta['theta_w2'], np.zeros((output_unit, 1)), axis=1)

        # update parameters
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      'theta': theta}
        return parameters

    def del_hidden_unit_momentum(self, parameters, input_unit, output_unit):
        """
       Method delete one unit to hidden layers of model using momentum.

       :param parameters = {"W1": W1,
                             "b1": b1,
                             "W2": W2,
                             "b2": b2,
                             'theta': theta}
                           dictionary parameters of model.
       :param input_unit : size of input layer.   
       :param output_unit: size of output layer.
       :return parameters.   
       """
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        theta = parameters['theta']

        # delete hidden units
        W1 = np.delete(W1, -1, axis=0)
        b1 = np.delete(b1, -1, axis=0)
        W2 = np.delete(W2, -1, axis=1)

        # update theta
        theta['theta_w1'] = np.delete(theta['theta_w1'], -1, axis=0)
        theta['theta_b1'] = np.delete(theta['theta_b1'], -1, axis=0)
        theta['theta_w2'] = np.delete(theta['theta_w2'], -1, axis=1)
        # update parameters
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "theta": theta}

        return parameters

    def parameters_initialization_momentum(self, input_unit, hidden_unit, output_unit):
        """
        Method initialize parameters for model using momentum.

        :param input_unit : size of input layer.
        :param hidden_unit : size of hidden layer.
        :param output_unit: size of output layer.
        :return parameters.
        """
        #     np.random.seed(42)
        theta_w1 = np.zeros((hidden_unit, input_unit))
        theta_b1 = np.zeros((hidden_unit, 1))
        theta_w2 = np.zeros((output_unit, hidden_unit))
        theta_b2 = np.zeros((output_unit, 1))
        theta = {'theta_w1': theta_w1, 'theta_b1': theta_b1, 'theta_w2': theta_w2, 'theta_b2': theta_b2}

        W1 = np.random.rand(hidden_unit, input_unit) * 2 - 1
        b1 = np.zeros((hidden_unit, 1))
        W2 = np.random.rand(output_unit, hidden_unit) * 2 - 1
        b2 = np.zeros((output_unit, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "theta": theta}

        return parameters

    def sigmoid(self, Z):
        """
        Method compute sigmoid for value z.
        """
        return 1 / (1 + np.exp(-Z))

    def dSigmoid(self, Z):
        """
        Method compute sigmoid deviation for value z.
        """
        s = 1 / (1 + np.exp(-Z))
        dZ = s * (1 - s)
        return dZ
    def softmax(self, Z):
        """
        Method compute softmax values for Z
        """
        e_Z = np.exp(Z)
        A = e_Z / e_Z.sum(axis=0)
        return A

    def forward_propagation(self, X, parameters):
        """
        Method compute forward propagation.

        :param X: data samples
        :param parameters: parameters of model.
        :return A2: output of forward propagation, cache= {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}.
        """
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.softmax(Z2)
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

        return A2, cache


    def backward_propagation(self, parameters, cache, X, Y):
        """
        Method compute backword propagation.

        :param X: data samples
        :param Y: data labels
        :param parameters: parameters of model.
        :param cache: values compute from forward propagation
        :return grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}.
        """

        # number of training example
        m = X.shape[1]

        W1 = parameters['W1']
        W2 = parameters['W2']
        A1 = cache['A1']
        A2 = cache['A2']

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), self.dSigmoid(A1))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        return grads

    def loss_function(self, Y, Yhat):
        '''
        Method comput loss value.
        :param Y: True labels.
        :param Yhat: Predict labels.
        :return: Error value.
        '''
        return 1 / 2 * np.sum(np.square(Y - Yhat))

    def gradient_descent(self, parameters, grads):
        '''
        Method comput gradient descent.

        :param parameters: parameters of model.
        :param grads: deviation of parameters compute from backward propagation
        :return: parameters
        '''
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        W1 = W1 - self.learning_rate * dW1
        b1 = b1 - self.learning_rate * db1
        W2 = W2 - self.learning_rate * dW2
        b2 = b2 - self.learning_rate * db2

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

        return parameters

    def gd_momentum(self, parameters, grads):
        '''
        Method comput gradient descent has use momentum.

        :param parameters: parameters of model.
        :param grads: deviation of parameters compute from backward propagation
        :return: parameters
        '''
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        theta = parameters["theta"]

        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        theta['theta_w1'] = self.momentum * theta['theta_w1'] + self.learning_rate * dW1
        W1 = W1 - theta['theta_w1']
        theta['theta_b1'] = self.momentum * theta['theta_b1'] + self.learning_rate * db1
        b1 = b1 - theta['theta_b1']
        theta['theta_w2'] = self.momentum * theta['theta_w2'] + self.learning_rate * dW2
        W2 = W2 - theta['theta_w2']
        theta['theta_b2'] = self.momentum * theta['theta_b2'] + self.learning_rate * db2
        b2 = b2 - theta['theta_b2']

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "theta": theta}

        return parameters

    def prediction(self, X, y, parameters):
        '''
        Method to predict the targel label
        :param X: test examples
        :param y: test labels
        :param parameters: parameters of model.
        :return: Test accuracy.
        '''
        A2, cache = self.forward_propagation(X, parameters)
        predicted_class = np.argmax(A2, axis=0)
        true_class = np.argmax(y, axis=0)
        acc = np.mean(predicted_class == true_class)
        return acc

    def check_3_decrease(self, list_values, threshold):
        '''
        Method check if list value has 3 last value decrease or not.

        :param list_values: list of values
        :param threshold: thresh hold
        :return: boolean
        '''
        i = len(list_values) - 1
        if i < 4:
            return False
        if (list_values[i] - list_values[i - 1]) < threshold and (list_values[i] - list_values[i - 2]) < threshold and (
                list_values[i] - list_values[i - 3]) < threshold:
            return True
        return False


    def check_3_increase(self, list_values, threshold):
        '''
        Method check if list value has 3 last value increase or not.

        :param list_values: list of values
        :param threshold: thresh hold
        :return: boolean
        '''
        i = len(list_values) - 1
        if i < 4:
            return False
        if (list_values[i] - list_values[i - 1]) > threshold and (list_values[i] - list_values[i - 2]) > threshold and (
                list_values[i] - list_values[i - 3]) > threshold:
            return True
        return False

    def check_err_train(self, list_values, threshold):
        '''
        Method check if list value has 2 last value decrease or not.

        :param list_values: list of values
        :param threshold: thresh hold
        :return: boolean
        '''
        i = len(list_values) - 1
        if i < 2:
            return True
        if (list_values[i - 1] - list_values[i]) >= threshold:
            return True
        return False


    def train_partial(self, X_train, y_train, parameters):
        '''
        Method train neural network in constant epochs
        :param X_train: train samples
        :param y_train: train labels
        :param parameters: parameters of model
        :return: parameters , train_loss
        '''
        for i in range(self.num_partial_trianing):
            A2, cache = self.forward_propagation(X_train, parameters)
            train_loss = self.loss_function(y_train, A2)
            grads = self.backward_propagation(parameters, cache, X_train, y_train)
            parameters = self.gd_momentum(parameters, grads)

        return parameters, train_loss

    def classification(self, df_train, df_val, df_test, sub_features, target_col, hidden_unit=1):
        '''
        Method training model .

        :param df_train: dataframe of training data
        :param df_val: dataframe of validation data
        :param df_test: dataframe of test data
        :param sub_features: a subset features solution
        :param target_col: column name of target
        :param hidden_unit: number of unit in hidden layer
        :return: test accuracy
        '''
        # -------------------- split data : train , validation , test ------------------
        X_train, y_train = self.create_X_y_data(df_train, target_col, sub_features)
        X_val, y_val = self.create_X_y_data(df_val, target_col, sub_features)
        X_test, y_test = self.create_X_y_data(df_test, target_col, sub_features)

        i_train = -1
        i_val = -1
        err_train = list()
        err_val = list()
        ca_val = list()
        # Initial model
        units = self.define_structure(X_train, y_train, hidden_unit)
        input_unit = units[0]
        output_unit = units[2]
        parameters = self.parameters_initialization_momentum(input_unit, hidden_unit, output_unit)
        # print(parameters)
        for i in range(2):
            parameters, train_loss = self.train_partial(X_train, y_train, parameters)
            # add train err to err_train
            err_train.append(train_loss)
            i_train += 1
            # add val error and val acc to err_val , ca_val
            A2_val, cache_val = self.forward_propagation(X_val, parameters)
            val_loss = self.loss_function(y_val, A2_val)
            val_acc = self.prediction(X_val, y_val, parameters)
            err_val.append(val_loss)
            ca_val.append(val_acc)
            i_val += 1

        while (self.check_3_increase(err_val, self.val_err_threshold) == False
               or self.check_3_decrease(ca_val, 0) == False) and hidden_unit < 20:

            # check training criterion
            if self.check_err_train(err_train, self.train_err_threshold):
                parameters, train_loss = self.train_partial(X_train, y_train, parameters)
                # add train_err to err_train
                err_train.append(train_loss)
                i_train += 1
            # add hidden units
            else:
                if ca_val[i_val] <= ca_val[i_val - 1]:
                    parameters_mem = copy.deepcopy(parameters)
                    parameters = self.add_hidden_unit_momentum(parameters, input_unit, output_unit)
                    hidden_unit += 1
                    # print("Number of hidden units = {}".format(hidden_unit))
                    parameters, train_loss = self.train_partial(X_train, y_train, parameters)
                    # add train err to err_train
                    err_train.append(train_loss)
                    i_train += 1
                    # check if add hidden units not improve validation accuracy then restore before parameters
                    A2_val, cache_val = self.forward_propagation(X_val, parameters)
                    tmp_val_acc = self.prediction(X_val, y_val, parameters)
                    if tmp_val_acc < ca_val[i_val]:
                        parameters = copy.deepcopy(parameters_mem)
                        hidden_unit -= 1
                        # print("Number of hidden units = {}".format(hidden_unit))
                        parameters, train_loss = self.train_partial(X_train, y_train, parameters)
                        # add val error and val acc to err_val , ca_val
                        A2_val, cache_val = self.forward_propagation(X_val, parameters)
                        val_loss = self.loss_function(y_val, A2_val)
                        val_acc = self.prediction(X_val, y_val, parameters)
                        err_val.append(val_loss)
                        ca_val.append(val_acc)
                        i_val += 1
                        # print("Iteration {}: Train_loss = {}, Val_loss = {}, Val_acc = {}"
                        #       .format(i_val, train_loss, val_loss, val_acc))
                        continue
                    # reset val_acc, val_err
                    err_val = list()
                    ca_val = list()
                    err_train = list()
                    i_val = -1
            # add val error and val acc to err_val , ca_val
            A2_val, cache_val = self.forward_propagation(X_val, parameters)
            val_loss = self.loss_function(y_val, A2_val)
            val_acc = self.prediction(X_val, y_val, parameters)
            err_val.append(val_loss)
            ca_val.append(val_acc)
            i_val += 1
            # print("Iteration {}: Train_loss = {}, Val_loss = {}, Val_acc = {}"
            #       .format(i_val, train_loss, val_loss, val_acc))
        acc = self.prediction(X_test, y_test, parameters)
        print("Test acc = {}".format(acc))
        return acc
