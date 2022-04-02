from tensorflow.keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
class NeuralNetwork(object):

    def __init__(self, learning_rate=0.1, momentum=0.9, t_stop=3, train_err_threshold=0.003, val_err_threshold=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.t_stop = t_stop
        self.train_err_threshold = train_err_threshold
        self.val_err_threshold = val_err_threshold

    def create_model(self, n_input, n_classes, n_unit=1):
        # define model
        model = Sequential()
        model.add(Dense(n_unit, input_dim=n_input, activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense(n_classes, activation='softmax'))
        # compile model
        opt = SGD(lr=self.learning_rate, momentum=self.momentum)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # return True : if list l have 3 last value decrease
    def check_3_decrease(self, list_values):
        i = len(list_values) - 1
        if (list_values[i] - list_values[i - 1]) < 0 and (list_values[i] - list_values[i - 2]) < 0:
            return True
        return False

    # return True : if list l have 3 last value increase
    def check_3_increase(self, list_values):
        i = len(list_values) - 1
        if (list_values[i] - list_values[i - 1]) > 0 and (list_values[i] - list_values[i - 2]) > 0:
            return True
        return False

    def classification_paper(self, df_train, df_val, df_test, sub_features, target_col):
        # -------------------- split data : train , validation , test ------------------
        features = df_train.columns.tolist()
        dropped_col = [features[i] for i in range(len(sub_features)) if sub_features[i] == 0]
        dropped_col.append(target_col)
        # select training data
        X_train = df_train.drop(columns=dropped_col).to_numpy()
        y_train = df_train[target_col].values
        y_train = to_categorical(y_train)
        # select validation data
        X_val = df_val.drop(columns=dropped_col).to_numpy()
        y_val = df_val[target_col].values
        y_val = to_categorical(y_val)
        # select test data
        X_test = df_test.drop(columns=dropped_col).to_numpy()
        y_test = df_test[target_col].values
        y_test = to_categorical(y_test)

        # configure the model based on the data
        n_input, n_classes = X_train.shape[1], y_train.shape[1]
        n_unit = 1

        # ------------------------------- training model ----------------
        i_train = -1
        i_val = -1
        err_train = list()
        err_val = list()
        ca_val = list()
        # define model
        model = self.create_model(n_input=n_input, n_classes=n_classes, n_unit=n_unit)
        # partial training for t = 20 epochs
        for ind in range(2):
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, verbose=0)
            # add train_err to err_train
            train_loss = history.history['loss'][-1]
            err_train.append(train_loss)
            i_train += 1
            # add val error and val_acc to err_val
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_accuracy'][-1]
            err_val.append(val_loss)
            ca_val.append(val_acc)
            i_val += 1
            # Criterion for continue training : error don't increase in 3 t and acc don't decrease in 3 t
        while (i_val < self.t_stop) or (self.check_3_increase(err_val) == False and self.check_3_decrease(ca_val) == False):

            # check training criterion
            if err_train[i_train - 1] - err_train[i_train] > self.train_err_threshold:
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, verbose=0)
                # add train_err to err_train
                train_loss = history.history['loss'][-1]
                err_train.append(train_loss)
                i_train += 1
            # add hidden units
            else:
                if ca_val[i_val] <= ca_val[i_val - 1]:
                    n_unit += 1
                    model = self.create_model(n_input=n_input, n_classes=n_classes, n_unit=n_unit)
                    # partial training for t = 20 epochs
                    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, verbose=0)
                    # add train_err to err_train
                    train_loss = history.history['loss'][-1]
                    err_train.append(train_loss)
                    i_train += 1
            # add val error and val_acc to err_val
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_accuracy'][-1]
            err_val.append(val_loss)
            ca_val.append(val_acc)
            i_val += 1
        print("iterator = {}".format(i_val))
        # evaluate model on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        return test_acc

    def create_X_y_data(self,dataset, target_col, sub_features):
        y = dataset[target_col].values
        y = to_categorical(y)
        features = dataset.columns.tolist()
        dropped_col = [features[i] for i in range(len(sub_features)) if sub_features[i] == 0]
        dropped_col.append(target_col)
        X = dataset.drop(columns=dropped_col).to_numpy()
        return X, y

    def classification(self, df_train, df_val, df_test, sub_features, target_col, verbose = 0):
        # -------------------- split data : train , validation , test ------------------
        X_train, y_train = self.create_X_y_data(df_train, target_col, sub_features)
        X_val, y_val = self.create_X_y_data(df_val, target_col, sub_features)
        X_test, y_test = self.create_X_y_data(df_test, target_col, sub_features)
        n_input, n_classes = X_train.shape[1], y_train.shape[1]
        # define model
        model = Sequential()
        model.add(Dense(16, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(n_classes, activation='softmax'))
        # compile model
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=20,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True)
        # fit model on train set
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, verbose=verbose, batch_size=32,
                            callbacks=[early_stopping_monitor])
        loss, test_acc = model.evaluate(X_test, y_test, verbose=verbose)
        return test_acc
