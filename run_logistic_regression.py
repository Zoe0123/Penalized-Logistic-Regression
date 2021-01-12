from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    # For mnist_train
    hyperparameters = { 
        "learning_rate": 0.01,           
        "weight_regularization": 0.,
        "num_iterations": 800
    }
    # For mnist_train_small
    # hyperparameters = { 
    #     "learning_rate": 0.008,           
    #     "weight_regularization": 0.,
    #     "num_iterations": 1000
    # }
    weights = np.zeros((M+1, 1))

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
   
    # list of training and validation errors and cross entropy
    train_errors, valid_errors = [], []
    train_ces, valid_ces = [], []

    iterations = hyperparameters["num_iterations"]
    for t in range(iterations):
        # perfrom logistic regression
        df = logistic(weights, train_inputs, train_targets, hyperparameters)[1]
        weights = weights - hyperparameters["learning_rate"] * df
        
        # compute and record train and validation error and cross entropy
        train_y = logistic_predict(weights, train_inputs)
        train_ce, train_accu = evaluate(train_targets, train_y)
        train_errors.append(1 - train_accu)
        train_ces.append(train_ce)

        valid_y = logistic_predict(weights, valid_inputs)
        valid_ce, valid_accu = evaluate(valid_targets, valid_y)
        valid_errors.append(1 - valid_accu)
        valid_ces.append(valid_ce)
    
    # 2.2 b) / c) cross entropy plot for training set and validation set
    plt.figure(1)
    plt.plot(range(iterations), train_ces, '-b')
    plt.plot(range(iterations), valid_ces, '-g')
    plt.xlabel('trainig iterations')
    plt.ylabel('cross-entropy cost')
    plt.legend(['training cross-entropy cost', 'validation cross-entropy cost'])
    plt.title('Cross-entropy cost for each iteration')
    plt.savefig('./2.2 c). cross entropy plot for training set and validation set.png')

    # 2.2 b) errors plot for training set and validation set
    plt.figure(2)
    plt.plot(range(iterations), train_errors, '-b')
    plt.plot(range(iterations), valid_errors, '-g')
    plt.xlabel('trainig iterations')
    plt.ylabel('errors')
    plt.legend(['training errors', 'validation errors'])
    plt.title('Errors for each iteration')

    print('The final cross entropy and error on training set is {}, {}'.format(train_ce, 1 - train_accu))
    print('The final cross entropy and error on validation set is {}, {}'.format(valid_ce, 1 - valid_accu))

    # 2.2 b) cross entropy and error on test set (with best hyperparameter)
    test_inputs, test_targets = load_test()
    test_y = logistic_predict(weights, test_inputs)
    test_ce, test_accu = evaluate(test_targets, test_y)

    print('The final cross entropy and error on test set is {}, {}'.format(test_ce, 1 - test_accu))


def run_pen_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    # For mnist_train dataset
    hyperparameters = { 
        "learning_rate": 0.15,           
        "weight_regularization": 0.,
        "num_iterations": 1000
    }

    # For mnist_train_small dataset
    # hyperparameters = { 
    #     "learning_rate": 0.01,           
    #     "weight_regularization": 0.,
    #     "num_iterations": 400
    # }

    lambd_list = [0, 0.001, 0.01, 0.1, 1.0]

    for lambd in lambd_list:
        hyperparameters["weight_regularization"] = lambd
        
        train_errors, valid_errors = [], []
        train_ces, valid_ces = [], []

        select = 4   # select one run to plot ce curves
        t_ces, v_ces = [], [] # list of ce for the selected run

        iterations = hyperparameters["num_iterations"]

        for run in range(5): 
            # reinitialize weights for each run
            weights = np.zeros((M+1, 1))
            for t in range(iterations): 
                # perform penalized logistic regression
                df = logistic_pen(weights, train_inputs, train_targets, hyperparameters)[1]
                weights = weights - hyperparameters["learning_rate"] * df
        
                # compute train and validation ce and accuracy
                train_y = logistic_predict(weights, train_inputs)
                train_ce, train_accu = evaluate(train_targets, train_y)
                # add penalized term to ce
                train_ce += lambd/2 * sum(np.square(weights[:-1]))
                # record ce at selected run
                if run == select:
                    t_ces.append(train_ce)

                valid_y = logistic_predict(weights, valid_inputs)
                valid_ce, valid_accu = evaluate(valid_targets, valid_y)
                valid_ce += lambd/2 * sum(np.square(weights[:-1]))  
                if run == select:
                    v_ces.append(valid_ce)
            
            # record final train and validation ce and errors for each run
            train_errors.append(1- train_accu)
            train_ces.append(train_ce)

            valid_errors.append(1-valid_accu)
            valid_ces.append(valid_ce)
        
            # plot ce curves for selected run
            if run == select:
                plt.figure()
                plt.plot(range(iterations), t_ces, '-b')
                plt.plot(range(iterations), v_ces, '-g')
                plt.xlabel('trainig iterations')
                plt.ylabel('cross-entropy cost')
                plt.ylim(0, 1)
                plt.legend(['training cross-entropy cost', 'validation cross-entropy cost'])
                plt.title('lamda {}: Cross-entropy cost at each iteration'.format(lambd))
                plt.savefig(('./2.3 b).lamda {}: Cross-entropy cost at each iteration.png'.format(lambd)))
        
        # calculate average final errors and ce
        avg_t_error, avg_t_ce = np.mean(train_errors), np.mean(train_ces)
        avg_v_error, avg_v_ce = np.mean(valid_errors), np.mean(valid_ces)  
        
        print('for lambda {}, The average final error and ce on training set is {}, {}'.format(lambd, avg_t_error, avg_t_ce))
        print('for lambda {}, The average final error and ce on validation set is {}, {}'.format(lambd, avg_v_error, avg_v_ce))

    # 2.3 c) test ce and classification rate for the best lambd (best lambd is 0.1 from experiemnts)
    lambd = 0.1
    hyperparameters["weight_regularization"] = lambd
    weights = np.zeros((M+1, 1))
    for t in range(hyperparameters["num_iterations"]): 
        df = logistic_pen(weights, train_inputs, train_targets, hyperparameters)[1]
        weights = weights - hyperparameters["learning_rate"] * df
       
    test_inputs, test_targets = load_test()

    test_y = logistic_predict(weights, test_inputs)
    test_ce, test_accu = evaluate(test_targets, test_y)
    test_ce += lambd/2 * sum(np.square(weights[:-1])) #?

    print('test ce and classification rate for the best value of lambd is {}, {}'.format(test_ce, test_accu)) # [test_ce]?


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    run_pen_logistic_regression()
