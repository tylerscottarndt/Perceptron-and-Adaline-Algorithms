import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Perceptron import*
from AdalineGD import*

td = pd.read_csv('dataset_linear.csv')
linear_criteria_1 = td.iloc[0:, 0]
linear_criteria_2 = td.iloc[0:, 1]

td2 = pd.read_csv('dataset_nonlinear.csv')
nonlinear_criteria_1 = td2.iloc[0:, 0]
nonlinear_criteria_2 = td2.iloc[0:, 1]

learning_rate = 0.1
p = Perceptron()


def __populate_samples(arr1, arr2):
    samples = []
    i = 0
    while i < len(arr1):
        samples.append([arr1[i], arr2[i]])
        i += 1
    return samples


def __run_perceptron(p, t, arr1, arr2):
    samples = __populate_samples(arr1, arr2)
    targets = t.iloc[0:, 2]
    p.fit(samples, targets)


def __generate_user_graph(arr1, arr2):
    plt.plot(arr1[:8], arr2[:8], 'ro', label="-1")
    plt.plot(arr1[8:], arr2[8:], 'b^', label="1")
    plt.title("Self-Created Data Set", fontsize=15)
    plt.xlabel("Criteria A")
    plt.ylabel("Criteria B")
    plt.legend(loc='upper right')
    plt.show()


def __graph_epoch_error_rate(p, t, arr1, arr2):
    __run_perceptron(p, t, arr1, arr2)
    weight_updates = p.get_number_updates()

    plt.plot(weight_updates, marker=".")
    plt.title("Self-Created Data Set", fontsize=15)
    plt.xlabel("Epochs")
    plt.ylabel("Number of Updates")
    plt.show()


def run_homework_problem_1():
    __generate_user_graph(linear_criteria_1, linear_criteria_2)
    __graph_epoch_error_rate(p, td, linear_criteria_1, linear_criteria_2)


def run_homework_problem_2():
    __generate_user_graph(nonlinear_criteria_1, nonlinear_criteria_2)
    __graph_epoch_error_rate(p, td2, nonlinear_criteria_1, nonlinear_criteria_2)

# run_homework_problem_1()
# run_homework_problem_2()
