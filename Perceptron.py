class Perceptron(object):
    learning_rate: float
    iterations: int
    threshold: int
    errors_by_iteration = []

    def __init__(self):
        self.learning_rate = 0.1
        self.iterations = 25
        self.threshold = 4

    def fit(self, samples, target_values):
        weights = [0.1, 0.1]    # load initial small weights

        for _ in range(self.iterations):
            errors = 0
            i = 0
            while i < len(samples):
                net_input = samples[i][0] * weights[0] + samples[i][1] * weights[1] - self.threshold

                if net_input >= 0:
                    prediction = 1
                else:
                    prediction = -1

                if prediction != target_values[i]:
                    weights[0] = weights[0] + self.learning_rate * (target_values[i] - prediction) * samples[i][0]
                    weights[1] = weights[1] + self.learning_rate * (target_values[i] - prediction) * samples[i][1]
                    errors += 1

                i += 1
            self.errors_by_iteration.append(errors)

    def get_number_updates(self):
        return self.errors_by_iteration
