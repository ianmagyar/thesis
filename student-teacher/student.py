from collections import deque

import numpy as np


class Record:
    def __init__(self, item, time_step, n=5):
        self.item = item
        self.last_checked = time_step
        self.no_attempts = 1
        self.correct_recalls = 1
        self.incorrect_recalls = 0
        self.last_n = deque(maxlen=n)
        self.leitner = 1

        self.decay = 0.9
        self.weights = [self.decay ** i for i in range(n)]
        self.weights.reverse()

    def update(self, time_step, success):
        self.no_attempts += 1
        self.last_n.append(success)
        if time_step != self.last_checked:
            self.correct_recalls += success
            if success:
                self.leitner += 1
        self.last_checked = time_step
        self.incorrect_recalls += not success
        if not success:
            self.leitner = max(1, self.leitner - 1)

    def get_record_strength(self):
        # if len(self.last_n) == 0:
        #     return 1.0
        # return sum(self.last_n) / len(self.last_n)
        try:
            weighted_sum = np.dot(self.last_n, self.weights[-len(self.last_n):])
            return weighted_sum / len(self.weights)
        except ValueError:
            return 1.0

    def get_strength(self):
        return self.no_attempts

    def get_correct_recalls(self):
        return self.correct_recalls

    def get_leitner(self):
        return self.leitner


class EFCStudent:
    def __init__(self, memory_type):
        self.memory = dict()
        self.constant = 1
        self.memory_type = memory_type

    def __learn_item(self, item, time_step):
        item_record = Record(item, time_step)
        self.memory[item] = item_record

    def get_item_strength(self, item_record):
        if self.memory_type == 'strength':
            return item_record.get_strength()
        if self.memory_type == 'leitner':
            return item_record.get_leitner()
        if self.memory_type == 'correct':
            return item_record.get_correct_recalls()

    def calculate_recall_likelihood(self, item, time_step):
        try:
            item_record = self.memory[item]
        except KeyError:
            return 1 - item.difficulty
        elapsed_time = 1.0
        elapsed_time *= self.constant
        if item_record.no_attempts == 0:
            recall_likelihood = 1 - item.difficulty
        else:
            item_strength = self.get_item_strength(item_record)
            recall_likelihood = np.exp(
                -item_record.item.difficulty * elapsed_time / item_strength)

        # record_strength = item_record.get_record_strength()
        # recall_likelihood *= record_strength

        return recall_likelihood

    def __update_record(self, item, time_step, success):
        item_record = self.memory[item]
        item_record.update(time_step, success)

    def review_item(self, item, time_step):
        if item not in self.memory:
            self.__learn_item(item, time_step)

        recall_likelihood = self.calculate_recall_likelihood(item, time_step)
        success = np.random.random() < recall_likelihood
        if success:
            answer = item
        else:
            answer = item.get_wrong_answer()

        self.__update_record(item, time_step, success)
        return answer

    def get_performance(self, items, time_step, test_per_item):
        results = np.zeros((len(items), len(items)), dtype='int')

        for item_id, item in enumerate(items):
            if item not in self.memory:
                self.__learn_item(item, time_step)

            for _ in range(test_per_item):
                recall_likelihood = self.calculate_recall_likelihood(
                    item, time_step)
                success = np.random.random() < recall_likelihood
                if success:
                    answer = item
                else:
                    answer = item.get_wrong_answer()

                answer_id = items.index(answer)
                results[item_id][answer_id] += 1

        return results
