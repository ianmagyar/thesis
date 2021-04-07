import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from schedulers import *
from student import EFCStudent
from test_materials import *


def train_with_random(items, student_type, constant, runs, simulation_length, questions_per_session, evaluation_per_item):
    item_count = len(items)
    item_probs_history = np.zeros((runs, item_count, simulation_length))
    scheduler = RandomScheduler(item_count)
    for run in tqdm(range(runs)):
        student = EFCStudent(student_type)
        student.constant = constant
        for day in range(1, simulation_length + 1):
            for idx, item in enumerate(items):
                item_prob = student.calculate_recall_likelihood(item, day)
                # print(day, idx, item_prob)
                item_probs_history[run, idx, day - 1] = item_prob

            student_performance = student.get_performance(
                items, day, evaluation_per_item)
            dist = scheduler.generate(student_performance, questions_per_session)
            # print(dist)

            for item, question_no in zip(items, dist):
                for _ in range(question_no):
                    student.review_item(item, day)

            check = item_probs_history[:, :, -1] > .98
            # if check.all():
            #     break

    if verbose:
        points = np.apply_over_axes(np.mean, item_probs_history, (0, 1))[0][0]
        plt.plot(points)
        plt.show()

    filename = "c{}_t{}_random_{}_runs_{}_items_{}_days_{}_student".format(
        constant, test, runs, item_count, simulation_length, student_type
    )
    np.save("save_{}.npy".format(filename), item_probs_history)

    item_file = open("items_{}.csv".format(filename), 'w')
    for item in items:
        item_file.write("{}\n".format(item.difficulty))
    item_file.close()

    param_file = open("params_{}.log".format(filename), 'w')
    param_file.write("scheduler={}\n".format(
        "random" if isinstance(scheduler, RandomScheduler) else "adaptive"))
    param_file.write("student={}\n".format(student_type))
    param_file.write("questions_per_session={}\n".format(questions_per_session))
    param_file.write("evaluation_per_item={}\n".format(evaluation_per_item))
    param_file.close()


def train_with_adaptive(items, student_type, constant, runs, simulation_length, questions_per_session, evaluation_per_item):
    item_count = len(items)
    item_probs_history = np.zeros((runs, item_count, simulation_length))
    for run in tqdm(range(runs)):
        scheduler = AdaptiveScheduler(item_count)
        student = EFCStudent(student_type)
        student.constant = constant
        for day in range(1, simulation_length + 1):
            for idx, item in enumerate(items):
                item_prob = student.calculate_recall_likelihood(item, day)
                item_probs_history[run, idx, day - 1] = item_prob

            student_performance = student.get_performance(
                items, day, evaluation_per_item)
            pre_acc = student_performance.trace() / student_performance.sum()
            action, dist = scheduler.generate(student_performance, questions_per_session)
            # print(dist)

            for item, question_no in zip(items, dist):
                for _ in range(question_no):
                    student.review_item(item, day)

            post_performance = student.get_performance(
                items, day, evaluation_per_item)
            new_acc = post_performance.trace() / post_performance.sum()

            if pre_acc != 1.0:
                reward = (new_acc - pre_acc) / (1 - pre_acc)
            else:
                reward = 0.0

            scheduler.process(student_performance, action, reward, post_performance)

            check = item_probs_history[:, :, -1] > .98
            # if check.all():
            #     break

    filename = "c{}_t{}_adaptive_{}_runs_{}_items_{}_days_{}_student".format(
        constant, test, runs, item_count, simulation_length, student_type
    )
    np.save("save_{}.npy".format(filename), item_probs_history)

    item_file = open("items_{}.csv".format(filename), 'w')
    for item in items:
        item_file.write("{}\n".format(item.difficulty))
    item_file.close()

    param_file = open("params_{}.log".format(filename), 'w')
    param_file.write("scheduler={}\n".format(
        "random" if isinstance(scheduler, RandomScheduler) else "adaptive"))
    param_file.write("student={}\n".format(student_type))
    param_file.write("questions_per_session={}\n".format(questions_per_session))
    param_file.write("evaluation_per_item={}\n".format(evaluation_per_item))
    param_file.close()

    if verbose:
        points = np.apply_over_axes(np.mean, item_probs_history, (0, 1))[0][0]
        # print(points)
        plt.plot(points)
        plt.show()


for c in [2, 3, 5]:
    print("DELAY CONSTANT", c)
    for test, material in enumerate([TEST1, TEST2, TEST3], start=1):
        for student_type in ['leitner', 'strength', 'correct']:
            print("testing T{}, {}".format(test, student_type))
            verbose = False

            # material = TEST1
            items = material.items
            item_count = material.item_count

            # student_type = 'leitner'

            simulation_length = 30
            runs = 1000
            questions_per_session = 6
            evaluation_per_item = 5


            train_with_random(items, student_type, c, runs, simulation_length, questions_per_session, evaluation_per_item)
            train_with_adaptive(items, student_type, c, runs, simulation_length, questions_per_session, evaluation_per_item)
