import matplotlib.pyplot as plt
import numpy as np

from student import EFCStudent
from item import Item, Material


def generate_random_batch(batch_size, item_count):
    counts = [0] * item_count
    for _ in range(batch_size):
        idx = np.random.randint(0, item_count)
        counts[idx] += 1
    return counts


item_count = 10
items = list()
item_probs = list()
for _ in range(item_count):
    items.append(Item())
    item_probs.append(list())
material = Material(items)
student = EFCStudent('leitner')

simulation_length = 100
questions_per_session = 20
for day in range(1, simulation_length + 1):
    dist = generate_random_batch(questions_per_session, item_count)
    for item, question_no in zip(items, dist):
        for _ in range(question_no):
            student.review_item(item, day)

    for idx, item in enumerate(items):
        try:
            item_prob = student.calculate_recall_likelihood(item, day)
            item_probs[idx].append(item_prob)
        except KeyError:
            item_probs[idx].append(0.0)

    success = True
    for item_prob in item_probs:
        if item_prob[-1] < 0.98:
            success = False
            break

    if success:
        print("Final")
        print(day)
        # break

# print(item_probs)
for i_probs in item_probs:
    plt.plot(i_probs)
plt.show()
