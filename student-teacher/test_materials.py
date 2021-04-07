import numpy as np

from item import Item, Material


test1_items = list()
for _ in range(10):
    test1_items.append(Item())
TEST1 = Material(test1_items)


test2_items = list()
for _ in range(10):
    test2_items.append(Item(np.random.uniform(0.6, 0.8)))
TEST2 = Material(test2_items)


test3_items = list()
for _ in range(10):
    test3_items.append(Item(np.random.uniform(0.35, 0.45)))
TEST3 = Material(test3_items)
