import numpy as np


class Item:
    def __init__(self, difficulty=None):
        if difficulty is None:
            center = 0.9
            dev = 0.1
            difficulty = np.random.uniform(
                low=max(0.0, center - dev),
                high=min(1.0, center + dev)
            )
        self.difficulty = difficulty

    def added_to_material(self, material):
        self.material = material

    def get_similarities(self):
        return self.material.get_similarities_for_item(self)

    def get_wrong_answer(self):
        return self.material.get_wrong_answer_for_item(self)


class Material:
    def __init__(self, items, similarities=None):
        if len(items) == 0:
            raise ValueError("No items specified")

        self.items = items
        self.item_count = len(self.items)

        for item in self.items:
            item.added_to_material(self)

        if similarities is not None:
            if similarities.shape[0] != len(items) != similarities.shape[1]:
                raise ValueError("Incorrect definition of similarities")

            self.similarities = similarities
        else:
            self.generate_similarities()

    def generate_similarities_for_item(self, item_difficulty):
        total_similarity = item_difficulty
        sim_rates = np.round(np.random.uniform(size=self.item_count - 1), 3)
        sim_rates /= np.sum(sim_rates)
        sim_rates *= total_similarity
        return sim_rates

    def generate_similarities(self):
        self.similarities = np.identity(self.item_count)

        for item_idx, item in enumerate(self.items):
            item_diff = item.difficulty
            similarities = self.generate_similarities_for_item(item_diff)
            sim_id = 0
            for sim_item_id, sim_item in enumerate(self.items):
                if sim_item != item:
                    self.similarities[item_idx][sim_item_id] = similarities[sim_id]
                    sim_id += 1

    def get_similarities_for_item(self, item):
        item_id = self.items.index(item)
        return self.similarities[item_id]

    def get_wrong_answer_for_item(self, item):
        item_id = self.items.index(item)
        item_similarities = self.similarities[item_id].copy()
        item_similarities = np.where(
            item_similarities == 1.0, 0.0, item_similarities)
        item_similarities /= np.sum(item_similarities)
        # print(item_similarities)
        try:
            answer_sim = np.random.choice(item_similarities, p=item_similarities)
        except ValueError:
            return item
        answer_id = np.where(item_similarities == answer_sim)[0][0]
        answer = self.items[answer_id]

        return answer


if __name__ == '__main__':
    all_items = list()
    for _ in range(20):
        all_items.append(Item())
        print(all_items[-1])

    material = Material(all_items)
    for item in all_items:
        print(item.get_similarities())
        print(np.sum(item.get_similarities()) - 1 - item.difficulty)

    for item in all_items:
        diff = item.difficulty
        similarities = item.get_similarities()
        assert np.sum(similarities) - 1.0 - diff < 1e-10

        print(similarities)
        wrong_answers = list()
        for _ in range(1000):
            wrong_answers.append(item.get_wrong_answer())

        answer_items = set(wrong_answers)
        answer_rates = list()
        for answer_item in answer_items:
            print(answer_item, wrong_answers.count(answer_item) / 1000)
        print(answer_rates)
