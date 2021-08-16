import json
import numpy as np


class LabelsInfo:
    def __init__(self, json_path):
        with open(json_path) as json_file:
            self.data = json.load(json_file)

        self.id2value_dict = {}
        for k in self.data['value_id_tree'].keys():
            for v, v_id in self.data['value_id_tree'][k].items():
                self.id2value_dict[v_id] = v

    def id2value(self, id):
        return self.id2value_dict[id]

    def value2id(self, type, value):
        return self.data['value_id_tree'][type][value]

    def encode_values(self, seqs):
        n = len(seqs)
        encoded_seqs = [[] for _ in range(n)]
        for si in range(n):
            seq = seqs[si]
            for j in range(3):
                encoded_seqs[si].append(self.value2id('number', seq[4 * j]))
                encoded_seqs[si].append(self.value2id('size', seq[1 + 4 * j]))
                encoded_seqs[si].append(self.value2id('color', seq[2 + 4 * j]))
                encoded_seqs[si].append(self.value2id('position', seq[3 + 4 * j]))
        return encoded_seqs

    def decode_ids(self, seqs):
        n = len(seqs)
        decoded_seqs = [[] for _ in range(n)]
        for si in range(n):
            seq = seqs[si]
            for id in seq:
                v = self.id2value(id)
                decoded_seqs[si].append(v)
        return decoded_seqs


class DescriptionGenerator:
    def __init__(self,
                 batch_size,
                 sizes=['20', '30', '40'],
                 colors=['w', 'r', 'g', 'b'],
                 positions=['up', 'middle', 'down']):
        self.sizes = sizes
        self.colors = colors
        self.positions = positions
        self.batch_size = batch_size

    def sample(self):
        x_txt = []
        for _ in range(self.batch_size):
            seq = []
            for i in range(3):
                num = str(np.random.randint(10))
                s = str(np.random.choice(self.sizes))
                c = str(np.random.choice(self.colors))
                p = str(np.random.choice(self.positions))
                seq += [num, s, c, p]
            x_txt.append(seq)
        return x_txt

    def get_properties_by_index(self, index):
        if index > 3:
            index = index % 4
        if index == 1:
            return self.sizes
        elif index == 2:
            return self.colors
        elif index == 3:
            return self.positions
        return list(map(str, range(10)))

    def modify(self, seq, n_changes):
        new_seq = seq.copy()
        assert n_changes <= 12, "Can't change more than 12 tokens."
        change_index = np.random.choice(range(12), size=n_changes)
        for i in range(12):
            if i in change_index:
                properties = self.get_properties_by_index(i)
                properties = list(set(properties) - set([new_seq[i]]))
                new_seq[i] = np.random.choice(properties)
        return new_seq

    def sample_with_modifications(self):
        x_txt = []
        samples = self.sample()
        for s in range(self.batch_size):
            x_txt.append(samples[s])
            for m in range(1, 13):
                x_txt.append(self.modify(samples[s], m))
        return x_txt

