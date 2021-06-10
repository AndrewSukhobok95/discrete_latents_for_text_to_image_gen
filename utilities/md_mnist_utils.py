import json


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





