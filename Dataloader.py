
import numpy as np
import tqdm
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.utils.data_utils import Sequence

class SmilesTokenizer(object):
    def __init__(self, use_word_vectors = False):

############    FIXME #######################
        # Implement word-vector-type Tokenizing option 
        self.use_word_vecs = use_word_vectors


        atoms = [
            'Al', 'As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'K', 'Li', 'N',
            'Na', 'O', 'P', 'S', 'Se', 'Si', 'Te'
        ]
        special = [
            '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 's'
        ]
        padding = ['G', 'A', 'E']

        self.table = sorted(atoms, key=len, reverse=True) + special + padding
        table_len = len(self.table)

        self.table_2_chars = list(filter(lambda x: len(x) == 2, self.table))
        self.table_1_chars = list(filter(lambda x: len(x) == 1, self.table))

        self.one_hot_dict = {}
        for i, symbol in enumerate(self.table):
            vec = np.zeros(table_len, dtype=np.float32)
            vec[i] = 1
            self.one_hot_dict[symbol] = vec

    def tokenize(self, smiles):
        smiles = smiles + ' '
        N = len(smiles)
        token = []
        i = 0
        while (i < N):
            c1 = smiles[i]
            c2 = smiles[i:i + 2]

            if c2 in self.table_2_chars:
                token.append(c2)
                i += 2
                continue

            if c1 in self.table_1_chars:
                token.append(c1)
                i += 1
                continue

            i += 1

        return token

    def one_hot_encode(self, tokenized_smiles):
        result = np.array(
            [self.one_hot_dict[symbol] for symbol in tokenized_smiles],
            dtype=np.float32)
        result = result.reshape(1, result.shape[0], result.shape[1])
        return result



class DataLoader(Sequence):
    def __init__(self,  data_filename, batch_size = 32, sequence_length = 128, validation_split = 0.1, seed = None, data_type = "train", use_word_vectors = False):
        
        self.data_filename = data_filename
        self.batch_size = batch_size
        self.seq_len = sequence_length
        self.seed = seed
        self.data_type = data_type
        self.validation_split = validation_split
        self.smiles = self._load(self.data_filename)

        ################ FIXME ##############
        # update tokenizer to be able to make word-vector-type representations of atoms
        self.use_word_vecs = use_word_vectors

        self.st = SmilesTokenizer()
        self.one_hot_dict = self.st.one_hot_dict
        self.tokenized_smiles = self._tokenize(self.smiles)

        if self.seed != None:
          np.random.seed(self.seed)

        self.idx = np.arange(len(self.tokenized_smiles))
        self.valid_size = int(
                            np.ceil(
                                      len(self.tokenized_smiles) * self.validation_split)
                              )
        
        np.random.shuffle(self.idx)



    def _set_data(self):
        if self.data_type == 'train':
            ret = [
                self.tokenized_smiles[self.idx[i]]
                for i in self.idx[self.valid_size:]
            ]
        elif self.data_type == 'val':
            ret = [
                self.tokenized_smiles[self.idx[i]]
                for i in self.idx[:self.valid_size]
            ]
        else:
            ret = self.tokenized_smiles
        return ret

    def _load(self, data_filename):
        length = self.seq_len
        print('loading SMILES...')
        with open(data_filename) as f:
            smiles = [s.rstrip() for s in f]
        if length != 0:
            smiles = smiles[:length]
        print('done.')
        return smiles

    def _tokenize(self, smiles):

        assert isinstance(smiles, list)

        print('tokenizing SMILES...')
        tokenized_smiles = [self.st.tokenize(smi) for smi in tqdm(smiles)]

        print('done.')
        return tokenized_smiles

    def __len__(self):
        target_tokenized_smiles = self._set_data()
        ret = int(
            np.ceil(
                len(target_tokenized_smiles) /
                float(self.batch_size)
                )
            )
        return ret

    def __getitem__(self, idx):
        target_tokenized_smiles = self._set_data()
        #print(target_tokenized_smiles)
        
        data = target_tokenized_smiles[ idx * self.batch_size : (idx + 1) * self.batch_size ]

        self.X, self.y = [], []
        for tp_smi in data:
            X = [self.one_hot_dict[symbol] for symbol in tp_smi[:-1]]
            self.X.append(X)
            y = [self.one_hot_dict[symbol] for symbol in tp_smi[1:]]
            self.y.append(y)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        return self.X, self.y

    def _pad(self, tokenized_smi):
        return ['G'] + tokenized_smi + ['E'] + [
            'A' for _ in range(self.seq_len - 1 - len(tokenized_smi))
        ]

    def _padding(self, data):
        padded_smiles = [self._pad(t_smi) for t_smi in data]
        return padded_smiles
