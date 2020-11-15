import numpy as np
import tqdm
from Dataloader import SmilesTokenizer


class Discriminator_DataLoader(object):
    def __init__(self,  data_filename, batch_size = 32, sequence_length = 128,
                    fake_batch_size = 32, seed = None, use_word_vectors = False):
        
        self.data_filename = data_filename
        self.batch_size = batch_size
        self.seq_len = sequence_length
        self.seed = seed
        self.smiles = self._load(self.data_filename)
        self.fake_batch_size = fake_batch_size

        ################ FIXME ##############
        # update tokenizer to be able to make word-vector-type representations of atoms
        self.use_word_vecs = use_word_vectors

        self.st = SmilesTokenizer()
        self.one_hot_dict = self.st.one_hot_dict
        self.tokenized_smiles = self._tokenize(self.smiles)
        self.batch_idx = 0
        self.completed_epochs = 0

        self.max_batch_idx = np.int(np.ceil(
            len(self.tokenized_smiles)/self.batch_size
        ))
        if self.seed != None:
          np.random.seed(self.seed)

        self.idx = np.arange(len(self.tokenized_smiles))

        
        np.random.shuffle(self.idx)



    def _set_data(self):

        ret = [
                self.tokenized_smiles[self.idx[i]]
                for i in self.idx
            ]
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

    def get_batch(self, gen):
        target_tokenized_smiles = self._set_data()
        #print(target_tokenized_smiles)
        
        data = target_tokenized_smiles[ self.batch_idx * self.batch_size : (self.batch_idx + 1) * self.batch_size ]

        self.X, self.y = [], []
        for tp_smi in data:
            real_x = [self.one_hot_dict[symbol] for symbol in tp_smi[:-1]]
            self.X.append(real_x)
            real_y = 1.0
            self.y.append(real_y)
        
        fakes = gen.generate(self.fake_batch_size)
        for fake in fakes:
            self.X.append(fake)
            self.y.append(0.0)
            

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
        self.batch_idx += 1
        epoch_finished = False
        if self.batch_idx > self.max_batch_idx:
            self.batch_idx = 0
            self.completed_epochs += 1
            epoch_finished = True
            print("ALLLLLLL Done")
            print("Restarting at beginning of dataset")

        return epoch_finished, self.X, self.y

    def _pad(self, tokenized_smi):
        return ['G'] + tokenized_smi + ['E'] + [
            'A' for _ in range(self.seq_len - 1 - len(tokenized_smi))
        ]

    def _padding(self, data):
        padded_smiles = [self._pad(t_smi) for t_smi in data]
        return padded_smiles
