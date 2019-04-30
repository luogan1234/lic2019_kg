import Schema
import Vocab
import Data
import paras
import time
import os
import pickle
import model

class DataCollection:
    def __init__(self, vocab, schemas, train_data, test_data):
        self.vocab = vocab
        self.schemas = schemas
        self.train_data = train_data
        self.test_data = test_data

def load_data():
    global data_collection
    start = time.time()
    if os.path.exists(paras.DATA_COL):
        with open(paras.DATA_COL, 'rb') as f:
            data_collection = pickle.load(f)
    else:
        vocab = Vocab.Vocab()
        schemas = Schema.load_schema()
        train_data = Data.load_data(paras.TRAIN_DATA, schemas)
        test_data = Data.load_data(paras.TEST_DATA, schemas)
        train_data.get_indexes(vocab)
        test_data.get_indexes(vocab)
        data_collection = DataCollection(vocab, schemas, train_data, test_data)
        with open(paras.DATA_COL, 'wb') as f:
            pickle.dump(data_collection, f)
    end = time.time()
    data_collection.vocab.print_info()
    data_collection.schemas.print_info()
    print('load data time cost:', end - start)

def test():
    res = data_collection.train_data.output_result()
    with open(paras.RESULT, 'w', encoding='utf-8') as f:
        f.write(res)
    batch_data = data_collection.train_data.next_batch(100)

def main():
    model.train(data_collection)

if __name__ == '__main__':
    load_data()
    #test()
    main()