from datasets import load_dataset
from torch.utils.data import IterableDataset
import glob
from dataclasses import dataclass
import itertools
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from multiprocessing import Pool
from functools import partial

@dataclass
class UnTokenizedDatasetConfig:
    data_dir: str
    is_shuffle: bool
    random_seed: int = 3142
    # l_ctx: int
    # tokenizer: PreTrainedTokenizer
    shuffle_buffer_size: int = 1000

class UnTokenizedDataset:
    def __init__(self, conf: UnTokenizedDatasetConfig):
        self.data_dir = conf.data_dir
        self.is_shuffle = conf.is_shuffle
        self.shuffle_buffer_size = conf.shuffle_buffer_size
        self.random_seed = conf.random_seed

        if self.data_dir.endswith('/'):
            raise "The data_dir cannot end with '/'!"
        list_files = glob.glob(f"{self.data_dir}/**/*.jsonl", recursive=True)
        
        self._dataset_raw = (load_dataset("json", data_files=list_files, streaming=True))['train']
        if self.is_shuffle:
            self._dataset_raw = self._dataset_raw.shuffle(seed=conf.random_seed, buffer_size=self.shuffle_buffer_size)
        
        self.iter = iter(self._dataset_raw)
    
    def take(self, n: int):
        list_sentence_dict = list(self._dataset_raw.take(n))
        list_sentence = list(map(lambda x: x["text"], list_sentence_dict))
        return list_sentence
    
    def __next__(self):
        return next(self.iter)["text"]

    def __iter__(self):
        self.iter = iter(self._dataset_raw)
        return self

@dataclass
class ChunkedDatasetConfig:
    data_dir: str
    is_shuffle: bool
    l_ctx: int
    tokenizer: PreTrainedTokenizer
    random_seed: int = 3142
    num_worker: int = 1
    buffer_size: int = 100000
    shuffle_buffer_size: int = 1000


class ChunkedDataset(IterableDataset):
    def __init__(self, conf: ChunkedDatasetConfig, 
                 global_parallel_rank: int = 0, 
                 num_devices: int = 1):
        self.data_dir = conf.data_dir
        self.is_shuffle = conf.is_shuffle
        self.shuffle_buffer_size = conf.shuffle_buffer_size
        self.random_seed = conf.random_seed
        self.l_ctx = conf.l_ctx
        self.tokenizer = conf.tokenizer
        self.buffer_size = conf.buffer_size
        self.num_worker = 1

        conf_untokenized = _config_transfer(conf)
        self._data_set_untokenized = UnTokenizedDataset(conf_untokenized)
        self.iter = iter(self._data_set_untokenized)

        self.buffer_sentence = []
        self.buffer = []

        check_is_rank_devices_legal(global_parallel_rank,
                                    num_devices, 
                                    conf.buffer_size)
        self.global_parallel_rank = global_parallel_rank
        self.num_devices = num_devices
        

    
    def __next__(self):
        
        # If buffer is empty, add more sentences and then tokenize them
        if self.buffer == []:
            self.sentence_buffer = []
            for _ in range(self.buffer_size):
                self.sentence_buffer.append(next(self.iter)) 
            with Pool(processes=self.num_worker) as pool:
                tokenize_and_chunk_partial: function = \
                    partial(tokenize_and_chunk, 
                            tokenizer=self.tokenizer,
                            l_ctx=self.l_ctx)
                buffer = pool.map(tokenize_and_chunk_partial, self.sentence_buffer)
            for chunks in buffer:
                for chunk in chunks:
                    self.buffer.append(chunk)
        
        next_chunk = self.buffer[self.global_parallel_rank]
        self.buffer = self.buffer[self.num_devices:]
        return next_chunk
    

    def __iter__(self):
        self.buffer = []
        self.iter = iter(self._data_set_untokenized)
        return self

def check_is_rank_devices_legal(global_parallel_rank: int,
                                num_devices: int,
                                buffer_size: int):
    if buffer_size%num_devices != 0:
        raise f"buffer_size must be divisible by num_devices"
    if global_parallel_rank >= num_devices:
        raise f"global_parallel_rank must be less than num_devices"

def tokenize_and_chunk(text: str, tokenizer: PreTrainedTokenizer, l_ctx: int):
    tokenized: torch.Tensor = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    token_ids = tokenized['input_ids'].squeeze()

    chunks: list[torch.Tensor] = [token_ids[i:i + l_ctx + 1] for i in range(0, len(token_ids), l_ctx)]

    pad_token_id = tokenizer.pad_token_id or 0
    if len(chunks[-1]) < l_ctx + 1:
        vec = pad_token_id*torch.ones(l_ctx + 1, dtype=chunks[0].dtype) 
        vec[:len(chunks[-1])] = chunks[-1]
        chunks[-1] = vec

    return chunks

def _config_transfer(conf: ChunkedDatasetConfig) -> UnTokenizedDatasetConfig:
    return UnTokenizedDatasetConfig(data_dir=conf.data_dir, 
                                is_shuffle=conf.is_shuffle,
                                random_seed=conf.random_seed,
                                shuffle_buffer_size=conf.shuffle_buffer_size)

class ChunkedDataLoader(DataLoader):
    def __init__(self, dataset: ChunkedDataset, 
                 batch_size: int = 1, 
                 batch_start: int = 0):
        
        self.batch_size = batch_size
        self.batch_start = batch_start
        self.global_parallel_rank = dataset.global_parallel_rank
        self.num_devices = dataset.num_devices
        if batch_size % dataset.num_devices != 0:
              raise f"batch_size must be divisible by num_devices!"
        super().__init__(dataset, batch_size//dataset.num_devices)
        # self._dataloader = DataLoader(dataset, batch_size)

    def __iter__(self):
        # _iter = iter(self._dataloader)
        _iter = super().__iter__()
        return itertools.islice(_iter, self.batch_start, None)


def _test_files(data_dir: str):
    print("\n"*2)
    print('='*30 + 'test_files' + '='*30)
    print("")
    list_files = glob.glob(f"{data_dir}/**/*.jsonl", recursive=True)
    _dataset = load_dataset("json", data_files=list_files, streaming=True)
    for i, data in enumerate(_dataset['train']):
        if i > 1:
            break
        print(data)
        print("")
    print("\n"*2)

def _test_dataset_untokenized(data_dir: str):
    conf_dataset_untokenized = UnTokenizedDatasetConfig(data_dir=data_dir,
                                 is_shuffle=False)
    dataset = UnTokenizedDataset(conf=conf_dataset_untokenized)
    
    print("\n"*2)
    print('='*30 + 'test_dataset_untokenized' + '='*30)
    print("")
    print("1. test of iteration")
    for i, data in enumerate(dataset):
        print(data)
        print("")
        if i > -1:
            break

    for i, data in enumerate(dataset):
        print(data)
        print("")
        if i > -1:
            break

    print("2. test of take function")
    for i in range(2):
        print(dataset.take(1))
        print("")

    print("\n"*2)

def _test_dataset_chuncked(data_dir: str):
    from transformers import AutoTokenizer
    tokenizer_name = "Qwen/Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, pad_token='<|endoftext|>')
    conf_dataset = ChunkedDatasetConfig(data_dir=data_dir,
                                        is_shuffle=False,
                                        l_ctx=100,
                                        tokenizer=tokenizer,
                                        num_worker=2,
                                        buffer_size=100)
    dataset = ChunkedDataset(conf=conf_dataset)
    
    print("\n"*2)
    print('='*30 + 'test_dataset_chuncked' + '='*30)
    print("")
    print("1. test of chuncked dataset")
    for i, data in enumerate(dataset):
        print(tokenizer.decode(data[:-1]))
        print("")
        if i > 0:
            break

    for i, data in enumerate(dataset):
        print(tokenizer.decode(data[:-1]))
        print("")
        if i > 0:
            break

    print("2. test of dataloader")
    dataloader = ChunkedDataLoader(dataset, batch_start=1)
    for i, data in enumerate(dataloader):
        print(tokenizer.decode(data[0][:-1]))
        print("")
        if i > 5:
            break
    print("\n"*2)

    # print("3. test of dataloader -- multiple epoch")
    # dataloader = ChunkedDataLoader(dataset, batch_start=0)
    # for _ in range(5):
    #     for i, data in enumerate(dataloader):
    #         if i == 0:
    #             print(tokenizer.decode(data[0][:-1]))
    #             print("")
    #         # if i > -1:
    #         #     break
    #     print("Finish an epoch!")
    # print("\n"*2)

    print("4. test of distributed dataloader")
    dataloader = ChunkedDataLoader(dataset, 
                                   batch_start=0)
    for i, data in enumerate(dataloader):
        print(i, tokenizer.decode(data[0][:-1]))
        print("")
        if i > 13:
            break
    print("\n"*2)
    dataloader = ChunkedDataLoader(dataset, batch_size=2, batch_start=0)
    for i, data in enumerate(dataloader):
        print(i, tokenizer.decode(data[0][:-1]))
        print("")
        if i > 5:
            break
    print("\n"*2)

    

if __name__ == '__main__':
    import os 
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_downloaded/_test/step2_processed/sky")
    _test_files(data_dir)
    _test_dataset_untokenized(data_dir)
    _test_dataset_chuncked(data_dir)