# todo 1 重写dataloader, dataloader_skip: IterableDataset type 不对，后面可能有bug
# todo 2 重写dataloader 成一个class，能reload
# todo 3 重写一个新的dataloader, load过程要tokenize




from datasets import load_dataset
from torch.utils.data import IterableDataset
import glob
from dataclasses import dataclass
from torch.utils.data import DataLoader
import itertools
import torch

@dataclass
class DataloaderConfig:
    batch_size: int
    num_worker: int
    batch_start: int = 0
    is_shuffle: bool = True
    random_seeds: int = 3142


def create_dateset(data_dir: str) -> IterableDataset:
    if data_dir.endswith('/'):
        raise "The data_dir cannot end with '/'!"
    list_files = glob.glob(f"{data_dir}/**/*.jsonl", recursive=True)
    dataset = load_dataset("json", data_files=list_files, streaming=True)
    return dataset['train']

def create_dataloader(dataset: IterableDataset, conf_dataloader: DataloaderConfig):
    if conf_dataloader.is_shuffle and \
    not isinstance(conf_dataloader.random_seeds, int) and \
        conf_dataloader.random_seeds >= 0:
        raise "You must set conf_dataloader.random_seeds when conf_dataloader.is_shuffle = True!"
    local_generator = torch.Generator()
    local_generator.manual_seed(conf_dataloader.random_seeds)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=conf_dataloader.batch_size,
                            shuffle=conf_dataloader.is_shuffle,
                            num_workers=conf_dataloader.num_worker,
                            generator = local_generator)
    # if conf_dataloader.batch_start != 0:
    #     for batch, _ in enumerate(dataloader):
    #         if not (batch < conf_dataloader.batch_start - 1):
    #             break 
    dataloader_skip: IterableDataset = itertools.islice(dataloader, conf_dataloader.batch_start, None)
    return dataloader_skip

# class StreamDataset(IterableDataset):
#     def __init__(self, data_dir: str):
#         super(StreamDataset).__init__()
#         self.data_dir = data_dir
#         if data_dir.endswith('/'):
#             raise "The data_dir cannot end with '/'!"
#         self.list_files = glob.glob(f"{self.data_dir}/**/*.jsonl", recursive=True)
#         self._dataset = load_dataset("json", data_files=self.list_files, streaming=True)
    
#     def __iter__(self):
#         return self._dataset['train'].__iter__()
    

if __name__ == '__main__':
    import os 
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_downloaded/step2_processed/_test")
    dataset = create_dateset(data_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1)
    # for i, data in enumerate(dataloader):
    #     if i > 5:
    #         break
    #     print(data)

    conf_dataloader = DataloaderConfig(batch_size=1, num_worker=1, batch_start=1, is_shuffle=False)
    dataloader = create_dataloader(dataset, conf_dataloader)
    for i, data in enumerate(dataloader):
        if i > 5:
            break
        print(data)


    # list_files = glob.glob(f"{data_dir}/**/*.jsonl", recursive=True)
    # _dataset = load_dataset("json", data_files=list_files, streaming=True)
    # for i, data in enumerate(_dataset['train']):
    #     if i > 5:
    #         break