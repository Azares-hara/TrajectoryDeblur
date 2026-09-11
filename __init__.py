import traceback
from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from .sampler import DistributedEvalSampler

class Data:
    def __init__(self, args):
        self.args = args
        self.modes = ['train', 'val', 'test', 'demo']

        self.action = {
            'train': args.do_train,
            'val': args.do_validate,
            'test': args.do_test,
            'demo': args.demo
        }

        self.dataset_name = {
            'train': args.data_train,
            'val': args.data_val,
            'test': args.data_test,
            'demo': 'Demo'
        }

        self.loaders = {}
        for mode in self.modes:
            if self.action[mode]:
                try:
                    self.loaders[mode] = self._get_data_loader(mode)
                    print(f"===> Loaded {mode} dataset: {self.dataset_name[mode]}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {mode} dataset: {e}")
                    self.loaders[mode] = None
            else:
                self.loaders[mode] = None

    def _get_data_loader(self, mode='train'):
        dataset_name = self.dataset_name[mode]
        try:
            dataset_module = import_module(f'dataset.{dataset_name.lower()}')
            dataset_class = getattr(dataset_module, dataset_name)
            dataset = dataset_class(self.args, mode)
        except (ImportError, AttributeError) as e:
            print(traceback.format_exc())
            raise RuntimeError(f"Could not import dataset '{dataset_name}': {e}")

        print(f"[DEBUG] Mode: {mode}")
        print(f"[DEBUG] Dataset class: {dataset.__class__.__name__}")
        print(f"[DEBUG] Dataset length: {len(dataset)}")
        print(f"[DEBUG] Data root: {self.args.data_root}")

        if self.args.n_GPUs > 0:
            base_batch_size = int(self.args.batch_size / self.args.n_GPUs)
            base_num_workers = int((self.args.num_workers + self.args.n_GPUs - 1) / self.args.n_GPUs)
        else:
            base_batch_size = self.args.batch_size
            base_num_workers = self.args.num_workers

        if mode == 'train':
            batch_size = base_batch_size
            num_workers = base_num_workers
            sampler = DistributedSampler(dataset, shuffle=True,num_replicas=self.args.world_size,rank=self.args.rank) if self.args.distributed else RandomSampler(dataset)
            drop_last = True
        else:
            batch_size = 1 if self.args.distributed else (
                self.args.val_batch_size if mode == 'val' and hasattr(self.args, 'val_batch_size') else
                self.args.n_GPUs if self.args.n_GPUs > 0 else 1
            )
            num_workers = base_num_workers
            sampler = DistributedEvalSampler(dataset, shuffle=False,
                                             num_replicas=self.args.world_size,
                                             rank=self.args.rank) if self.args.distributed else SequentialSampler(dataset)
            drop_last = False

        print(f"[DEBUG] Final batch_size for {mode}: {batch_size}")

        pin_memory = (str(self.args.device_type).startswith("cuda"))

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,     
            pin_memory=pin_memory,               
            prefetch_factor=2 if num_workers > 0 else None,            
            persistent_workers=(num_workers > 0),      
            drop_last=drop_last,
        )


    def get_loader(self):
        return self.loaders
