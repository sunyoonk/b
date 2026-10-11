# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from datasets import Dataset


def user_fold_of(user, seed=2024) -> str:
    """Deterministic 50/50 user split for the VSID cross-fitted probes."""
    import hashlib
    return 'A' if int(hashlib.md5(f'{seed}:{user}'.encode()).hexdigest(), 16) % 2 == 0 else 'B'


def user_fold_index(user, n_folds=5, seed=2024) -> int:
    """Deterministic K-fold user split (md5 user hash); n_folds = 2 gives index 0 <-> 'A', 1 <-> 'B' of user_fold_of."""
    import hashlib
    return int(hashlib.md5(f'{seed}:{user}'.encode()).hexdigest(), 16) % n_folds


def is_selection_user(user, frac, seed=2024) -> bool:
    """AdaRQ: users reserved for early stopping / δ selection (their events never enter the PVI / depth estimate).
    Independent of the fold hash (different salt)."""
    import hashlib
    return frac > 0 and int(hashlib.md5(f'selection:{seed}:{user}'.encode()).hexdigest(), 16) % 10000 < frac * 10000


class AbstractDataset:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config['accelerator']
        self.logger = getLogger()

        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = None
        self.split_data = None

    def __str__(self) -> str:
        return f'[Dataset] {self.__class__.__name__}\n' \
                f'\tNumber of users: {self.n_users}\n' \
                f'\tNumber of items: {self.n_items}\n' \
                f'\tNumber of interactions: {self.n_interactions}\n' \
                f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: The number of users in the dataset.
        """
        return len(self.user2id)

    @property
    def n_items(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.item2id)

    @property
    def n_interactions(self):
        """
        Returns the total number of interactions in the dataset.

        Returns:
            int: The total number of interactions.
        """
        n_inters = 0
        for user in self.all_item_seqs:
            n_inters += len(self.all_item_seqs[user])
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Returns the average length of item sequences in the dataset.

        Returns:
            float: The average length of item sequences.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.id_mapping['user2id']

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.id_mapping['item2id']

    def _download_and_process_raw(self):
        """
        This method should be implemented in the subclass.
        It is responsible for downloading and processing the raw data.
        """
        raise NotImplementedError('This method should be implemented in the subclass')

    def _leave_one_out(self):
        """
        Splits the dataset into train, validation, and test sets using the leave-one-out strategy.

        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
                  Each dataset is represented as a dictionary with 'user' and 'item_seq' keys.
                  The 'user' key contains a list of users, and the 'item_seq' key contains a list of item sequences.
        """
        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}
        fold = self.config.get('user_fold')  # VSID probes: keep only users of fold 'A' or 'B' (item ids unchanged)
        # AdaRQ cross-fitted probes: drop every user of fold `holdout_fold` (of n_user_folds); item ids unchanged
        holdout, n_folds = self.config.get('holdout_fold'), self.config.get('n_user_folds', 5)
        # AdaRQ event separation: validation (early stopping) restricted to the selection users
        val_set, sel_frac = self.config.get('val_user_set', 'all'), self.config.get('selection_user_frac', 0.0)
        assert val_set in ('all', 'selection')
        seed = self.config.get('fold_seed', 2024)
        for user in self.all_item_seqs:
            if fold in ('A', 'B') and user_fold_of(user, seed) != fold:
                continue
            if holdout is not None and user_fold_index(user, n_folds, seed) == int(holdout):
                continue
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(self.all_item_seqs[user])
            if len(self.all_item_seqs[user]) > 1 and (val_set == 'all' or is_selection_user(user, sel_frac, seed)):
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(self.all_item_seqs[user][:-1])
            if len(self.all_item_seqs[user]) > 2:
                datasets['train']['user'].append(user)
                datasets['train']['item_seq'].append(self.all_item_seqs[user][:-2])
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def split(self):
        """
        Split the dataset into train, validation, and test sets based on the specified split strategy.

        Returns:
            datasets (dict): A dictionary containing the train and test datasets.
        """
        if self.split_data is not None:
            return self.split_data

        split_strategy = self.config['split']
        if split_strategy in ['leave_one_out', 'last_out']:
            datasets = self._leave_one_out()
        else:
            raise NotImplementedError(f'Split strategy [{split_strategy}] not implemented.')

        self.split_data = datasets
        return self.split_data

    def log(self, message, level='info'):
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)
