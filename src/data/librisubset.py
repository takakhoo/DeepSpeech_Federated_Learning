# put above lines into a torch dataset that take file_path as init argument
import torch
import soundfile
import os
import h5py
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from torchvision.transforms import Compose

class HDF5SampledDataset(Dataset):
    '''
    Dataset to load individual HDF5 files containing (numpy array, text, file_path, duration, sampling_rate).
    '''
    def __init__(self, hdf5_dir, min_length=0, max_length=100000, transform=None, target_transform=None):
        # Get list of HDF5 files
        self.hdf5_files = sorted([f for f in os.listdir(hdf5_dir) if f.endswith('.h5')])
        self.hdf5_dir = hdf5_dir
        self.min_length = min_length
        self.max_length = max_length
        self._transform = transform
        self._target_transform = target_transform

        # Filter the dataset based on the duration in the HDF5 files
        self.filtered_files = []
        for hdf5_file in self.hdf5_files:
            file_path = os.path.join(hdf5_dir, hdf5_file)
            with h5py.File(file_path, 'r') as hdf5:
                duration = hdf5['duration'][()]
                if self.min_length <= duration < self.max_length:
                    self.filtered_files.append(hdf5_file)

    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, index):
        # Get the corresponding HDF5 file path
        hdf5_file = self.filtered_files[index]
        hdf5_path = os.path.join(self.hdf5_dir, hdf5_file)

        # Read the data from the HDF5 file
        with h5py.File(hdf5_path, 'r') as hdf5:
            audio = hdf5['array'][:]
            text = hdf5['text'][()].decode('utf-8')
            # file_path = hdf5['file_path'][()].decode('utf-8')
            # duration = hdf5['duration'][()]
            # rate = hdf5['sampling_rate'][()]


        # Apply optional transformations to audio
        if self._transform is not None:
            audio = self._transform(audio)

        # Apply optional transformations to target text
        if self._target_transform is not None:
            text = self._target_transform(text)

        return audio, text

class LibriSampledDataset(Dataset):
    '''
    600 samples of Libri test set
    '''
    def __init__(self, file_path, min_length=0, max_length=100000, transform=None, target_transform=None):
        with open(file_path) as f:
            lines = f.read().strip().split('\n')
        lines = lines
        lines = [l.split(',') for l in lines]
        lines = sorted(lines, key=lambda l: int(l[1]))
        lines = [l[0] for l in lines if min_length <= int(l[1]) < max_length]

        self.lines = lines
        self.utts = []
        for fn  in lines:
            self.utts.append(open(os.path.join('/scratch/f006pq6/datasets/.deep-speaker-wd/samples_deepspeaker/samples/librispeech/single', '%s.csv' % fn)).read().strip().split('\n')[-1])
        self.max_length = max(len(l.split(',')[-1]) for l in self.utts)

        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, index):
        # utts[idx] give path, duration, transcript seperated by comma

        path, dur, target= self.utts[index].split(',') #target is transcript
        audio, rate = soundfile.read(path, dtype='int16')
        print(path)
        assert rate == 16000, '%r sample rate != 16000' % path

        if self._transform is not None:
            audio = self._transform(audio)

        if self._target_transform is not None:
            target = self._target_transform(target)

        return audio, target

def get_dataset_libri_sampled_folder_subset(net,FLAGS):

    target_transform = Compose([str.lower,
                net.ALPHABET.get_indices,
                torch.IntTensor])
    file_path = '/scratch/f006pq6/projects/asr-grad-reconstruction/samples/samples_below_4s_bucket_500_all_minh.txt'
    # dataset = LibriSampledDataset(file_path, min_length=FLAGS.batch_min_dur, max_length=FLAGS.batch_max_dur, transform=net.transform, target_transform=target_transform)
    dataset = HDF5SampledDataset(FLAGS.dataset_path, min_length=FLAGS.batch_min_dur, max_length=FLAGS.batch_max_dur, transform=net.transform, target_transform=target_transform)

    #get subset of the dataset start from FLAGS.start_idx to FLAGS.end_idx
    #using torch Subset
    if FLAGS.batch_end == -1 or FLAGS.batch_end > len(dataset): 
        FLAGS.batch_end = len(dataset)

    # import ipdb;ipdb.set_trace()
    dataset = torch.utils.data.Subset(dataset, range(FLAGS.batch_start, FLAGS.batch_end))
    


    loader = torch.utils.data.DataLoader(dataset,
                                         collate_fn=collate_input_sequences,
                                         pin_memory=torch.cuda.is_available(),
                                         num_workers=0,
                                         batch_size=FLAGS.batch_size,
                                         shuffle=False)
    print('number of utterances:', len(dataset))
    print('example shape of input:', dataset[0][0][0].shape)
    print('number of frames in example:', dataset[0][0][1])
    print('example target:', dataset[0][1])
    return dataset, loader

def get_dataset_libri_sampled_loader(net,FLAGS):
    # dataset_1 = LibriSpeech(root='/scratch/f006pq6/datasets/librispeech/', subsets=['test-clean'], download=True,
    #                       transform=net.transform)
    
    target_transform = Compose([str.lower,
                net.ALPHABET.get_indices,
                torch.IntTensor])
    file_path = '/scratch/f006pq6/projects/asr-grad-reconstruction/samples/samples_below_4s_bucket_500_all_minh.txt'
    dataset = LibriSampledDataset(file_path, min_length=FLAGS.batch_min_dur, max_length=FLAGS.batch_max_dur, transform=net.transform, target_transform=target_transform)

    loader = torch.utils.data.DataLoader(dataset,
                                         collate_fn=collate_input_sequences,
                                         pin_memory=torch.cuda.is_available(),
                                         num_workers=0,
                                         batch_size=FLAGS.batch_size,
                                         shuffle=False)
    print('number of utterances:', len(dataset))
    print('example shape of input:', dataset[0][0][0].shape)
    print('number of frames in example:', dataset[0][0][1])
    print('example target:', dataset[0][1])
    return dataset, loader

def get_datapoint_i(loader_iterator, idx):
    for i in range(idx):
        next(loader_iterator)
    next_item = next(loader_iterator)
    print('next item shape of input:', next_item[0][0].shape)
    print('next item number of frames:', next_item[0][1])
    print('next item target:', next_item[1])
    return next_item


def collate_input_sequences(samples):
    """Returns a batch of data given a list of samples.

    Args:
        samples: List of (x, y) where:

            `x`: A tuple:
                - `torch.Tensor`: an input sequence to the network with size
                      `(len(torch.Tensor), n_features)`.
                - `int`: the length of the corresponding output sequence
                      produced by the network given the `torch.Tensor` as
                      input.
            `y`: A `torch.Tensor` containing the target output sequence.

    Returns:
        A tuple of `((batch_x, batch_out_lens), batch_y)` where:

            batch_x: The concatenation of all `torch.Tensor`'s in `x` along a
                new dim in descending order by `torch.Tensor` length.

                This results in a `torch.Tensor` of size (L, N, D) where L is
                the maximum `torch.Tensor` length, N is the number of samples,
                and D is n_features.

                `torch.Tensor`'s shorter than L are extended by zero padding.

            batch_out_lens: A `torch.IntTensor` containing the `int` values
                from `x` in an order that corresponds to the samples in
                `batch_x`.

            batch_y: A list of `torch.Tensor` containing the `y` `torch.Tensor`
                sequences in an order that corresponds to the samples in
                `batch_x`.

    Example:
        >>> x = [# input seq, len 5, 2 features. output seq, len 2
        ...      (torch.full((5, 2), 1.0), 2),
        ...      # input seq, len 4, 2 features. output seq, len 3
        ...      (torch.full((4, 2), 2.0), 3)]
        >>> y = [torch.full((4,), 1.0), # target seq, len 4
        ...      torch.full((3,), 2.0)] # target seq, len 3
        >>> smps = list(zip(x, y))
        >>> (batch_x, batch_out_lens), batch_y = collate_input_sequences(smps)
        >>> print('%r' % batch_x)
        tensor([[[ 1.,  1.],
                 [ 2.,  2.]],

                [[ 1.,  1.],
                 [ 2.,  2.]],

                [[ 1.,  1.],
                 [ 2.,  2.]],

                [[ 1.,  1.],
                 [ 2.,  2.]],

                [[ 1.,  1.],
                 [ 0.,  0.]]])
        >>> print('%r' % batch_out_lens)
        tensor([ 2,  3], dtype=torch.int32)
        >>> print('%r' % batch_y)
        [tensor([ 1.,  1.,  1.,  1.]), tensor([ 2.,  2.,  2.])]
    """

    samples = [(*x, y) for x, y in samples]
    sorted_samples = sorted(samples, key=lambda s: len(s[0]), reverse=True)

    seqs, seq_lens, labels = zip(*sorted_samples)

    x = (pad_sequence(seqs), torch.IntTensor(seq_lens))
    y = list(labels)

    return x, y


