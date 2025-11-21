# Add the 'modules/deepspeech/src/' directory to the system path

import sys, os
import torch



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../modules/deepspeech/src')))
import deepspeech

from deepspeech.networks.utils import OverLastDim
from deepspeech.data import preprocess
from deepspeech.data.alphabet import Alphabet
from torchvision.transforms import Compose
from deepspeech.data.loader import collate_input_sequences

_BLANK_SYMBOL = '_'
def _gen_alphabet():
    symbols = list(" abcdefghijklmnopqrstuvwxyz'")
    symbols.insert(0,_BLANK_SYMBOL)
    return Alphabet(symbols)

class AddContextFramesTorch:
    def __init__(self, n_context, n_feature):
        self.n_context = n_context
        self.n_feature= n_feature 
        self.unfold_op = torch.nn.Unfold(kernel_size=(2 * n_context + 1, n_feature), dilation=1, padding=(n_context, 0))
    
    def __call__(self, signal):
        # input of this transformation will have shape (t,B,f)
        # convert it (t,B,f) -> (1,t,B,f) -> B, 1, t,f
        signal = signal.unsqueeze(0).permute(2,0,1,3)
        n,c,t,f = signal.shape
        # unfold signal
        signal_unfolded = self.unfold_op(signal)
        # reshape signal
        signal_unfolded = signal_unfolded.transpose(1,2)
        return signal_unfolded.transpose(0,1).contiguous() # t, B, f 

class DeepSpeech1WithContextFrames(torch.nn.Module):
    '''
    Default network wont have context frames input
    modify it so it context frames
    '''

    def __init__(self, optimiser_cls=None, optimiser_kwargs=None,
                 decoder_cls=None, decoder_kwargs=None,
                 n_hidden=2048, n_context=9, n_mfcc=26, drop_prob=0.25,
                 winlen=0.025, winstep=0.02, sample_rate=16000,use_relu=True):
        super().__init__()
        
        self._n_hidden = n_hidden
        self._n_context = n_context
        self._n_mfcc = n_mfcc
        self._drop_prob = drop_prob
        self._winlen = winlen
        self._winstep = winstep
        self._sample_rate = sample_rate

        self._use_relu = use_relu
        self.network = self._get_network()
        self.ALPHABET = _gen_alphabet()
        self.get_context = AddContextFramesTorch(self._n_context, self._n_mfcc)

        
    def forward(self, x):
        '''
        Args:
        x: A MFCC tensor of shape (seq_len, batch, n_features).
        Perform extract n_context and put it to network.
        '''
        x = self.get_context(x)
        return self.network(x)



    def _get_network(self):
        return Network(in_features=self._n_mfcc*(2*self._n_context + 1),
                       n_hidden=self._n_hidden,
                       out_features=29,
                       drop_prob=self._drop_prob, use_relu=self._use_relu)
    @property
    def transform(self):
        return Compose([preprocess.MFCC(self._n_mfcc),
                        # preprocess.AddContextFrames(self._n_context),
                        # preprocess.Normalize(),
                        torch.FloatTensor,
                        lambda t: (t, len(t))])
                    
    @property
    def target_transform(self):
        return Compose([str.lower,
                        self.ALPHABET.get_indices,
                        torch.IntTensor])

class Network(torch.nn.Module):
    """A network with 3 FC layers, a Bi-LSTM, and 2 FC layers.

    Args:
        in_features: Number of input features per step per batch.
        n_hidden: Internal hidden unit size.
        out_features: Number of output features per step per batch.
        drop_prob: Dropout drop probability.
        relu_clip: ReLU clamp value: `min(max(0, x), relu_clip)`.
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
    """

    def __init__(self, in_features, n_hidden, out_features, drop_prob,
                 relu_clip=20.0, forget_gate_bias=1.0, use_relu=True):
        super().__init__()

        self._relu_clip = relu_clip
        self._drop_prob = drop_prob

        self.fc1 = self._fully_connected(in_features, n_hidden, relu=use_relu, dropout=False)
        self.fc2 = self._fully_connected(n_hidden, n_hidden, relu=use_relu, dropout=False)
        self.fc3 = self._fully_connected(n_hidden, 2*n_hidden, relu=use_relu, dropout=False)
        self.bi_lstm = self._bi_lstm(2*n_hidden, n_hidden, forget_gate_bias)
        self.fc4 = self._fully_connected(2*n_hidden, n_hidden, relu=use_relu, dropout=False)
        self.out = self._fully_connected(n_hidden,
                                         out_features,
                                         relu=False,
                                         dropout=False)

    def _fully_connected(self, in_f, out_f, relu=True, dropout=True):
        layers = [torch.nn.Linear(in_f, out_f)]
        if relu:
            layers.append(torch.nn.Hardtanh(0, self._relu_clip, inplace=True))
        if dropout:
            layers.append(torch.nn.Dropout(p=self._drop_prob))
        return OverLastDim(torch.nn.Sequential(*layers))

    def _bi_lstm(self, input_size, hidden_size, forget_gate_bias):
        lstm = torch.nn.LSTM(input_size=input_size,
                       hidden_size=hidden_size,
                       bidirectional=True)
        if forget_gate_bias is not None:
            for name in ['bias_ih_l0', 'bias_ih_l0_reverse']:
                bias = getattr(lstm, name)
                bias.data[hidden_size:2*hidden_size].fill_(forget_gate_bias)
            for name in ['bias_hh_l0', 'bias_hh_l0_reverse']:
                bias = getattr(lstm, name)
                bias.data[hidden_size:2*hidden_size].fill_(0)
        return lstm

    def forward(self, x):
        """Computes a single forward pass through the network.

        Args:
            x: A tensor of shape (seq_len, batch, in_features).

        Returns:
            Logits of shape (seq_len, batch, out_features).
        """
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        h, _ = self.bi_lstm(h)
        h = self.fc4(h)
        out = self.out(h)
        return out