import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var

class SentenceJMVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, latent_size,

                sos_idx, eos_idx, pad_idx, max_sequence_length,label_sequence_len, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.label_sequence_len = label_sequence_len
        
        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout = nn.Dropout(p=word_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size+label_sequence_len, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.encoder_rnn_w = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.relu = nn.LeakyReLU()
        self.sigmoid =nn.Sigmoid()

        self.encoder_linear_y1 = nn.Linear(label_sequence_len, hidden_size)
        self.encoder_linear_y2 = nn.Linear(hidden_size, hidden_size)
        #
        add_y_to_z =  False 
        if add_y_to_z ==True:
            self.decoder_rnn = rnn(embedding_size+label_sequence_len, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        else:
            self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2mean_w = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv_w = nn.Linear(hidden_size * self.hidden_factor, latent_size)        
        self.hidden2mean_y = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv_y = nn.Linear(hidden_size * self.hidden_factor, latent_size)

        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

        self.latent2hidden_y =nn.Linear(latent_size, hidden_size)
        self.decoder_linear_y = nn.Linear(hidden_size, label_sequence_len)

    def encode_joint(self,input_embedding, label_sequence,sorted_lengths):
        label_sequence2 = torch.Tensor(input_embedding.shape[0],input_embedding.shape[1],label_sequence.shape[1])
        label_sequence = label_sequence.unsqueeze(1)
        label_sequence2 = torch.cat([label_sequence]*input_embedding.shape[1],dim=1)
        input_embedding = torch.cat((input_embedding,label_sequence2),dim=2)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        return mean,logv

    def encode_w(self,input_embedding,sorted_lengths):
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn_w(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean_w = self.hidden2mean_w(hidden)
        logv_w = self.hidden2logv_w(hidden)
        return mean_w,logv_w

    def encode_y(self,label_seq):
        label_seq = label_seq#.float()
        hidden = self.relu(self.encoder_linear_y1(label_seq))
        hidden = self.encoder_linear_y2(hidden)

        # REPARAMETERIZATION
        mean_y = self.hidden2mean_y(hidden)
        logv_y = self.hidden2logv_y(hidden)
        return mean_y, logv_y

    def decode_to_y(self,z,sorted_lengths,sorted_idx):
        hidden_y = self.relu(self.latent2hidden_y(z))
        #OOPS!!
        _,reversed_idx = torch.sort(sorted_idx)
        hidden_y = hidden_y[reversed_idx]
        logp2 = self.sigmoid(self.decoder_linear_y(hidden_y))
        return logp2

    def decode_joint(self,z,input_embedding,sorted_lengths,sorted_idx):
        add_y_to_z = False
        hidden = self.latent2hidden(z)
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)
        input_embedding = self.word_dropout(input_embedding)
        """
        if add_y_to_z== True:
            label_sequence2 = torch.Tensor(input_embedding.shape[0],input_embedding.shape[1],label_sequence.shape[1])
            for i in range(input_embedding.shape[0]):
                for j in range(input_embedding.shape[1]):
                    for k in range(label_sequence.shape[1]):
                        label_sequence2[i,j,k] = label_sequence[i,k]

            label_sequence = label_sequence2
            input_embedding = torch.cat((input_embedding,label_sequence),dim=2)
        """
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        #hidden_y = self.relu(self.latent2hidden_y(z))
        #OOPS!!
        #hidden_y = hidden_y[reversed_idx]
        #logp2 = self.sigmoid(self.decoder_linear_y(hidden_y))
        logp2 = self.decode_to_y(z,sorted_lengths,sorted_idx)
        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)
        return logp, logp2


    def sample_z(self,batch_size,mean,logv):
        std = torch.exp(0.5 * logv)
        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        return z

    def forward(self, input_sequence,label_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        input_embedding = self.embedding(input_sequence)

        #OOPS!!!!
        label_sequence = label_sequence[sorted_idx]


        # ENCODERS
        mean, logv = self.encode_joint(input_embedding, label_sequence, sorted_lengths)
        mean_w, logv_w = self.encode_w(input_embedding,sorted_lengths)
        mean_y, logv_y = self.encode_y(label_sequence)
        # SAMPLER
        z = self.sample_z(batch_size, mean, logv)
        # DECODER
        logp,logp2 = self.decode_joint(z,input_embedding,sorted_lengths,sorted_idx)

        return logp, logp2, mean, logv, z, mean_w, logv_w, mean_y, logv_y


    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)
        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            #EXTRA LINES TO MAKE IT WORK:
            #Unknown why sometimes 
            if (input_sequence.shape == torch.Size([])):
                input_sequence = input_sequence.unsqueeze(0)



            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
