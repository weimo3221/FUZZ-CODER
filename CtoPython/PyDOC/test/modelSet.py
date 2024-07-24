from torch import nn
import torch
import math
from common import pad
import torch.nn.functional as F
from einops import rearrange
# data_size: 数据集的大小
# T是seq_len的长度， T1是src的seq_len长度， T2是tgt的seq_len的长度
# B: batch_size
# H: hidden_dim
# E: embedding_dim
# V: vocab_size
# 模型的代码定义


def attention_model(input_size, attention_size):
    model = nn.Sequential(
        nn.Linear(input_size, attention_size, bias=False), 
        nn.Tanh(),                
        nn.Linear(attention_size, 1, bias=False)
    )
    return model


def attention_forward(model, enc_states, dec_state):
    """
    <bos> x1 x2 x3 <eos> -> <eos> hidden state
    """
    """
    model:函数attention_model返回的模型
    enc_states: 编码端的输出，shape是(批量⼤⼩, 时间步数, 隐藏单元个数)
    dec_state: 解码端一个时间步的输出，shape是(批量⼤⼩, 隐藏单元个数)
    """
    # 将解码器隐藏状态⼴播到和编码器隐藏状态形状相同后进⾏连结
    if dec_state.dim() == 2:
        dec_state = dec_state.unsqueeze(dim=1)  # [B, H] -> [B, 1, H]
    dec_states = dec_state.expand_as(enc_states)
    """
    dec_states: [B, H] -> [B, T, H]
    enc_steate: [B, T ,H]
    torch.cat(enc_states, dec_states): [B, T, H + H] -> [B, T, 1]
    """
    # 形状为(批量⼤⼩, 时间步数, 1)
    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
    e = model(enc_and_dec_states)  # 这里的model是上面attention_model函数的返回值
    alpha = F.softmax(e, dim=1)  # 在时间步维度做softmax运算
    return (alpha * enc_states).sum(dim=1)  # 返回背景变量 [B, H]


class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, att_dropout=0.0, aropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, c, h, w = x.shape

        x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(context)

        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim]

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        out = self.proj_out(out)   # [batch_size, c, h, w]

        print(out.shape)

        return out, att_weights


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Module3(nn.Module):
    def __init__(self, hidden_dim, layer_dim=1, d_model=128, vocab_size=500, n_head=8, pad_id=pad,
                 device=torch.device("cpu")):
        super(Module3, self).__init__()

        # 定义词向量，词典数为vocab_size。我们不预测两位小数。
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_id)
        # 定义Transformer和LSTM。

        self.encoder = LSTMEncoder(d_model, hidden_dim, layer_dim).to(device)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2).to(device)
        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(src.device)  # [T2, T2]
        tgt_key_padding_mask = Module3.get_key_padding_mask(tgt).type(torch.bool).to(src.device)  # [B, T2]

        # 对src和tgt进行编码
        src = self.embedding(src)  # [B, T1, E]
        tgt = self.embedding(tgt)  # [B, T2, E]
        # 给tgt的token增加位置信息
        tgt = self.positional_encoding(tgt)  # [B, T2, E]

        # 将准备好的数据送给transformer
        enc_state = self.encoder.begin_state()  # None
        memory, _ = self.encoder(src, enc_state)  # [B, T1, E]
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)  # [B, T2, E]
        # out = self.transformer(src, tgt,
        #                        tgt_mask=tgt_mask,
        #                        src_key_padding_mask=src_key_padding_mask,
        #                        tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())  # [B, T2]
        key_padding_mask[tokens == pad] = 1
        return key_padding_mask  # [B, T2]


class Module4(nn.Module):

    def __init__(self, d_model=128, vocab_size=500, num_layers=2, n_head=8, pad_id=pad):
        super(Module4, self).__init__()

        # 定义词向量，词典数为10。我们不预测两位小数。
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_id)
        # 定义Transformer
        # self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=num_layers,
        #                                   num_decoder_layers=num_layers,
        #                                   dim_feedforward=512,
        #                                   batch_first=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(src.device)  # [T2, T2]
        src_key_padding_mask = Module4.get_key_padding_mask(src).type(torch.bool).to(src.device)  # [B, T1]
        tgt_key_padding_mask = Module4.get_key_padding_mask(tgt).type(torch.bool).to(src.device)  # [B, T2]

        # 对src和tgt进行编码
        src = self.embedding(src)  # [B, T1, E]
        tgt = self.embedding(tgt)  # [B, T2, E]
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)  # [B, T1, E]
        tgt = self.positional_encoding(tgt)  # [B, T2, E]

        # 将准备好的数据送给transformer
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)  # [B, T1, E]
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)  # [B, T2, E]
        # out = self.transformer(src, tgt,
        #                        tgt_mask=tgt_mask,
        #                        src_key_padding_mask=src_key_padding_mask,
        #                        tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == pad] = 1
        return key_padding_mask


class LstmEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layers, pad_id=pad):
        """
        vocab_size:词典长度
        embedding_dim:词向量的维度
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(LstmEncoder, self).__init__()
        self.hidden_dim = hidden_dim  # RNN神经元个数
        self.layers = layers  # RNN的层数
        self.num_directions = 2  # 用于双向LSTM
        # 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        # 双向LSTM ＋ 全连接层
        if layers == 1:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers,
                                batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers,
                                batch_first=True, bidirectional=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.linear_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_content = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, state):
        embeds = self.embedding(x)  # [B, T1, H]
        # r_out: [B, T1, H]
        # h_n: [num_layers*num_directions, B, H]   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c: [num_layers*num_directions, B, H]
        r_out, (h_n, h_c) = self.lstm(embeds, state)  # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的out输出
        if self.num_directions == 2:
            h_n = torch.cat(
                [h_n[:self.layers * 2:2, :, :], h_n[1:self.layers * 2 + 1:2, :, :]], dim=2)  # [num_layers, B, 2H]
            h_c = torch.cat(
                [h_c[:self.layers * 2:2, :, :], h_c[1:self.layers * 2 + 1:2, :, :]], dim=2)  # [num_layers, B, 2H]
        h_n = self.linear_hidden(h_n)  # [num_layers, B, 2H] -> [num_layers, B, H]
        h_c = self.linear_content(h_c)  # [num_layers, B, 2H] -> [num_layers, B, H]
        batch_size, seq_len, hid_dim = r_out.size()
        r_out = r_out.contiguous().view(batch_size, seq_len, self.num_directions, self.hidden_dim)
        # [B, T1, 2H] -> [B, T1, 2, H]
        r_out = torch.mean(r_out, dim=2)  # [B, T1, 2, H] -> [B, T1, H]
        out = r_out.view(batch_size, seq_len, -1)
        return out, (h_n, h_c)  # [B, T1, H], ([num_layers, B, H], [num_layers, B, H])

    def begin_state(self):
        return None  # 隐藏态初始化为None时PyTorch会⾃动初始化为0


class LstmDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layers, output_dim, attention_size, pad_id=pad):
        """
        vocab_size:词典长度
        embedding_dim:词向量的维度
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(LstmDecoder, self).__init__()
        self.hidden_dim = hidden_dim  # RNN神经元个数
        # 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.attention = attention_model(2 * hidden_dim, attention_size)
        self.layers = layers
        self.hidden_dim = hidden_dim
        # LSTM ＋ 全连接层
        if layers == 1:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers,
                                batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers,
                                batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(self.layers * hidden_dim, hidden_dim)

    def forward(self, x, state, enc_states):
        # r_out: [B, T1, H]
        # h_n: [num_layers*num_directions, B, H]   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c: [num_layers*num_directions, B, H]
        # state: [2, num_layers, B, H]
        # 使⽤注意⼒机制计算背景向量
        # 这里取的state是取的是隐藏状态
        if self.layers == 1:
            c = attention_forward(self.attention, enc_states, state[0].squeeze(dim=0))  # [B, H]
        else:
            # 这里由于Decoder是一个单向的LSTM，所以还得对state进行一定的处理
            # 我们必须求出每一个
            c = None
            for k in range(state[0].size(0)):
                middle = attention_forward(self.attention, enc_states, state[0][k])
                if k == 0:
                    c = middle
                else:
                    c = torch.cat((c, middle), dim=1)
            c = self.fc3(c)  # [B, H]
        # c: [B, H]
        # 将嵌⼊后的输⼊和背景向量在特征维连结
        x = self.embedding(x)  # [B, E]
        # x: [B, E]
        # 这里需要注意c的hidden_size和x的embed_size不一定相等，但下面这个必须得保证相等才能相加起来
        input_and_c = torch.cat((x, c), dim=1)  # [B, 2*E]
        input_and_c = self.fc2(input_and_c)  # [B, E]
        r_out, state = self.lstm(input_and_c.unsqueeze(1), state)  # None 表示 hidden state 会用全0的 state 
        # r_out: [B, 1, H]
        # state: [2, num_layers, B, H]
        # 移除时间步维，输出形状为(批量⼤⼩, 输出词典⼤⼩)
        r_out = r_out.contiguous().view(-1, self.hidden_dim)  # r_out: [B, H]
        output = self.fc1(r_out)  # [B, V]
        return output, state  # [B, V], [2, num_layers, B, H]

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers):
        """
        vocab_size:词典长度
        embedding_dim:词向量的维度
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        另外这个LSTM是用于LSTM和Transformer的，有略微不同于LSTM和LSTM
        """
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim  # RNN神经元个数
        self.layers = layers  # RNN的层数
        self.num_directions = 2  # 用于双向LSTM
        # 双向LSTM ＋ 全连接层
        if layers == 1:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers,
                                batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers,
                                batch_first=True, bidirectional=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.linear_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_content = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, state):
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x, state)  # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的out输出
        if self.num_directions == 2:
            h_n = torch.cat(
                [h_n[:self.layers * 2:2, :, :], h_n[1:self.layers * 2 + 1:2, :, :]], dim=2)  # [num_layers, B, 2H]
            h_c = torch.cat(
                [h_c[:self.layers * 2:2, :, :], h_c[1:self.layers * 2 + 1:2, :, :]], dim=2)  # [num_layers, B, 2H]
        h_n = self.linear_hidden(h_n)  # [num_layers, B, 2H] -> [num_layers, B, H]
        h_c = self.linear_content(h_c)  # [num_layers, B, 2H] -> [num_layers, B, H]
        batch_size, seq_len, hid_dim = r_out.size()
        r_out = r_out.contiguous().view(batch_size, seq_len, self.num_directions, self.hidden_dim)
        # [B, T1, 2H] -> [B, T1, 2, H]
        r_out = torch.mean(r_out, dim=2)  # [B, T1, 2, H] -> [B, T1, H]
        out = r_out.view(batch_size, seq_len, -1)
        return out, (h_n, h_c)  # [B, T1, H], ([num_layers, B, H], [num_layers, B, H])

    def begin_state(self):
        return None  # 隐藏态初始化为None时PyTorch会⾃动初始化为0


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head=8, layers=2, pad_id=pad):
        super(TransformerEncoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=0)
        self.layers = layers
        self.nhead = n_head
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            # dim_feedforward=4 * embedding_dim,
            nhead=self.nhead, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers)

    def forward(self, src):
        src_key_padding_mask = TransformerEncoder.get_key_padding_mask(src).type(torch.bool)  # [B, T1]
        src = self.emb(src)  # [B, T1, E]
        # 添加位置信息
        src = self.positional_encoding(src)  # [B, T1, E]
        src = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)  # [B, T1, E]
        return src

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size()).to(tokens.device)  # [B, T1]
        key_padding_mask[tokens == pad] = 1
        return key_padding_mask  # [B, T1]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, n_head=8, layers=6, pad_id=pad):
        super(TransformerDecoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=n_head)
        self.positional_decoding = PositionalEncoding(embedding_dim, dropout=0)
        self.transformer_decoder = nn.TransformerDecoder(self.encoder_layer, num_layers=layers)
        self.fc1 = nn.Linear(embedding_dim, output_dim)

    def forward(self, memory, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])  # [T2, T2]
        tgt_key_padding_mask = TransformerDecoder.get_key_padding_mask(tgt)  # [B, T2]
        tgt = self.emb(tgt)  # [B, T2, E]
        # 添加位置信息
        tgt = self.positional_decoding(tgt)  # [B, T2, E]
        out = self.transformer_decoder(memory, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # [B, T2, E]
        out = self.fc1(out)  # [B, T2, E] -> [B, T2, V]
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size()).to(tokens.device)  # [B, T2]
        key_padding_mask[tokens == pad] = -torch.inf
        return key_padding_mask  # [B, T2]
