import torch
import torch.utils.data
from torch import nn
import pandas as pd
import timm.scheduler
pad = 499

# data_size: 数据集的大小
# T是seq_len的长度， T1是src的seq_len长度， T2是tgt的seq_len的长度
# B: batch_size
# H: hidden_dim
# E: embedding_dim
# V: vocab_size


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, samples, targets, device):
        self.samples = torch.LongTensor(samples).to(device)  # [data_size, T1]
        self.labels = torch.LongTensor(targets).to(device)  # [data_size, T2 + 1] 实际训练时只是取T2

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)


def get_dataloader(batch_size, seed_data):
    """
    获取loader的信息
    batch_size: batch_size的大小
    输出: 按批量处理好的data_loader
    """

    data_loader = torch.utils.data.DataLoader(seed_data,
                                              batch_size,
                                              shuffle=True)
    return data_loader


def train_Transformer(model, trainloader, testloader, lr, num_epochs, device):
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,
                                                     t_initial=num_epochs,
                                                     lr_min=1e-5,
                                                     warmup_t=50,
                                                     warmup_lr_init=1e-4
                                                     )
    criteria = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # 调整优化器的学习率
        scheduler.step(epoch)
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        # 训练阶段
        model.train()
        num = 0
        for step, batch in enumerate(trainloader):
            src = batch[0]  # [B, T1]
            tgt = batch[1]  # [B, T2 + 1]
            tgt_y = tgt[:, 1:]  # [B, T2]
            tgt = tgt[:, :-1]  # [B, T2]
            # n_tokens 统计不是pad的个数
            n_tokens = (tgt_y != pad).sum()
            # 清空梯度
            optimizer.zero_grad()
            # 进行transformer的计算
            out = model(src, tgt)  # [B, T, H]
            # 将结果送给最后的线性层进行预测
            out = model.predictor(out)  # [B, T, V]
            """
            计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                    我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
                    (batch_size*词数, 词典大小)。
                    而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                    除以n_tokens。
            """
            loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            y = torch.argmax(out, dim=2)  # [B, T]
            train_num = 0
            for i in range(y.size(0)):
                for j in range(y.size(1)):
                    if y[i][j] == tgt_y[i][j] and tgt_y[i][j] != pad:
                        train_num += 1
            train_corrects += (train_num / n_tokens.item())
            train_loss += loss.item()
            num += 1
        train_loss_all.append(train_loss / num)
        train_acc_all.append(train_corrects / num)
        # train_loss_all: [epoch]
        # train_acc_all: [epoch]
        print('{} Train Loss: {:.8f}  Train Acc: {:.8f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))

        # 计算一个epoch的训练后在验证集上的损失和精度
        model.eval()  # 设置模型为训练模式评估模式
        num = 0
        for step, batch in enumerate(testloader):
            src = batch[0]  # [B, T1]
            tgt = batch[1]  # [B, T2 + 1]
            tgt_y = tgt[:, 1:]  # [B, T2]
            # tgt从<bos>开始，看看能不能重新输出src中的值
            tgt = torch.LongTensor([[0]] * src.size(0)).to(device)  # [B, 1]
            # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
            for j in range(tgt_y.size(1)):
                # 进行transformer计算
                out = model(src, tgt)  # [B, 1+j, H]
                # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
                predict = model.predictor(out[:, -1])  # [B, V]
                # 找出最大值的index
                y = torch.argmax(predict, dim=1)  # [B]
                # 和之前的预测结果拼接到一起
                tgt = torch.concat([tgt, y.unsqueeze(1)], dim=1)  # [B, 1+j]
            n_tokens = (tgt_y != pad).sum()
            out = model.predictor(out)  # [B, T2, V]
            loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
            val_loss += loss.item()
            y = tgt[:, 1:]  # [B, T2]
            val_num = 0
            for i in range(y.size(0)):
                for j in range(y.size(1)):
                    if y[i][j] == tgt_y[i][j] and tgt_y[i][j] != pad:
                        val_num += 1
            val_corrects += (val_num / n_tokens.item())
            num += 1
        # 计算一个epoch在训练集上的损失和精度
        val_loss_all.append(val_loss / num)
        val_acc_all.append(val_corrects / num)
        # val_loss_all: [epoch]
        # val_acc_all: [epoch]
        print('{} Val Loss: {:.8f}  Val Acc: {:.8f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "val_acc_all": val_acc_all})
    return model, train_process


def batch_loss(encoder, decoder, X, Y, loss, flag, mode, device):
    # mode为0时为双向LSTM+单向LSTM, mode为1时为Transformer+单向LSTM
    batch_size = X.shape[0]
    if mode == 1:
        enc_outputs = encoder(X)  # [B, T1, H]
        # 初始化解码器的隐藏状态，这里的隐藏状态初始化设置为全0
        #dec_state = (enc_outputs[:, -1, :].unsqueeze(0), enc_outputs[:, -1, :].unsqueeze(0))
        batch_size, time_step, hidden_size = enc_outputs.size()
        #dec_state = (decoder.embedding(X.new(1, batch_size).fill_(0)), decoder.embedding(X.new(1, batch_size).fill_(0)))
        h0 = torch.zeros(decoder.layers, batch_size, decoder.hidden_dim).to(device)
        c0 = torch.zeros(decoder.layers, batch_size, decoder.hidden_dim).to(device)
        dec_state = (h0, c0)
        # dec_state: [2, num_layers, B, H]
        # h0: [num_layers, B, H]
        # c0: [num_layers, B, H]
    else:
        enc_state = encoder.begin_state()  # None
        enc_outputs, enc_state = encoder(X, enc_state)
        # 初始化解码器的隐藏状态
        dec_state = decoder.begin_state(enc_state)
        # enc_outputs: [B, T1, H]
        # enc_state: [2, num_layers, B, H]
        # dec_state: [2, num_layers, B, H]
    # 解码器在最初时间步的输⼊是0
    dec_input = torch.tensor([0] * batch_size).to(device)
    # 我们将使⽤掩码变量mask来忽略掉标签为填充项PAD的损失
    mask, num_not_pad_tokens = torch.ones(batch_size,).to(device), 0
    # mask: [B]
    l = torch.tensor([0.0]).to(device)
    Y = Y[:, 1:]  # [B, T2]
    seq_len = Y.size(1)  # seq_len的求解
    train_corrects = torch.tensor([0.0])
    for i in range(seq_len):  # Y: [B, T2]
        # dec_input: [B]
        # enc_outputs: [B, T1, H]
        # dec_output: [B, V]
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        mid = Y[:, i]
        # loss(dec_output, Y[:,i,:].squeeze(dim=1).type(torch.float32)) shape: (batch)
        l = l + (mask * loss(dec_output, mid)).sum()
        rea_lab = mask * mid  # rea_lab: [B]
        pre_lab = mask * torch.argmax(dec_output, 1)  # pre_lab: [B]
        # train_corrects += torch.sum(pre_lab == rea_lab).item()
        dec_input = mid if flag == 0 else torch.argmax(dec_output, 1)
        # flag=0的时候使⽤强制教学，flag=1的时候为了验证将得到的值作为下一次的输入
        # 这里是看rea_lab[j]的值是不是0来判断是否为pad
        for j in range(pre_lab.size(0)):
            if rea_lab[j] != 0 and rea_lab[j] == pre_lab[j]:
                train_corrects += 1
            if rea_lab[j] != 0:
                num_not_pad_tokens += 1
        #  将PAD对应位置的掩码设成0,
        mid1 = []
        for j in range(Y.size(0)):
            mid1.append((Y[j][i] != pad).float())
        mask = mask * torch.tensor(mid1).to(device)
    return l / num_not_pad_tokens, train_corrects / num_not_pad_tokens


def train_LSTM(encoder, decoder, trainloader, testloader, lr, num_epochs, mode, device):
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    enc_optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
    enc_scheduler = timm.scheduler.CosineLRScheduler(optimizer=enc_optimizer,
                                                 t_initial=num_epochs,
                                                 lr_min=1e-5,
                                                 warmup_t=75,
                                                 warmup_lr_init=1e-4
                                                 )
    dec_scheduler = timm.scheduler.CosineLRScheduler(optimizer=dec_optimizer,
                                                 t_initial=num_epochs,
                                                 lr_min=1e-5,
                                                 warmup_t=75,
                                                 warmup_lr_init=1e-4
                                                 )
    loss = nn.CrossEntropyLoss(reduction='none')
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        if epoch == 100:
            print(123)
        # 调整两个优化器的学习率
        enc_scheduler.step(epoch)
        dec_scheduler.step(epoch)
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        # 训练阶段
        encoder.train()
        decoder.train()
        num = 0
        # if epoch == 50:
        #     print(1)
        for step, batch in enumerate(trainloader):
            X = batch[0]  # [B, T1, H]
            Y = batch[1]  # [B, T2+1, H]
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l, t = batch_loss(encoder, decoder, X, Y, loss,  flag=0, mode=mode, device=device)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            train_loss += l.item()
            train_corrects += t.item()
            num += 1
        train_loss_all.append(train_loss / num)
        train_acc_all.append(train_corrects / num)
        # train_loss_all: [epoch]
        # train_acc_all: [epoch]
        print('{} Train Loss: {:.8f}  Train Acc: {:.8f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))

        # 计算一个epoch的训练后在验证集上的损失和精度
        encoder.eval()  # 设置模型为训练模式评估模式
        decoder.eval()  # 设置模型为训练模式评估模式
        num = 0
        for step, batch in enumerate(testloader):
            X = batch[0]  # [B, T1, H]
            Y = batch[1]  # [B, T2+1, H]
            l, t = batch_loss(encoder, decoder, X, Y, loss, flag=1, mode=mode, device=device)
            val_loss += l.item()
            val_corrects += t.item()
            num += 1
        # 计算一个epoch在训练集上的损失和精度
        val_loss_all.append(val_loss / num)
        val_acc_all.append(val_corrects / num)
        # val_loss_all: [epoch]
        # val_acc_all: [epoch]
        print('{} Val Loss: {:.8f}  Val Acc: {:.8f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "val_acc_all": val_acc_all})
    return encoder, decoder, train_process