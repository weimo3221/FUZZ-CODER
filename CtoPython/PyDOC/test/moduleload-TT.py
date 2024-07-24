import torch
import os
from common import pad


# 用于控制op的范围
def control_op(out, mode):
    # out: [B, T, V] if mode == 1
    # out: [B, V] if mode == 0
    if mode == 1:
        for i in range(out.size(0)):
            for j in range(out.size(1)):
                if torch.argmax(out[i][j], dim=0) == pad:
                    continue
                if j % 2 == 0:
                    for k in range(out.size(2)):
                        if k == 0 or k > 13:
                            out[i][j][k] = torch.tensor(-torch.inf)
        return out
    else:
        for i in range(out.size(0)):
            if torch.argmax(out[i], dim=0) == pad:
                continue
            for j in range(out.size(1)):
                if j == 0 or j > 13:
                    out[i][j] = torch.tensor(-torch.inf)
        return out


def writefragment(mes):
    with open("./fragment.txt", "a") as f:
        f.write(repr(mes))
        f.write("\n")


def gettopn(input_seq, p, c):
    # input_seq是输入的队列，p则是我想取的top n个的概率的总和
    # c 为 0 时表示的是op，c 为 1 时表示的是pos
    middle_seq = (input_seq - torch.min(input_seq)) / (torch.max(input_seq) - torch.min(input_seq))
    middle_seq = torch.div(middle_seq, torch.sum(middle_seq))
    num = 1
    while True:
        values, indices = torch.topk(middle_seq, k=num+1)
        if torch.sum(values) > p:
            break
        else:
            num += 1
    values, indices = torch.topk(middle_seq, k=num)
    results = []
    for i in indices:
        if c == 0:
            results.append(i.item()+1)
        else:
            results.append(i.item())
    results_seq = torch.div(values, torch.sum(values))
    out = torch.multinomial(results_seq, 1, replacement=False)
    pre_value = results[out.item()]
    return results, pre_value


def run(data):
    # 这里如果cuda:0的存储量不够的话可以调整为其他的cuda，服务器中有0,1,2,3
    device = torch.device("cuda:0")
    top_p = 0.5
    data = data.strip().replace('\n', '').replace('\r', '')
    data = [data[i * 2: i * 2 + 2] for i in range(len(data) // 2)]
    data = [int(i, 16) for i in data]
    src = torch.LongTensor([data]).to(device)
    model = None
    try:
        # 这里得是绝对路径
        model = torch.load('./TTnet.pth', map_location=device)
    except Exception as e:
        writefragment(e)
        print(e)
    model.eval()
    # src = torch.tensor([[0, 1, 3, 1, 4, 2, 3, 2, 5, 5, 7, 89, 499, 499, 499]])
    tgt = torch.LongTensor([[0]]).to(device)
    num = 0
    middle_seq = []
    while True:
        # print(len(middle_seq))
        # 进行transformer计算
        out = model(src, tgt)  # [B, 1+j, H]
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])  # [B, V]
        # 找出最大值的index
        middle = torch.argmax(predict, dim=1)  # [B]
        if num % 2 == 0 and middle != pad:
            results, y = gettopn(predict[0][1:14], top_p, 0)
        else:
            results, y = gettopn(predict[0], top_p, 1)
        # 和之前的预测结果拼接到一起
        if middle == pad or len(middle_seq) == 12000:
            break
        middle_seq.append(results)
        y = torch.tensor([y]).to(device)
        tgt = torch.concat([tgt, y.unsqueeze(1)], dim=1)  # [B, 1+j]
        num += 1
    if len(middle_seq) % 2 != 0:
        middle_seq = middle_seq[:-1]
    dic = []
    for i in range(len(middle_seq) // 2):
        for j in (middle_seq[i*2]):
            for k in (middle_seq[i*2+1]):
                if k != pad and [j,k] not in dic:
                    dic.append([j,k])
    dic.sort(key=lambda x: (x[0], x[1]))
    # print(dic)
    result = ""
    for i in dic:
        result += (hex(i[0])[2:].rjust(3, '0') + hex(i[1])[2:].rjust(3, '0'))
    # print(result)
    result += "00e001"
    return result


def readdoc():
    with open("./in.txt", 'r') as f:
        content = f.read()
    return content


def writedoc(data):
    with open("./out.txt", "w") as f:
        f.write(data)


def writeerr(mes):
    with open("./err.txt", "a") as f:
        f.write(repr(mes))
        f.write("\n")


if __name__ == '__main__':
    data = readdoc()
    try:
        result = run(data)
    except Exception as e:
        writeerr(e)
    writedoc(result)
