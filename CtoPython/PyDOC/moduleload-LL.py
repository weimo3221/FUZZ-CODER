import torch
import os
pad = 499

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
    with open("../docset/fragment.txt", "a") as f:
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
    device = torch.device("cuda:0")
    top_p = 0.5
    data = data.strip().replace('\n', '').replace('\r', '')
    data = [data[i * 2: i * 2 + 2] for i in range(len(data) // 2)]
    data = [int(i, 16) for i in data]
    src = torch.LongTensor([data]).to(device)
    encoder = None
    decoder = None
    try:
        # 这里得是绝对路径
        encoder = torch.load('./obj_LLEnet.pth', map_location=device)
        decoder = torch.load('./obj_LLDnet.pth', map_location=device)
    except Exception as e:
        writefragment(e)
        print(e)
    # print(encoder)
    # print(decoder)
    # src = torch.tensor([[0, 1, 3, 1, 4, 2, 3, 2, 5, 5, 7, 89, 499, 499, 499]])
    state = (torch.zeros(2, 1, 128).to(device), torch.zeros(2, 1, 128).to(device))
    # print(state)
    dec_input = torch.tensor([0]).to(device)
    enc_state = state
    enc_outputs, enc_state = encoder(src, enc_state)
    dec_state = decoder.begin_state(enc_state)
    num = 0
    middle_seq = []
    while True:
        # print(len(middle_seq))
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        middle = torch.argmax(dec_output, 1)
        if num % 2 == 0 and middle != pad:
            dec_output = control_op(dec_output, mode=0)
            # 这里我只取了前13个
            results, dec_input = gettopn(dec_output[0][1:14], top_p, 0)
        else:
            results, dec_input = gettopn(dec_output[0], top_p, 1)
        if middle == pad or len(middle_seq) == 12000:
            break
        middle_seq.append(results)
        dec_input = torch.tensor([dec_input]).to(device)
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
    with open("../docset/in.txt", 'r') as f:
    	content = f.read()
    return content
    
def writedoc(data):
    with open("../docset/out.txt", "w") as f:
    	f.write(data)
    	
def writeerr(mes):
    with open("../docset/err.txt", "w") as f:
    	f.write(repr(mes))
    	
if __name__ == '__main__':
    data = readdoc()
    try:
        result = run(data)
    except Exception as e:
        writeerr(e)
<<<<<<< HEAD
        result=""
=======
        result = ""
>>>>>>> 3a4cbf24e12b0cd408f0248d23fdb8ee7cdd2480
    writedoc(result)
        
