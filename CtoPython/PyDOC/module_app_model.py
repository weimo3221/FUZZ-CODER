from flask import Flask, request
import vllm
import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", "-p", type=str, default="nm", help="nm, objdump, readelf...")
    parser.add_argument("--computer", "-c", type=str, default="3", help="输入显卡的序号")
    parser.add_argument("--num", "-n", type=int, default=100, help="生成的次数，比如100次")
    parser.add_argument("--port", "-po", type=int, default=5000, help="服务器端口的序号")
    parser.add_argument("--model", "-m", type=str, default="/data2/hugo/fuzz/code_llama/model-checkpoint/model-checkpoint200/dpsk7b/cpfs01/shared/public/yj411294/fuzzy_test/stanford_alpaca/models/dpsk-7B/lr3e-5-ws100-wd0.0-bsz512/checkpoint-200", help="大模型的地址")
    args = parser.parse_args()
    return args

args = parse_args()
# 需要修改这里的program-n (Need to change the program-n here)
program_n = args.program
head1 = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n\n### Input:\n\nTask description:\nNow, you are a AFL (American Fuzzy Lop), which is a highly efficient and widely used fuzz testing tool designed for finding security vulnerabilities and bugs in software. \nYou are now fuzzing a program named "
head2 = ", which requires variable (a byte sequence) to run. \nI will give you a byte sequence as input sequence, and you need to mutate the input sequence to give me a output sequence through a mutation operation below. \nFinally you need to give me a output which includes input sequence, mutation operation and output sequence.\n\nMutation operations:\n1. Perform bitfilp on a bit randomly.\n2. Perform bitfilp on two neighboring bits randomly.\n3. Perform bitfilp on four neighboring bits randomly.\n4. Randomly select a byte and XOR it with 0xff.\n5. Randomly select two neighboring bytes and XOR them with 0xff.\n6. Randomly select four neighboring bytes and XOR them with 0xff.\n7. Randomly select a byte and perform addition or subtraction on it (operands are 0x01~0x23).\n8. Randomly select two neighboring bytes and convert these two bytes into a decimal number. Select whether to swap the positions of these two bytes. Perform addition or subtraction on it (operands are 1~35). Finally convert this number to 2 bytes and put it back to its original position.\n9. Randomly select four neighboring bytes. Select whether to swap the positions of these four bytes. Convert these four bytes into a decimal number. Perform addition or subtraction on it (operands are 1~35). Finally convert this number to 4 bytes and put it back to its original position.\n10. Randomly select a byte and replace it with a random byte in {0x80, 0xff,0x00,0x01,0x10,0x20,0x40,0x64,0x7F}.\n11. Randomly select two neighboring bytes and replace them with two random bytes in {(0xff 0x80),(0xff 0xff),(0x00 0x00),(0x00 0x01),(0x00 0x10),(0x00 0x20),(0x00 0x40),(0x00 0x64),(0x00 0x7f),(0x80 0x00),(0xff 0x7f),(0x00 0x80),(0x00 0xff),(0x01 0x00),(0x02 0x00),(0x03 0xe8),(0x04 0x00),(0x10 0x00),(0x7f 0xff)}.\n12. Randomly select four neighboring bytes and replace them with four random bytes in {(0xff 0xff 0xff 0x80),(0xff 0xff 0xff 0xff),(0x00 0x00 0x00 0x00),(0x00 0x00 0x00 0x01),(0x00 0x00 0x00 0x10),(0x00 0x00 0x00 0x20),(0x00 0x00 0x00 0x40),(0x00 0x00 0x00 0x64),(0x00 0x00 0x00 0x7f),(0xff 0xff 0x80 0x00),(0xff 0xff 0xff 0x7f),(0x00 0x00 0x00 0x80),(0x00 0x00 0x00 0xff),(0x00 0x00 0x01 0x00),(0x00 0x00 0x02 0x00),(0x00 0x00 0x03 0xe8),(0x00 0x00 0x04 0x00),(0x00 0x00 0x10 0x00),(0x00 0x00 0x7f 0xff),(0x80 0x00 0x00 0x00),(0xfa 0x00 0x00 0xfa),(0xff 0xff 0x7f 0xff),(0x00 0x00 0x80 0x00),(0x00 0x00 0xff 0xff),(0x00 0x01 0x00 0x00),(0x05 0xff 0xff 0x05),(0x7f 0xff 0xff 0xff)}.\n\nInput Sequence Definition:\nIt consists of bytes represented in hexadecimal, separated by spaces. It is the byte sequence to be mutated. It is a variable that can cause the program to crash or trigger a new path.\n\nOutput Sequence Definition:\nIt consists of bytes represented in hexadecimal, separated by spaces. It is the mutated byte sequence. It is a variable that can cause the program to crash or trigger a new path.\n\ninput sequence:\n"
head = head1 + program_n + head2
end = "\nPlease list all possible mutation strategies (mutation position and mutation operation) with the JSON format as:\noutput:\n{\n    \"mutation strategies\": [\n        (op_1, pos_1), \n        (op_2, pos_2), \n        ... , \n        (op_N, pos_N)\n    ]\n}\n\n\n### Response:\n"


# 加载服务器板块 (Loading server plate)
app = Flask(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = args.computer
# 加载大模型板块
base_model = args.model
sampling_params = vllm.SamplingParams(
    temperature=1, top_p=0.95, max_tokens=4096)
model = vllm.LLM(
    model=base_model, tensor_parallel_size=1, worker_use_ray=True
)
# 输出预处理板块 (Output pre-processing plate)


def text_deal(text):
    pos_s = text.find('"[')
    pos_e = text.find(']"')
    text = text[pos_s + 2: pos_e]
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace(" ", "")
    # print(text)
    num_list = text.split(",")
    return num_list


def post_process(dic):
    # dic: [[op, pos], [op, pos], ...]
    new_dic = []
    for i in range(len(dic)):
        if dic[i] not in new_dic and dic[i][0] <= 12:
            new_dic.append(dic[i])
    new_dic.sort(key=lambda x: (x[0], x[1]))
    # new_dic: [[op, pos], [op, pos], ...] 排序后并且删除了一些不满足条件的[op, pos]
    # (After sorting and removing some [op, pos] that don't satisfy the condition)
    result = ""
    for i in new_dic:
        result += (hex(i[0])[2:].rjust(3, '0') + hex(i[1])[2:].rjust(3, '0'))
    print(result)
    # result: 16进制, 每3位表示一个数字 (Hexadecimal, every 3 digits represent a number.)
    return result



@app.route('/')
def get_output():
    # 这个函数是主要用于输入得到的input，input是16进制的字符串，如"0011223344..."，但保证长度为2的倍数
    # (This function is mainly used to get input, input is a hexadecimal string, such as "0011223344 ...")
    # (but to ensure that the length of a multiple of 2)
    data = request.args.get('input', '')
    data = data.strip().replace('\n', '').replace('\r', '')
    if data == '':
        return []
    data = [data[i * 2: i * 2 + 2] for i in range(len(data) // 2)]
    byte_input = [hex(int(i, 16))[2:].rjust(2, '0') for i in data]
    input_num = args.num
    source = head + " ".join(byte_input) + end
    prompts = [source] * input_num
    outputs = model.generate(prompts, sampling_params)
    results = []
    dic = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        pos = generated_text.find('",')
        generated_text = generated_text[:pos + 1] + generated_text[pos + 2:]
        num_list = text_deal(generated_text)
        num_list = [x for x in num_list if x.isdigit()]
        for i in range(len(num_list) // 2):
            dic.append([int(num_list[i*2]), int(num_list[i*2+1])])
        results.append({
            "prompt": prompt,
            "response": generated_text
        })
    # 如果要将JSON字符串存储到文件中 (If you want to store a JSON string to a file)
    # with open('my_list.json', 'w', encoding="utf-8") as json_file:
    #     json.dump(results, json_file)
    # dic: [[op, pos], [op, pos], ...] (list)
    # 必须在发送result之前用post_process处理dic再发送给client
    # (The dic must be processed with post_process before the result is sent to the client.)
    result = post_process(dic)
    return result


if __name__ == '__main__':
    # 如果5000端口被占用了，注意换下面的port，并保持module_client.py里面的port是一致的
    # If port 5000 is occupied, be careful to change the port below and keep the port inside module_client.py the same
    app.run(port=args.port, debug=True, use_reloader=False)
