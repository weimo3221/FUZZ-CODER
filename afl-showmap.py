import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default="fuzz_out",
                        help="The output folder after fuzzing a program")
    parser.add_argument("--program", "-p", type=str, default="none",
                        help="The program which is fuzzing, such as: objdump, magick, readelf, nm, xmllint, jpegtran, mp3gain, tiffsplit")
    args_r = parser.parse_args()
    return args_r


if __name__ == '__main__':
    args = parse_args()
    pre_path = os.path.abspath(args.folder)
    program = args.program
    f_p = ""
    o_p = ""
    command = ""
    if program == "objdump":
        command = " ../binutils-2.27/binutils/objdump -x -a -d "
    elif program == "readelf":
        command = " ../binutils-2.27/binutils/readelf -a "
    elif program == "nm":
        command = " ../binutils-2.27/binutils/nm-new -a "
    elif program == "xmllint":
        command = " ../libxml2-2.9.2/xmllint --valid --recover "
    elif program == "jpegtran":
        command = " ../jpeg-9e/jpegtran "
    elif program == "mp3gain":
        command = " ../mp3gain-1.5.2/mp3gain "
    elif program == "magick":
        command = " ../ImageMagick-7.1.0-49/utilities/magick identify "
    elif program == "tiffsplit":
        command = " ../libtiff-Release-v3-9-7/tools/tiffsplit "
    else:
        print("please input a program's name")
        sys.exit()
    folders = os.listdir(pre_path)
    if not os.path.exists(os.path.join(pre_path, "map_data")):
        os.system("mkdir " + os.path.join(pre_path, "map_data"))
    lines_list = [[], []]
    map_dic = []
    if "queue" not in folders:
        for folder in folders:
            if folder == "map_data":
                continue
            if not os.path.exists(os.path.join(pre_path, "map_data", folder)):
                os.system("mkdir " + os.path.join(pre_path, "map_data", folder))
            files = os.listdir(os.path.join(pre_path, folder, "queue"))
            for file in files:
                print(f"!!!!!!!!{pre_path}!!{folder}!{file}!!!!!!!!!")
                if "id" not in file:
                    continue
                o_p = file + ".txt"
                f_p = os.path.join(os.path.join(pre_path, folder, "queue", file))
                full_command = "afl-showmap -o " + os.path.join(pre_path, "map_data", folder, o_p) + command + f_p
                os.system(full_command)
                with open(os.path.join(pre_path, "map_data", folder, o_p), "r") as f:
                    lines = f.readlines()
                    line_number = str(len(lines))
                    lines_list[0].append(os.path.join(pre_path, "map_data", folder, o_p))
                    lines_list[1].append(line_number)
                    for line in lines:
                        pos = line.find(":")
                        num = line[:pos]
                        if num not in map_dic:
                            map_dic.append(num)
    else:
        files = os.listdir(os.path.join(pre_path, "queue"))
        if not os.path.exists(os.path.join(pre_path, "map_data", "fuzzer-one")):
            os.system("mkdir " + os.path.join(pre_path, "map_data", "fuzzer-one"))
        for file in files:
            if "id" not in file:
                continue
            o_p = file + ".txt"
            f_p = os.path.join(pre_path, "queue", file)
            full_command = "afl-showmap -o " + os.path.join(pre_path, "map_data", "fuzzer-one", o_p) + command + f_p
            os.system(full_command)
            with open(os.path.join(pre_path, "map_data", "fuzzer-one", o_p), "r") as f:
                lines = f.readlines()
                line_number = str(len(lines))
                lines_list[0].append(os.path.join(pre_path, "map_data", "fuzzer-one", o_p))
                lines_list[1].append(line_number)
                for line in lines:
                    pos = line.find(":")
                    num = line[:pos]
                    if num not in map_dic:
                        map_dic.append(num)
    with open(os.path.join(pre_path, "map_data", "map_all.txt"), "w") as f:
        for i in range(len(lines_list[0])):
            f.write(lines_list[0][i] + ': ' + lines_list[1][i] + "\n")
    with open(os.path.join(pre_path, "map_data", "map_density.txt"), "w") as f:
        for i in map_dic:
            f.write(i + "\n")
        f.write(f"map_density_num:{len(map_dic)}\n")
