num = 0
with open("../docset/number.txt", "r") as f:
    num = int(f.read())
    num += 1
with open("../docset/number.txt", "w") as f:
    f.write(str(num))

