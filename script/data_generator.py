# with open("../data/train.txt", "a") as rf:
#     for i in range(640):
#         rf.write(f"{i+1:0>4}\n")


with open("../data/val.txt", "a") as rf:
    for i in range(40):
        rf.write(f"{i+641:0>4}\n")
