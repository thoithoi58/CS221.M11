def read(file):
    with open(file, 'r', encoding='utf8') as f:
        file = f.read().splitlines()
    data = [[] for _ in range(len(file))]
    for idx, i in enumerate(file):
        a = i.split()
        for j in a:
            tmp = (j.rsplit('/',1))
            data[idx].append((tmp[0], tmp[1]))
    return data

train_set = read('corpus/train.txt')
test_set = read('corpus/test.txt')