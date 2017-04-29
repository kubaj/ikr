import sys
from keras.models import load_model
import ikrdata
import pdb
import numpy as np

if len(sys.argv) < 2:
    print("usage: speechmap-eval.py [model-file]")
    exit(42)

model = load_model(sys.argv[1])

succ = 0
total = 0

edata, elabel = ikrdata.get_speech_set(ikrdata.DEV_DIR, pack=True)
for data, label in zip(edata, elabel):
    #print()
    #print(len(data), "maps, expected label:", label)
    #data = data.reshape(data.shape[0], 26, 26, 1)
    pred = model.predict(data)
    #print("Predicted labels: ", end='')
    prob = [0 for _ in range(31)]
    for p in pred:
        amax = np.argmax(p)
        prob[int(amax)] += 1
        # print(amax, end=' ')
    #print()
    #print("Counts:", prob)
    win = np.argmax(prob)
    #print("Argmax:", win)

    total += 1
    if win == label: succ += 1
    #print('✅' if win == label else '❌')


print(total, "samples, accuracy:", succ/total)
