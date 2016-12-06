import sys, os

proj_roots = [
    '/Users/joeliven/repos/carproject',
    '/scratch/cluster/joeliven/carproject',
    '/u/joeliven/repos/carproject',
    ]

for proj_root in proj_roots:
    if proj_root not in sys.path:
        if os.path.exists(proj_root):
            sys.path.append(proj_root)

import numpy as np
import matplotlib.pyplot as plt

def main():

    if len(sys.argv) < 2:
      print ("viz_utils2.py: <raw_path>")
      sys.exit()

    sums = []
    decisions = []
    raw_path = sys.argv[1]
    ct = 0
    with open(raw_path, 'r') as f:
        for line in f:
            if line.startswith('Sum:'):
                ct += 1
                print('ct: %d' % ct)
                # train_acc: 0.851562 	 val_acc: 0.796875 	saving checkpoint to file: /scratch/cluster/joeliven/carproject/models/vgg16/vgg16_a_checkpoint
                splits = line.split(':')
                print('splits')
                print(splits)
                sums.append(float(splits[1].strip()))
            elif line.startswith('prediction'):
                splits = line.split(':')
                print('splits')
                print(splits)
                decision = line.split()[1].strip()
                if decision.startswith('TURN'):
                    decisions.append(True)
                else:
                    decisions.append(False)
    val_accs = np.asarray(sums)
    print('sums.shape[0]')
    print(len(sums))
    # assert train_accs.shape[0] == 100
    epochs_S = []
    epochs_T = []
    sums_S = []
    sums_T = []
    for t in range(len(sums)):
        if decisions[t] == True:
            sums_T.append(sums[t])
            epochs_T.append(t)
        else:
            sums_S.append(sums[t])
            epochs_S.append(t)

    sums_S = np.asarray(sums_S)
    sums_T = np.asarray(sums_T)
    epochs_S = np.asarray(epochs_S)
    epochs_T = np.asarray(epochs_T)

    plt.plot(epochs_S, sums_S, 'bo', epochs_T, sums_T, 'rx')
    # plt.scatter(np.asarray([epochs_S, sums_S]))
    plt.show()


if __name__ == "__main__":
  main()
