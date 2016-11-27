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
      print ("viz_utils.py: <raw_path>")
      sys.exit()

    train_accs = []
    val_accs = []
    raw_path = sys.argv[1]
    ct = 0
    with open(raw_path, 'r') as f:
        for line in f:
            if line.startswith('train_acc:'):
                ct += 1
                print('ct: %d' % ct)
                # train_acc: 0.851562 	 val_acc: 0.796875 	saving checkpoint to file: /scratch/cluster/joeliven/carproject/models/vgg16/vgg16_a_checkpoint
                splits = line.split(' ')
                print('splits')
                print(splits)
                train_accs.append(float(splits[1].strip()))
                val_accs.append(float(splits[4].strip()))
    train_accs = np.asarray(train_accs)
    val_accs = np.asarray(val_accs)
    print('train_accs.shape[0]')
    print(train_accs.shape[0])
    assert train_accs.shape[0] == 100
    epochs = np.arange(train_accs.shape[0]) + 1
    plt.plot(epochs, train_accs, 'b', epochs, val_accs, 'r')
    plt.show()


if __name__ == "__main__":
  main()
