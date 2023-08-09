import pandas as pd
import matplotlib.pyplot as plt
import os

summary_root = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/output/oversample-sqrt-pretrained-uniformer_small_original-mixup-excluded6-cbloss-smooth-aug-val-resample_space4')

summary_file = pd.read_csv(os.path.join(summary_root,'summary.csv'))

x = summary_file['epoch']
y1 = summary_file['train_loss']
y2 = summary_file['eval_loss']

plt.plot(x,y1,'blue')
plt.plot(x,y2,'orange')

# add labels and title
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss_graph')

# save plot to file
plt.savefig(os.path.join(summary_root,'loss_graph.png'))

# display plot
plt.show()