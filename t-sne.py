import pandas as pd
import numpy as np
import os,time
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



fd_folder='All/fd/fd_groundtruthsegmentation.csv'
dd_folder='All/dd/dd_groundtruthsegmentation.csv'
sd_folder='All/sd/sd_groundtruthsegmentation.csv'
cd_folder='All/cd/cd_groundtruthsegmentation.csv'
null_folder='All/null/null_groundtruthsegmentation.csv'

fd,dd=pd.read_csv(fd_folder),pd.read_csv(dd_folder)
sd,cd,null=pd.read_csv(sd_folder),pd.read_csv(cd_folder),pd.read_csv(null_folder)

# fd_dd=pd.concat([fd,dd])
fd_dd = fd
sd_cd=pd.concat([sd,cd,null])
fd_dd['label']=1
sd_cd['label']=0
together=pd.concat([fd_dd,sd_cd])

time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=30,n_iter=1000)
data_tsne=tsne.fit_transform(together.values[:,:-1])
print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_tsne[:,0], data_tsne[:,1],data_tsne[:,2], c=together.values[:,-1])
# plt.colorbar(ticks=range(2))
plt.show()