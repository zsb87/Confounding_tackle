import pandas as pd
import numpy as np
import os,time
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler



fd_folder='All/fd/fd_groundtruthsegmentation.csv'
dd_folder='All/dd/dd_groundtruthsegmentation.csv'
sd_folder='All/sd/sd_groundtruthsegmentation.csv'
cd_folder='All/cd/cd_groundtruthsegmentation.csv'
null_folder='All/null/null_groundtruthsegmentation.csv'

fd,dd=pd.read_csv(fd_folder),pd.read_csv(dd_folder)
sd,cd,null=pd.read_csv(sd_folder),pd.read_csv(cd_folder),pd.read_csv(null_folder)

fd_dd=pd.concat([fd,dd])
# fd_dd = fd
# sd_cd=pd.concat([sd,cd,null])
sd_cd_null=pd.concat([sd,cd,null])

sd['label']=2
fd_dd['label']=1
null['label']=0

together=pd.concat([fd_dd,sd,null])

X = together.values[:,:-1]
y = together.values[:,-1]
names = together.columns

# X,y = SMOTE().fit_sample(X,y)	

## save to file
# together = pd.DataFrame(data = np.hstack((X,y[:,np.newaxis])),columns=names)
# together.to_csv('All/fd_dd_null_groundtruthsegmentation_SMOTE.csv',index=None)

time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=30,n_iter=1000)
data_tsne=tsne.fit_transform(X)
print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_tsne[:,0], data_tsne[:,1],data_tsne[:,2], c=y)
# plt.colorbar(ticks=range(2))
plt.show()

