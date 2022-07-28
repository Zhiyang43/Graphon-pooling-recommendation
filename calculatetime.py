import os
import pickle
import numpy as np

thisFilename = 'movieGNN-user-001-20211105154321'
saveDirRoot = 'graph-neural-networks-master/experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename)
saveDirVars = os.path.join(saveDir, 'trainVars')
pathToFile = os.path.join(saveDirVars,'trainVarsG00.pkl')
# load : get the data from file
data = pickle.load(open(pathToFile, "rb"))
# loads : get the data from var
print(np.mean(data['timeTrain']['GraphonGNNRegularIntegrationG00']))
print(np.mean(data['timeTrain']['GraphonGNNRegularSamplingG00']))
print(np.mean(data['timeTrain']['SelGNN2LyG00']))

print(np.mean(data['timeTrain']['CoarseningG00']))


# print('Graphon %f +- %f' %(np.mean(data['timeValid']) ,np.std(data['timeValid'])))
# pathToFile = os.path.join(saveDirVars,'SelGNNsprR00trainVars.pkl')
# data = pickle.load(open(pathToFile, "rb"))
# # loads : get the data from var
# print('SelGNNspr %f +- %f' %(np.mean(data['timeValid']), np.std(data['timeValid'])))
# pathToFile = os.path.join(saveDirVars,'SelGNNcrsR00trainVars.pkl')
# data = pickle.load(open(pathToFile, "rb"))
# # loads : get the data from var
# print(' SelGNNcrs %f +- %f' %(np.mean(data['timeValid']), np.std(data['timeValid'])))