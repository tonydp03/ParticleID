from root_pandas import read_root
from pylab import *
import pandas as pd
from tqdm import trange

class NtupleReader():
    def __init__(self, inputFileName, n=-1, tqdmLabel=None):
        
        self.inputFileName = inputFileName

        self.n = n

        if tqdmLabel is None:
            self.tqdmLabel = 'Making Dataset'
        else:
            self.tqdmLabel = tqdmLabel

        self.variableName = [
            'cluster2d_layer',
            'cluster2d_energy',
            'cluster2d_eta',
            'cluster2d_phi',
            'gen_energy',
            'gen_pdgid'
            ]

    def makeDataset(self,outputFileName=None):
        events  = pd.DataFrame(self.getEvents())
        if not outputFileName is None:
            events.to_hdf(outputFileName, 'event_clusters', append=False)
        return events

    def getEvents(self):

        df = read_root(self.inputFileName, 'ana/hgc', columns=self.variableName )

        if self.n <= 0:
            numberOfEvents = len(df)
        else:
            numberOfEvents = self.n
        # print('Nevts ', numberOfEvents)

        for ievt in trange(numberOfEvents, desc=self.tqdmLabel, unit=' events', ncols=100):
            event = df.loc[ievt]

            # get feature
            feature = []
            for l in range(1,51,1):
                ## select good layer clusters
                slt = event.cluster2d_layer==l 
                slt &= event.cluster2d_eta>0
                slt &= event.cluster2d_energy>0

                layer_feature = [event['cluster2d_{}'.format(var)][slt] for var in ['eta','phi','energy']]
                layer_feature = np.array(layer_feature).T
                layer_feature = layer_feature[layer_feature[:,2].argsort()][::-1]
                layer_feature_pixels = []
                for cl in range(30):
                    if cl < layer_feature.shape[0]:
                        layer_feature_pixels.append(layer_feature[cl])
                    else:
                        layer_feature_pixels.append(np.zeros(3))
                layer_feature_pixels = np.array(layer_feature_pixels) 
                feature.append(layer_feature_pixels)

            # get label
            ## select good genpart
            gen_pid = np.abs(event.gen_pdgid)[0]
            gen_energy = event.gen_energy[0]

            if gen_pid ==22:
                label=0
            if gen_pid == 11:
                label=1
            if gen_pid == 13:
                label=2
            if gen_pid == 15:
                label=3
            if gen_pid == 111:
                label=4
            if gen_pid == 211:
                label=5
                
            # save feature and label
            eventFeatureLabel = {'feature':feature, 'label':label, 'gen_energy':gen_energy}

            yield eventFeatureLabel

if __name__ == '__main__':
    # configuration
    directory = "data/"

    # read data from root files
    df = []
    for pid in ['gamma', 'electron', 'muon', 'tau', 'pion_n', 'pion_c']:
        inputFileName = directory + '4_{}_NTUPLE.root'.format(pid)
        # print("processing " + inputFileName)
        df_pid = NtupleReader(inputFileName, -1, pid).makeDataset()
        # print(df_pid)
        df.append(df_pid)

    df = pd.concat(df,ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # partition dataset to train and test
    train = df[0:int(0.9*len(df))].copy().reset_index(drop=True)
    test = df[int(0.9*len(df)):].copy().reset_index(drop=True)
    # save to hdf file
    train.to_pickle(directory + "pkl/dataset_train.pkl")
    # train.to_hdf(directory + "hdf/dataset_train.h5", "training_data", append=False)
    test.to_pickle(directory + "pkl/dataset_test.pkl")
    # test.to_hdf(directory + "hdf/dataset_test.h5", "test_data", append=False)
