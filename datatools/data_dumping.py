import sys
import uproot

import os
import numpy as np
import pandas as pd

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path',  type=str, default="./data")
parser.add_argument('--size', type=int, default=64)
parser.add_argument('--grid',action="store_true")
parser.add_argument('--dir',   type=str, default="ana")
parser.add_argument('--tree',  type=str, default="hgc")

args = parser.parse_args()

max_perlayer = args.size
number_layers = 50

variableName = [
            'event','run','lumi',
            'cluster2d_layer',
            'cluster2d_energy',
            'cluster2d_eta',
            'cluster2d_phi',
            'cluster2d_pt',
            'cluster2d_x',
            'cluster2d_y',
            'cluster2d_z',
            'cluster2d_nhitCore',
            'cluster2d_nhitAll',
            'gen_energy',
            'gen_pdgid',
            'gen_daughters',
            'gen_phi',
            'gen_eta',
            ]
newVars =["run","lumi","event","x","y","z","r","layer","E","nCore","nHits","id","genDR","gen_phi","gen_eta","phi","eta","pid"]

print("Starting data production")
path = args.path
#name = "/pion.root"
files = [f for f in os.listdir(args.path) if f.endswith("root")]

for name in tqdm(files):

    try:
        df = uproot.open(args.path + "/" + name)[args.dir][args.tree].pandas.df(variableName,flatten=False)


        num_events = np.unique(df["event"].values).shape[0]
        xs = df["cluster2d_x"].values
        ys = df["cluster2d_y"].values
        zs = df["cluster2d_z"].values
        es = df["cluster2d_energy"].values
        ps = df["cluster2d_pt"].values
        nh = df["cluster2d_nhitAll"].values
        nc = df["cluster2d_nhitCore"].values
        ll = df["cluster2d_layer"].values
        ee = df["event"].values
        lu = df["lumi"].values
        ru = df["run"].values

        sizes = [x.shape[0] for x in xs]

        indices = [np.full((a[0]),a[1]) for a in zip(sizes,range(len(sizes)))]
        #gendau = [np.full((a[0]),a[1]) for a in zip(sizes,df["gen_daughters"].values)]

        cphi = df["cluster2d_phi"].values
        ceta = df["cluster2d_eta"].values
        gpid = df["gen_pdgid"].values

        gphi = [np.full((a[0]),a[1]) for a in zip(sizes,df["gen_phi"].values)]
        geta = [np.full((a[0]),a[1]) for a in zip(sizes,df["gen_eta"].values)]
        gpid = [np.full((a[0]),a[1]) for a in zip(sizes,df["gen_pdgid"].values)]

        rs = [np.sqrt(f[0]**2+f[1]**2) for f in zip(xs,ys)]

        drs = [np.sqrt((a[0]-a[1])**2 + (a[2]-a[3])**2) for a in zip(gphi,cphi,geta,ceta)]

        XS = np.array([item for sublist in xs for item in sublist])
        YS = np.array([item for sublist in ys for item in sublist])
        ZS = np.array([item for sublist in zs for item in sublist])
        RS = np.array([item for sublist in rs for item in sublist])
        LL = np.array([item for sublist in ll for item in sublist])
        ES = np.array([item for sublist in es for item in sublist])
        NC = np.array([item for sublist in nc for item in sublist])
        NH = np.array([item for sublist in nh for item in sublist])
        II = np.array([item for sublist in indices for item in sublist])
        DRS = np.array([item for sublist in drs for item in sublist])
        GPHI = np.array([item for sublist in gphi for item in sublist])
        GETA = np.array([item for sublist in geta for item in sublist])
        GPID = np.array([item for sublist in gpid for item in sublist])
        CPHI = np.array([item for sublist in cphi for item in sublist])
        CETA = np.array([item for sublist in ceta for item in sublist])

        #GENDAU = np.array([item for sublist in gendau for item in sublist])

        SS = [np.full((s,),s) for s in sizes]
        EE = [np.full((s,),i) for i,s in zip(ee,sizes)]
        LU = [np.full((s,),i) for i,s in zip(lu,sizes)]
        RU = [np.full((s,),i) for i,s in zip(ru,sizes)]


        SS = np.array([item for sublist in SS for item in sublist])
        EE = np.array([item for sublist in EE for item in sublist])
        LU = np.array([item for sublist in LU for item in sublist])
        RU = np.array([item for sublist in RU for item in sublist])

        datas = np.vstack((RU,LU,EE,XS,YS,ZS,RS,LL,ES,NC,NH,II,DRS,GPHI,GETA,CPHI,CETA,GPID)).T

        df = pd.DataFrame(datas,columns=newVars)
        df = df.sort_values(["id","layer","event"],ascending=[True,True,False]).reset_index(drop=True)
        #
        theIndex = list(df.groupby(["id","layer","event"]).indices.values())
        theIndex = np.array([item for sublist in theIndex for item in sublist[:min(len(sublist),args.size)]])
        df = df.iloc[theIndex]

        layer_sizes = df.groupby(["id","layer"]).size().values.tolist()
        layer_places = np.cumsum(layer_sizes)

        startes = np.array( [0] + list(layer_places[:-1]))
        layers = df["layer"].values[startes]
        ids = df["id"].values[startes]

        finishes = np.array(list(startes[1:]) +[len(df)])
        SSS = np.vstack((startes,finishes)).T

        hitIds = [[j +(n-1)*max_perlayer + max_perlayer*number_layers*e for j in range(s[1]-s[0])] for n,s,e in zip(layers,SSS,ids)]

        hitIds = np.array([item for sublist in hitIds for item in sublist])
        df.loc[:,"hitIds"] = hitIds
        df.to_hdf(args.path + "/" + name[:-4] + "h5","data",complevel=0)
        if not args.grid:
            continue

        df = df.set_index(hitIds.astype(int))
        bigMask = np.zeros((num_events*number_layers*max_perlayer,len(df.columns)))
        bigDF = pd.DataFrame(bigMask,columns=df.columns)

        fakeHit = [ [j for j in range(1,max_perlayer+1)] for i in range(number_layers*num_events)]
        fakeHit = np.array([item for sublist in fakeHit for item in sublist])

        fakeLayer = [ np.full(max_perlayer,i) for j in range(num_events) for i in range(number_layers)]

        fakeLayer = np.array([item for sublist in fakeLayer for item in sublist])

        fakeEvent = [ np.full(max_perlayer*number_layers,i) for i in range(num_events)]

        fakeEvent = np.array([item for sublist in fakeEvent for item in sublist])

        bigDF["layer"] = fakeLayer
        bigDF["id"] = fakeEvent
        bigDF["hitIds"] = fakeHit

        bigDF.iloc[df.index] = df
        bigDF.to_hdf(args.path + "/" + name[:-5] + "grid"+str(max_perlayer)+".h5","data",complevel=0)

    except:
        print(name + " not loaded. Some issue.")
