from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()
import sys
import datetime, time
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

names = ["amuon"  ,"aproton"  ,"electron"  ,"muon"  ,"photon"  ,"pion_0"  ,"pion_m"  ,"pion_p"  ,"positron"  ,"proton"  ,"tau_m"  ,"tau_p","kaons","neutron"]

NJOBS = 200
TIMEPER100EVTs = 25
EVTPERJOB = 1000
TIME = int(EVTPERJOB*TIMEPER100EVTs/100)
step = 'GEN-MINIAODSIM'
nEvents = NJOBS*EVTPERJOB

myname='electron'
job_label = myname + "_HGCalPID_"
myrun=myname + ".py"

if myname not in names:
        print("%s ??"%myname)
        print("Not a good name for a particle...")
        sys.exit()
step3File = '3_muon.root'

config.General.requestName = step+'_'+job_label+'_'+st
config.General.transferOutputs = True
config.General.transferLogs = False
config.General.workArea = 'crab_projects'

config.JobType.pluginName = 'PrivateMC'
config.JobType.psetName = myrun
config.JobType.inputFiles = [myrun,'step2.py','step3.py']
config.JobType.disableAutomaticOutputCollection = True

config.JobType.eventsPerLumi= 100
config.JobType.scriptExe = 'run.sh'
config.JobType.numCores = 8
config.JobType.maxMemoryMB = 5000
config.JobType.maxJobRuntimeMin = TIME

config.JobType.outputFiles = [step3File] #[step3Mini,trackRoot,trackHDF]#,stepCFile]

config.Data.outputPrimaryDataset = myname
config.Data.splitting = 'EventBased'
config.Data.unitsPerJob = EVTPERJOB#nEvents # the number of events here must match the number of events in the externalLHEProducer

config.Data.totalUnits = nEvents
config.Data.outLFNDirBase = '/store/user/%s/' % (getUsernameFromSiteDB())


config.Data.publication = False

config.JobType.allowUndistributedCMSSW = True
config.Site.storageSite = 'T2_IT_Bari'
config.Site.blacklist = ["T2_BR_*","T2_IN_*","T2_CN_*"]#,"T2_US_*","T3_US_*"]
#config.Site.whitelist = sites #['T2_IT_*','T2_CH_*','T2_GE_*','T2_FR_*','T2_ES_*','T2_UK_*']
