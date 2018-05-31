import numpy as np
import copy
import pandas as pd

###################
# Tools for reading XML input file
###################

class netica_scenario:
    def __init__(self):
        self.name = None
        self.nodesIn = None
        self.response = None

class stats_data_store:
    def __init__(self):
        self.NeticaTests = None
        self.allfolds = None

class scenario_store:
    ### For storing multiple scenarios to be run
    def __init__(self,netName,nodesIn,nodesOut):
        self.netName = netName
        self.nodesIn = nodesIn
        self.nodesOut = nodesOut
        self.statsData = stats_data_store()

    def add_stats_data(self,cnet):
        self.statsData.NeticaTests = copy.deepcopy(cnet.NeticaTests)
        self.statsData.allfolds = copy.deepcopy(cnet.allfolds)

class performance_store:
    ### For storing the performance of multiple scenarios run
    def __init__(self,initlen,input_vars,output_vars,
                 metric_skill,metric_loss):
        pdcols = ['invar_'+_ for _ in input_vars] + ['outvar_'+_ for _ in output_vars] + ['input_num', 'output_num']
        metric_stats = ['mean','median','std']
        for met in metric_skill+metric_loss:
            for stat in metric_stats:
                pdcols.append('stat_'+met+'_'+stat)
        self.df = pd.DataFrame(index=np.arange(initlen),columns=pdcols)

    def add_perf_data(self,ii,cnet,scenario,skill,loss):
        #first mark the variables used in this scenario
        for var in list(self.df.filter(regex=("invar_.*"))):
            #check inputs
            if var.split('invar_')[1] in scenario.nodesIn:
                self.df.loc[ii,var] = True
            else:
                self.df.loc[ii,var] = False
        #check outputs
        for var in list(self.df.filter(regex=("outvar_.*"))):
            #check inputs
            if var.split('outvar_')[1] in scenario.nodesOut:
                self.df.loc[ii,var] = True
            else:
                self.df.loc[ii,var] = False
        self.df.loc[ii,'input_num'] = 1 if not isinstance(scenario.nodesIn, list) else scenario.nodesIn.__len__()
        self.df.loc[ii,'output_num'] = 1 if not isinstance(scenario.nodesOut, list) else scenario.nodesOut.__len__()

        #now lets get the stats data!
        for sk in skill:
            vals = []
            for _ in cnet.allfolds.valpred:
                thisval = getattr(_[scenario.nodesOut].stats,sk)
                cnt = 0
                while not isinstance(thisval, float):
                    if cnt > 5:
                        #dont want any infinit loops!
                        raise NameError('Cant process skill metric %s from netica output' % (sk))
                    thisval = thisval[0]
                    cnt += 1
                vals.append(thisval)
            self.df.loc[ii,'stat_'+sk+'_mean'] = np.mean(vals)
            self.df.loc[ii,'stat_'+sk+'_median'] = np.median(vals)
            self.df.loc[ii,'stat_'+sk+'_std'] = np.std(vals)

        for lo in loss:
            vals = [getattr(_,lo)[scenario.nodesOut] for _ in cnet.NeticaTests['VAL']]
            self.df.loc[ii,'stat_'+lo+'_mean'] = np.mean(vals)
            self.df.loc[ii,'stat_'+lo+'_median'] = np.median(vals)
            self.df.loc[ii,'stat_'+lo+'_std'] = np.std(vals)

class input_parameters:
    # class and methods to read and parse XML input file
    def __init__(self, infile):
        #get the defaults
        paramdict = self.get_defaults()
        paramdict.update(infile)

        #set the params
        self.baseNET = paramdict['baseNET']
        self.baseCAS = paramdict['baseCAS']
        self.pwdfile = paramdict['pwdfile']

        self.CVflag = paramdict['CVflag']
        self.numfolds = paramdict['numfolds']
        self.scenario = netica_scenario()
        self.scenario.name = paramdict['working_dir']
        self.scenario.nodesIn = []
        self.scenario.nodesOut = []

        self.EMflag = paramdict['EMflag']
        self.voodooPar = paramdict['voodooPar']

    def update_node_parameters(self,nodesIn,nodesOut):
        if not isinstance(nodesIn,list):
            nodesIn = [nodesIn]
        if not isinstance(nodesOut,list):
            nodesOut = [nodesOut]

        self.scenario.nodesIn = []
        for cv in nodesIn:
            self.scenario.nodesIn.append(cv)
        self.scenario.response = []
        for cr in nodesOut:
            self.scenario.response.append(cr)

        self.CASheader = list(self.scenario.nodesIn)
        self.CASheader.extend(self.scenario.response)

    def get_defaults(self):
        #get default settings
        default_filled = {
            #run_settings
            'verboselvl': 2,
            'warningstofile': True,

            # control_data
            'baseNET': '',
            'baseCAS': '',
            'pwdfile': 'NA.txt',

            # kfold_data
            'CVflag': True,
            'numfolds': 5,

            #scenario
            'working_dir': '',
            'input_vars': [''],
            'response_vars': [''],
            'combinations': [2],
            'output_file': '',

            #learnCPTdata
            'voodooPar': 1,
            'EMflag': False,
        }
        return default_filled

###################
# Tools for k-fold setup
###################
class all_folds:
    # a class containing leftout and retained indices for cross validation

    def __init__(self):
        self.leftout = list()
        self.retained = list()
        self.casfiles = list() #filenames for calibration data (retained indices only)
        self.caldata = list()  # calibration data (same as written to the calibration case file)
        self.valdata = list()  # validation data (the data left out -- will be used to calc predictions)
        # calibration and validation output from making predictions
        self.calpred = list()
        self.valpred = list()
        self.calNODES = list()
        self.valNODES = list()
        self.valN = list()
        self.calN = list()
        self.numfolds = None

    def k_fold_maker(self,n,k):
        # k_fold index maker
        # a m!ke@usgs joint
        # mnfienen@usgs.gov
        # k_fold_maker(n,k,allfolds)
        # input:
        #   n is the length of the sequence of indices
        #   k is the number of folds to split it into
        #   allfolds is an all_folds class
        # returns an all_folds with each member having k elements
        # allfolds.leftout[i] is the left out indices of fold i
        # allfolds.retained[i] is the kept indices of fold i
        currinds = np.arange(n)
        inds_per_fold = np.int(np.floor(n/k))
        dingleberry = np.remainder(n, k)
        for i in np.arange(k-1):
            allinds = currinds.copy()
            np.random.shuffle(currinds)
            self.leftout.append(currinds[0:inds_per_fold].copy())
            self.retained.append(np.setdiff1d(np.arange(n), self.leftout[i]))
            currinds = currinds[inds_per_fold:]
        self.leftout.append(currinds)
        self.retained.append(np.setdiff1d(np.arange(n), self.leftout[-1]))
        self.numfolds = k

#################
# Error classes
#################

# -- cannot open an input file
class FileOpenFail(Exception):
    def __init__(self,filename):
        self.fn = filename
    def __str__(self):
        return('\n\nCould not open %s.' %(self.fn))
