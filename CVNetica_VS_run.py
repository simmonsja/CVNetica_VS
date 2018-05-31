import pythonNetica as pyn

import CVNetica_VS_tools as CVT
import numpy as np
import shutil
import itertools
import os, glob, sys, warnings
from neticaUtilTools import vprint

def run_VS(VS_settings):
    # (xml_file, input_vars, output_vars, combks, savefile=None, verboselvl=2, warningstofile=True):    
    ####################
    # Initialize
    ####################
    cdat = pyn.pynetica()
    cdat.pyt.verboselvl = VS_settings['verboselvl']
    
    # read in the problem parameters
    cdat.probpars = CVT.input_parameters(VS_settings)

    # Initialize a pynetica instance/env using password in a text file
    cdat.pyt.start_environment(cdat.probpars.pwdfile)

    # If the warningstofile bool is True then warnings will print to
    # 'warnings.txt' in this. If False, then warnings will print as per
    # default to sys.stdout. This is to avoid warnings that occur due
    # to > or < netica bins which take the range -inf to x or x to inf
    if VS_settings['warningstofile']:
        dir_path = os.path.dirname(os.path.realpath(__file__)) #get current path
        warningsfn = os.path.join(dir_path,'warnings.txt')
        vprint(2,cdat.pyt.verboselvl,'Printing warnings to ' + warningsfn + '\n')
        def tofilewarnings(message, category, filename, lineno, file=warningsfn, line=None):
            warf=open(warningsfn,'a')
            warf.write(warnings.formatwarning(message, category, filename, lineno))
            warf.close()
            # py3 just needs:
            ## print(warnings.formatwarning(message, category, filename, lineno),file=open(warningsfn,'a'))
        warnings.showwarning = tofilewarnings
        # wipe the file to start warnings again
        warf=open(warningsfn,'w')
        warf.write('Printing warnings from CVNetica_VS...')
        warf.close()
        ## print('Printing warnings from CVNetica_VS...',file=open(warningsfn,'w'))

    # --> crank up the memory available if needed
    # cdat.pyt.LimitMemoryUsage(ct.c_double(5.0e16))

    # read in the data from a base cas file
    cdat.read_cas_file(cdat.probpars.baseCAS)

    #create the net with desired links
    vprint(2,cdat.pyt.verboselvl,'*'*5 + 'Deleting unneccessary nodes' + '*'*5 + '\n')
    vprint(2,cdat.pyt.verboselvl,'*'*5 + 'Check net for missing nodes' + '*'*5 + '\n')
    vprint(2,cdat.pyt.verboselvl,'*'*5 + 'Adding links between nodes' + '*'*5 + '\n\n')

    # get all the cominations of inputs specified by combk
    inCombs = []
    for combk in VS_settings['combinations']:
        inCombs.extend(list(itertools.combinations(VS_settings['input_vars'],combk)))

    output_folder = cdat.probpars.scenario.name
    runNets = [] #list of nets to be run
    for output_var in VS_settings['response_vars']:
        for ii, inComb in enumerate(inCombs):
            thisInputvars = list(inComb)
            all_vars_list = thisInputvars + [output_var]
            netOutName = output_folder + 'outputnet_' + output_var + '_comb_' + str(ii) + '.neta'
            #delete nodes not needed
            cdat.deleteExtraNodes(cdat.probpars.baseNET, netOutName, all_vars_list)
            cdat.checkNodes(netOutName, all_vars_list)
            for input_var in thisInputvars:
                cdat.linkNodes(netOutName, netOutName, input_var, output_var)
            runNets.append(CVT.scenario_store(netOutName,thisInputvars,output_var))

    #create pandas dataframe for storing all entries
    pdcols = ['invar_'+_ for _ in VS_settings['input_vars']] + ['outvar_'+_ for _ in VS_settings['response_vars']] + ['input_num', 'output_num']
    metric_skill = ['skMean','rmseM']
    metric_loss = ['logloss','errrate','quadloss']
    perfdata = CVT.performance_store(runNets.__len__(), VS_settings['input_vars'], VS_settings['response_vars'],
                                     metric_skill ,metric_loss)
    totalnets = runNets.__len__()
    
    ### Now run all the nets we have setup!
    for ii,thisNet in enumerate(runNets):
        if cdat.pyt.verboselvl == 2:
            sys.stdout.write('\r')
            percdone = (ii+1)/float(totalnets)
            sys.stdout.write("[%-20s] %d%% - Processing combination: %d of %d" % ('='*int(np.floor(percdone*20)), percdone*100, ii+1, totalnets))
            sys.stdout.flush()
        
        ### reset the net
        #update the input/output vars and output net name
        cdat.probpars.update_node_parameters(thisNet.nodesIn,thisNet.nodesOut)
        cdat.probpars.scenario.name = thisNet.netName[:-5] #remove .neta
        # and reset the netica tests data
        cdat.cleanCVOutput()

        # run the CV part
        # set up the experience node indexing
        cdat.NodeParentIndexing(thisNet.netName, cdat.probpars.baseCAS)

        # create the folds desired
        cdat.allfolds = CVT.all_folds()
        cdat.allfolds.k_fold_maker(cdat.N, cdat.probpars.numfolds)

        # if requested, perform K-fold cross validation
        if cdat.probpars.CVflag:
            vprint(3,cdat.pyt.verboselvl,'\n' * 2 + '#'*20 + '\n Performing k-fold cross-validation for %d folds\n' %(cdat.probpars.numfolds) + '#'*20+'\n' * 2)
            
            # set up for cross validation
            # print('\nSetting up cas files and file pointers for cross validation\n')
            kfoldOFP_Val, kfoldOFP_Cal = cdat.cross_val_setup()
            # now build all the nets
            for cfold in np.arange(cdat.probpars.numfolds):
                vprint(3,cdat.pyt.verboselvl,'#' * 20 + '\n' + '#  F O L D --> {0:d}  #\n'.format(cfold) + '#' * 20)
                
                # rebuild the net
                cname = cdat.allfolds.casfiles[cfold]
                cdat.pyt.rebuild_net(thisNet.netName,
                                     cname,
                                     cdat.probpars.voodooPar,
                                     cname[:-4] + '.neta',
                                     cdat.probpars.EMflag)
                
                # make predictions for both validation and calibration data sets
                vprint(3,cdat.pyt.verboselvl,'*'*5 + 'Calibration predictions' + '*'*5)
                cdat.allfolds.calpred[cfold], cdat.allfolds.calNODES[cfold] = (
                    cdat.predictBayes(cname[:-4] + '.neta',
                                      cdat.allfolds.calN[cfold],
                                      cdat.allfolds.caldata[cfold])
                )
                vprint(3,cdat.pyt.verboselvl,'*'*5 + 'End Calibration predictions' + '*'*5 + '\n\n')

                vprint(3,cdat.pyt.verboselvl,'*'*5 + 'Making Calibration Testing using built-in Netica Functions' + '*'*5 + '\n\n')
                
                # ############### Now run the Netica built-in testing stuff ################
                cdat.PredictBayesNeticaCV(cfold,cname[:-4] + '.neta', 'CAL')
                vprint(3,cdat.pyt.verboselvl,'*'*5 + 'Finished --> Calibration Testing using built-in Netica Functions' + '*'*5 + '\n\n')
                
                vprint(3,cdat.pyt.verboselvl,'*'*5 + 'Start Validation predictions' + '*'*5)
                cdat.allfolds.valpred[cfold], cdat.allfolds.valNODES[cfold] = (
                    cdat.predictBayes(cname[:-4] + '.neta',
                                      cdat.allfolds.valN[cfold],
                                      cdat.allfolds.valdata[cfold]))
                vprint(3,cdat.pyt.verboselvl,'*'*5 + 'End Validation predictions' + '*'*5 + '\n\n')
                 
                vprint(3,cdat.pyt.verboselvl,'*'*5 + 'Making Validation Testing using built-in Netica Functions' + '*'*5 + '\n\n')
                # ############### Now run the Netica built-in testing stuff ################
                cdat.PredictBayesNeticaCV(cfold, cname[:-4] + '.neta', 'VAL')
                vprint(3,cdat.pyt.verboselvl,'*'*5 + 'Finished --> Validation Testing using built-in Netica Functions' + '*'*5 + '\n\n')
            
            vprint(3,cdat.pyt.verboselvl,'Write out validation')
            cdat.PredictBayesPostProcCV(cdat.allfolds.valpred,
                                        cdat.probpars.numfolds,
                                        kfoldOFP_Val,
                                        'Validation',
                                        cdat.NeticaTests['VAL'])
                
            vprint(3,cdat.pyt.verboselvl,'Write out calibration')     
            cdat.PredictBayesPostProcCV(cdat.allfolds.calpred,
                                        cdat.probpars.numfolds,
                                        kfoldOFP_Cal,
                                        'Calibration',
                                        cdat.NeticaTests['CAL'])
            kfoldOFP_Cal.close()
            kfoldOFP_Val.close()
            # summarize over all the folds to make a consolidated text file
            cdat.SummarizePostProcCV()
            #store the output stats in python
            perfdata.add_perf_data(ii,cdat,thisNet,metric_skill,metric_loss)

            #clean the output folder of files
            #adjusted - failure is not an option.
            try:
                for fn in glob.glob(output_folder + '*_fold_*cas'):
                    os.remove(fn)
                for fn in glob.glob(output_folder + '*_folds*dat'):
                    os.remove(fn)
                for fn in glob.glob(output_folder + '*_fold_*neta'):
                    os.remove(fn)
            except:
                pass

    vprint(2,cdat.pyt.verboselvl, '\n')
    vprint(2,cdat.pyt.verboselvl, 'Netica runs completed.')
    # Done with Netica so shut it down
    cdat.pyt.CloseNetica()

    # first need to sanitize away any ctypes/Netica pointers
    cdat.pyt.sanitize()

    if VS_settings['output_file']:
        vprint(1,cdat.pyt.verboselvl, 'Saving output to %s.' % VS_settings['output_file'])
        perfdata.df.to_csv(VS_settings['output_file'])
    
    return runNets, perfdata

