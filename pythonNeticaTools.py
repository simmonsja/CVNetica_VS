import numpy as np
import os
import ctypes as ct
import platform
import pythonNeticaConstants as pnC
import cthelper as cth
from neticaUtilTools import vprint

class statestruct:
    def __init__(self):
        self.obj = None
        self.name = None
        self.numeric = None

class experience:
    def __init__(self):
        self.parent_names = list()
        self.parent_states = None
        self.node_experience = list()
        self.case_experience = list()

class nodestruct:
    def __init__(self):
        self.name = None
        self.title = None
        self.beliefs = None
        self.Nbeliefs = None
        self.Likelihood = None
        self.continuous = False
        self.state = []

class pyneticaTools:
    def __init__(self):
        self.n = None
        self.mesg = ct.create_string_buffer(1024)
        self.env = None
        self.verboselvl = 3
        
    def sanitize(self):
        vprint(2,self.verboselvl,'Sanitizing pynetica object to remove pointers')
        # code to strip out all ctypes information from SELF to 
        # allow for pickling
        self.n = None
        self.mesg = None
        self.env = None
        
    def start_environment(self, licfile):
        # read in the license file information
        self.licensefile = licfile
        if os.path.exists(self.licensefile):
            self.license = open(self.licensefile, 'r').readlines()[0].strip().split()[0]
        else:
            vprint(2,self.verboselvl,"Warning: License File [{0:s}] not found.\n".format(self.licensefile) +
                   "Opening Netica without licence, which will limit size of nets that can be used.\n" +
                   "Window may become unresponsive.")
            self.license = None         
        self.NewNeticaEnviron()
            
    #############################################
    # Major validation and prediction functions #
    #############################################

    def OpenNeticaNet(self,netName):
        '''
        Open a net identified by netName.
        Returns a pointer to the opened net after it is compiled
        '''
        # meke a streamer to the Net file
        cname = netName
        if '.neta' not in netName:
            cname += '.neta'
        net_streamer = self.NewFileStreamer(ct.c_char_p(cname.encode()))
        # read in the net using the streamer        
        cnet = self.ReadNet(net_streamer)
        # remove the input net streamer
        self.DeleteStream(net_streamer)  
        self.CompileNet(cnet)
        return cnet
    
    def rebuild_net(self, NetName, newCaseFile, voodooPar, outfilename, EMflag=False):
        '''
         rebuild_net(NetName,newCaseFilename,voodooPar,outfilename)
         a m!ke@usgs joint <mnfienen@usgs.gov>
         function to build the CPT tables for a new CAS file on an existing NET
         (be existing, meaning that the nodes, edges, and bins are dialed)
         INPUT:
               NetName --> a filename, including '.neta' extension
               newCaseFilename --> new case file including '.cas' extension
               voodooPar --> the voodoo tuning parameter for building CPTs
               outfilename --> netica file for newly build net (including '.neta')
               EMflag --> if True, use EM to learn from casefile, else (default)
                         incorporate the CPT table directly
         '''   
        # create a Netica environment
        vprint(3,self.verboselvl,'Rebuilding net: {0:s} using Casefile: {1:s}'.format(NetName, newCaseFile))
        # make a streamer to the Net file
        net_streamer = self.NewFileStreamer(ct.c_char_p(NetName.encode()))
        # read in the net using the streamer        
        cnet = self.ReadNet(net_streamer)
        # remove the input net streamer
        self.DeleteStream(net_streamer)  
        self.CompileNet(cnet)      
        #get the nodes and their number
        allnodes = self.GetNetNodes(cnet)
        numnodes = self.LengthNodeList(allnodes)

        # loop over the nodes deleting CPT
        for cn in np.arange(numnodes):
            cnode = self.NthNode(allnodes,ct.c_int(cn))
            self.DeleteNodeTables(cnode)
        # make a streamer to the new cas file
        new_cas_streamer = self.NewFileStreamer(ct.c_char_p(newCaseFile.encode()))

        if EMflag:
            vprint(3,self.verboselvl,'Learning new CPTs using EM algorithm')
            # to use EM learning, must first make a learner and set a couple options
            newlearner = self.NewLearner(pnC.learn_method_bn_const.EM_LEARNING)
            self.SetLearnerMaxTol(newlearner, ct.c_double(1.0e-6))
            self.SetLearnerMaxIters(newlearner, ct.c_int(1000))
            # now must associate the casefile with a caseset (weighted by unity)
            newcaseset = self.NewCaseset(ct.c_char_p(b'currcases'))
            self.AddFileToCaseset(newcaseset, new_cas_streamer, 1.0)
            self.LearnCPTs(newlearner, allnodes, newcaseset, ct.c_double(voodooPar))
            self.DeleteCaseset(newcaseset)
            self.DeleteLearner(newlearner)
        else:
            vprint(3,self.verboselvl,'Learning new CPTs using ReviseCPTsByCaseFile')
            self.ReviseCPTsByCaseFile(new_cas_streamer, allnodes, ct.c_double(voodooPar))
        outfile_streamer = self.NewFileStreamer(ct.c_char_p(outfilename.encode()))
        self.CompileNet(cnet)

        outfile_streamer = self.NewFileStreamer(ct.c_char_p(outfilename.encode()))
        vprint(3,self.verboselvl,'Writing new net to: %s' %(outfilename))
        self.WriteNet(cnet,outfile_streamer)
        self.DeleteStream(outfile_streamer)  
        self.DeleteNet(cnet)

    def ReadNodeInfo(self, netName):
        '''
        Read in all information on beliefs, states, and likelihoods for all 
        nodes in the net called netName
        '''
        # open the net stored in netName
        cnet = self.OpenNeticaNet(netName)
        #get the nodes and their number
        allnodes = self.GetNetNodes(cnet)
        numnodes = self.LengthNodeList(allnodes)
        vprint(3,self.verboselvl,'Reading Node information from net --> {0:s}'.format(netName))
        cNETNODES = dict()
        # loop over the nodes
        for cn in np.arange(numnodes):
            cnode = self.NthNode(allnodes, ct.c_int(cn))
            cnodename = cth.c_char_p2str(self.GetNodeName(cnode))
            cNETNODES[cnodename] = nodestruct()
            cNETNODES[cnodename].name = cth.c_char_p2str(self.GetNodeName(cnode))
            cNETNODES[cnodename].title = cth.c_char_p2str(self.GetNodeTitle(cnode))
            vprint(3,self.verboselvl,'   Parsing node --> %s' %(cNETNODES[cnodename].title))
            cNETNODES[cnodename].Nbeliefs = self.GetNodeNumberStates(cnode)
            cNETNODES[cnodename].beliefs = cth.c_float_p2float(
                self.GetNodeBeliefs(cnode),
                cNETNODES[cnodename].Nbeliefs)
            cNETNODES[cnodename].likelihood = cth.c_float_p2float(
                self.GetNodeLikelihood(cnode),
                cNETNODES[cnodename].Nbeliefs)
            cNETNODES[cnodename].levels = cth.c_double_p2float(
                self.GetNodeLevels(cnode),
                cNETNODES[cnodename].Nbeliefs + 1)

            # loop over the states in each node
            for cs in range(cNETNODES[cnodename].Nbeliefs):
                cNETNODES[cnodename].state.append(statestruct())
                cNETNODES[cnodename].state[-1].name = cth.c_char_p2str(
                    self.GetNodeStateName(cnode,ct.c_int(cs)))   
       
        self.DeleteNet(cnet)
        return cNETNODES
    
    def ConfusionMatrix(self,ctester,cnode):
        '''
        Makes a confusion matrix for a particular node specified by name in cnode
        within the tester environment laid out in ctester
        '''
        numstates = self.GetNodeNumberStates(cnode)
        
        confusion_matrix = np.zeros((numstates,numstates))
        for a in np.arange(numstates):
            for p in np.arange(numstates):
                confusion_matrix[a,p] = self.GetTestConfusion(ctester,cnode,ct.c_int(p),ct.c_int(a))
        return confusion_matrix

    def ExperienceAnalysis(self,cn,cnet):
        '''
        calculate the experience for the node named in cn
        '''
        cnex = experience()
        # get a list of the parents of the node
        testnode = self.GetNodeNamed(ct.c_char_p(cn.encode()),cnet)
        #start a list for the cartesian sum of node states
        allstates = list()
        cparents = self.GetNodeParents(testnode)    
        numnodes = self.LengthNodeList(cparents)
        for cp in np.arange(numnodes):
            # append the name to the list of returned names
            cnode = self.NthNode(cparents,ct.c_int(cp))
            cnex.parent_names.append(cth.c_char_p2str(self.GetNodeName(cnode)))
            # find the number of states for each parent
            allstates.append(np.arange(self.GetNodeNumberStates(
                self.NthNode(cparents,ct.c_int(cp)))))
        if numnodes > 1:
            cnex.parent_states = self.cartesian(allstates)
        else:
            cnex.parent_states = allstates
        for cs in cnex.parent_states:
            cnex.node_experience.append(self.GetNodeExperience(
                testnode,cs.ctypes.data_as(ct.POINTER(ct.c_int))))
        cnex.node_experience = np.array(cnex.node_experience)
        # change the null pointers (meaning 
        cnex.node_experience[cnex.node_experience<1]=0.0
        
        return cnex

    ###################################
    # Key helper functions for Netica #   
    ###################################

    def NewNeticaEnviron(self):
        '''
        create a new Netica environment based on operating system
        '''
        # first access the .dll or .so in the same directory as CVNetica
        try:
            if 'window' in platform.system().lower():
                self.n = ct.windll.LoadLibrary(os.path.join(os.path.dirname(__file__),'Netica.dll'))
            else:
                self.n = ct.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libnetica.so'))
        except:
            raise(dllFail(platform.system()))
        
        # next try to establish an environment for Netica
        # need to be sure to specify argument and return types to send None
        self.n.NewNeticaEnviron_ns.argtypes = [ct.c_char_p, ct.c_void_p, ct.c_char_p]
        self.n.NewNeticaEnviron_ns.restype = ct.c_void_p
        self.env = self.n.NewNeticaEnviron_ns(self.license, None, None)

        # try to intialize Netica
        self.n.InitNetica2_bn.argtypes = [ct.c_void_p,ct.c_char_p]
        self.n.InitNetica2_bn.restype = ct.c_int
        res = self.n.InitNetica2_bn(self.env, self.mesg)
        # now check the initialisation
        if res >= 0:
            vprint(1,self.verboselvl,'\n'*2 + '#' * 40 + '\nOpening Netica:')
            vprint(1,self.verboselvl,self.mesg.value.decode('utf-8'))
        else:
            raise(NeticaInitFail(res.value))    
        vprint(1,self.verboselvl,'Netica is open\n' + '#'*40 + '\n' * 2)
        
    def CloseNetica(self):
        self.n.CloseNetica_bn.argtypes = [ct.c_void_p,ct.c_char_p]
        self.n.CloseNetica_bn.restype = ct.c_int
        res = self.n.CloseNetica_bn(self.env, self.mesg)    
        if res >= 0:
            vprint(1,self.verboselvl,"Closing Netica:")
            vprint(1,self.verboselvl,self.mesg.value.decode('utf-8'))
        else:
            raise(NeticaCloseFail(res.value))    
        self.n = None
        
    def GetError(self, severity = pnC.errseverity_ns_const.ERROR_ERR, after = None):
        self.n.GetError_ns.argtypes = [ct.c_void_p,ct.c_int,ct.c_void_p]
        self.n.GetError_ns.restype = ct.c_void_p
        res = self.n.GetError_ns(self.env, severity, after)
        if res: return res
        else:   return None

    def ErrorMessage(self, error):
        self.n.ErrorMessage_ns.argtypes = [ct.c_void_p]
        # self.n.ErrorMessage_ns.restype = ct.c_char_p
        return self.n.ErrorMessage_ns(error)

    # general error-checking function    
    def chkerr(self,err_severity = pnC.errseverity_ns_const.ERROR_ERR):
        if self.GetError(err_severity):
            exceptionMsg = ("\npythonNeticaUtils: \nError " +  cth.c_char_p2str(self.ErrorMessage(self.GetError(err_severity))))
            self.CloseNetica()
            raise NeticaException(exceptionMsg)

    ################################################################
    # Small definitions and little functions in alphabetical order #  
    ################################################################
    
    # NOTE!! These functions take as inputs and give as ouptuts as c
    # pointers (except where netica outputs are not pointers). Outputs
    # should be converted back into python using cthelper functions.
    
    def AddLink(self,pnode,chnode):
        self.n.AddLink_bn.argtypes = [ct.c_void_p,ct.c_void_p]
        self.n.AddLink_bn.restype = ct.c_int
        self.n.AddLink_bn(pnode,chnode)
        self.chkerr()
    
    def AddFileToCaseset(self,caseset,streamer,degree):
        self.n.AddFileToCaseset_cs.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_double,ct.c_char_p]
        self.n.AddFileToCaseset_cs.restype = ct.c_void_p
        self.n.AddFileToCaseset_cs(caseset,streamer,ct.c_double(degree),None)
        self.chkerr()

    def CompileNet(self, net):
        self.n.CompileNet_bn.argtypes = [ct.c_void_p]
        self.n.CompileNet_bn.restype = None
        self.n.CompileNet_bn(net)
        self.chkerr()

    def CopyNet(self,oldnet, newnetname,options):
        ### net_bn* CopyNet_bn ( const net_bn*  net,   const char*  new_name,   environ_ns*  new_env,   const char*  options )
        self.n.CopyNet_bn.argtypes = [ct.c_void_p,ct.c_char_p,ct.c_void_p,ct.c_char_p]
        self.n.CopyNet_bn.restype = ct.c_void_p
        newnet = self.n.CopyNet_bn(oldnet,newnetname,self.env,options)
        self.chkerr()
        return newnet

    def CopyNodes(self,oldnodes,newnet,options):
        ### nodelist_bn* CopyNodes_bn ( const nodelist_bn*  nodes,   net_bn*  new_net,   const char*  options )
        self.n.CopyNodes_bn.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_char_p]
        self.n.CopyNodes_bn.restype = ct.c_void_p
        newnodes = self.n.CopyNodes_bn(oldnodes,newnet,options)
        self.chkerr()
        return newnodes 

    def DeleteCaseset(self,caseset):
        ### void DeleteCaseset_cs ( caseset_cs*  cases )
        self.n.DeleteCaseset_cs.argtypes = [ct.c_void_p]
        self.n.DeleteCaseset_cs.restype = None
        self.n.DeleteCaseset_cs(caseset)
        self.chkerr()

    def DeleteLearner(self,newlearner):
        ### void DeleteLearner_bn ( learner_bn*  learner )
        self.n.DeleteLearner_bn.argtypes = [ct.c_void_p]
        self.n.DeleteLearner_bn.restype = None
        self.n.DeleteLearner_bn(newlearner)
        self.chkerr()

    def DeleteNet(self,cnet):
        ### void DeleteNet_bn ( net_bn*  net )
        self.n.DeleteNet_bn.argtypes = [ct.c_void_p]
        self.n.DeleteNet_bn.restype = None
        self.n.DeleteNet_bn(cnet)
        self.chkerr()
    
    def DeleteNetTester(self,ctester):
        ### void DeleteNetTester_bn ( tester_bn*  test )
        self.n.DeleteNetTester_bn.argtypes = [ct.c_void_p]
        self.n.DeleteNetTester_bn.restype = None
        self.n.DeleteNetTester_bn(ctester)
        self.chkerr()

    def DeleteNode(self,cnode):
        ### void DeleteNode_bn ( node_bn*  node )
        self.n.DeleteNode_bn.argtypes = [ct.c_void_p]
        self.n.DeleteNode_bn.restype = None #void is returned
        self.n.DeleteNode_bn(cnode)
        self.chkerr()
        
    def DeleteNodeTables(self,node):
        ### void DeleteNodeTables_bn ( node_bn*  node )
        self.n.DeleteNodeTables_bn.argtypes = [ct.c_void_p]
        self.n.DeleteNodeTables_bn.restype = None
        self.n.DeleteNodeTables_bn(node)
        self.chkerr()

    def DeleteNodeList(self,cnodes):
        ### void DeleteNodeList_bn ( nodelist_bn*  nodes )
        self.n.DeleteNodeList_bn.argtypes = [ct.c_void_p]
        self.n.DeleteNodeList_bn.restype = None
        self.n.DeleteNodeList_bn(cnodes)
        self.chkerr()

    def DeleteStream(self,cstream):
        ### void DeleteStream_ns ( stream_ns*  file )
        self.n.DeleteStream_ns.argtypes = [ct.c_void_p]
        self.n.DeleteStream_ns.restype = None
        self.n.DeleteStream_ns(cstream)
        self.chkerr()

    def DeleteSensvToFinding(self,sens):
        ### void DeleteSensvToFinding_bn ( sensv_bn*  sens )
        self.n.DeleteSensvToFinding_bn.argtypes = [ct.c_void_p]
        self.n.DeleteSensvToFinding_bn.restype = None
        self.n.DeleteSensvToFinding_bn(sens)
        self.chkerr()

    def EnterFinding(self,cnode,cval):
        ### void EnterFinding_bn ( node_bn*  node,   state_bn  state )
        self.n.EnterFinding_bn.argtypes = [ct.c_void_p, ct.c_int]
        self.n.EnterFinding_bn.restype = None
        self.n.EnterFinding_bn(cnode,cval)
        self.chkerr()

    def EnterNodeValue(self,cnode,cval):
        ### void EnterNodeValue_bn ( node_bn*  node,   double  value )
        self.n.EnterNodeValue_bn.argtypes = [ct.c_void_p, ct.c_double]
        self.n.EnterNodeValue_bn.restype = None
        self.n.EnterNodeValue_bn(cnode,cval)
        self.chkerr()
        
    def GetMutualInfo(self,sensentrop,Vnode):
        ### double GetMutualInfo_bn ( sensv_bn*  sens,   const node_bn*  Vnode )
        self.n.GetMutualInfo_bn.argtypes = [ct.c_void_p, ct.c_void_p]
        self.n.GetMutualInfo_bn.restype = ct.c_double
        retvar = self.n.GetMutualInfo_bn(sensentrop,Vnode)
        self.chkerr()
        return retvar        

    def GetNetNodes(self,cnet):
        ### const nodelist_bn* GetNetNodes_bn ( const net_bn*  net )
        self.n.GetNetNodes2_bn.argtypes = [ct.c_void_p, ct.c_char_p]
        self.n.GetNetNodes2_bn.restype = ct.c_void_p
        allnodes = self.n.GetNetNodes2_bn(cnet,None)
        self.chkerr()
        return allnodes

    def GetNodeBeliefs(self,cnode):
        ### const prob_bn* GetNodeBeliefs_bn ( node_bn*  node )
        self.n.GetNodeBeliefs_bn.argtypes = [ct.c_void_p]        
        # self.n.GetNodeBeliefs_bn.restype = ct.c_float_p
        beliefs = self.n.GetNodeBeliefs_bn(cnode)
        self.chkerr()
        return beliefs

    def GetNodeExpectedValue(self,cnode):
        ### double GetNodeExpectedValue_bn ( node_bn*  node,   double*  std_dev,   double*  x3,   double*  x4 )
        self.n.GetNodeExpectedValue_bn.argtypes = [ct.c_void_p, ct.c_double, ct.c_double, ct.c_double]
        self.n.GetNodeExpectedValue_bn.restype = ct.c_double

        # allocate pointer
        std_dev = ct.c_double()
        
        expected_val = self.n.GetNodeExpectedValue_bn(cnode,ct.byref(std_dev),
                                                      None,None)
        self.chkerr()
        return expected_val, std_dev

    def GetNodeExperience(self,cnode,parent_states):
        ### double GetNodeExperience_bn ( const node_bn*  node,   const state_bn*  parent_states )
        self.n.GetNodeExperience_bn.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        self.n.GetNodeExperience_bn.restype = ct.c_double
        experience = self.n.GetNodeExperience_bn(cnode,parent_states)
        self.chkerr()
        return experience
    
    def GetNodeFinding(self,cnode):
        ### state_bn GetNodeFinding_bn ( const node_bn*  node )
        self.n.GetNodeFinding_bn.argtypes = [ct.c_void_p]
        self.n.GetNodeFinding_bn.restype = ct.c_int
        cf = self.n.GetNodeFinding_bn(cnode)
        self.chkerr()
        return cf
    
    def GetNodeLevels(self,cnode):
        ### const level_bn* GetNodeLevels_bn ( const node_bn*  node )
        self.n.GetNodeLevels_bn.argtypes = [ct.c_void_p]
        # self.n.GetNodeLevels_bn.restype = ct.c_void_p
        nodelevels = self.n.GetNodeLevels_bn(cnode)
        self.chkerr()
        return nodelevels

    def GetNodeLikelihood(self,cnode):
        ### const prob_bn* GetNodeLikelihood_bn ( const node_bn*  node )
        self.n.GetNodeLikelihood_bn.argtypes = [ct.c_void_p]
        # self.n.GetNodeLikelihood_bn.restype = ct.c_void_p
        nodelikelihood = self.n.GetNodeLikelihood_bn(cnode)
        self.chkerr()
        return nodelikelihood

    def GetNodeName(self,cnode):
        ### const char* GetNodeName_bn ( const node_bn*  node )
        self.n.GetNodeName_bn.argtypes = [ct.c_void_p]        
        # self.n.GetNodeName_bn.restype = ct.c_char_p
        cname = self.n.GetNodeName_bn(cnode)
        self.chkerr()
        return cname

    def GetNodeNamed(self,nodename,cnet):
        ### node_bn* GetNodeNamed_bn ( const char*  name,   const net_bn*  net )
        self.n.GetNodeNamed_bn.argtypes = [ct.c_char_p, ct.c_void_p]
        self.n.GetNodeNamed_bn.restype = ct.c_void_p
        retnode = self.n.GetNodeNamed_bn(nodename,cnet)
        self.chkerr()
        return(retnode)
    
    def GetNodeNumberStates(self,cnode):
        ### int GetNodeNumberStates_bn ( const node_bn*  node )
        self.n.GetNodeNumberStates_bn.argtypes = [ct.c_void_p]
        self.n.GetNodeNumberStates_bn.restype = ct.c_int
        numstates = self.n.GetNodeNumberStates_bn(cnode)
        self.chkerr()
        return numstates

    def GetNodeParents(self,cnode):
        ### const nodelist_bn* GetNodeParents_bn ( const node_bn*  node )
        self.n.GetNodeParents_bn.argtypes = [ct.c_void_p]
        self.n.GetNodeParents_bn.restype = ct.c_void_p
        parents = self.n.GetNodeParents_bn(cnode)
        self.chkerr()
        return parents
    
    def GetNodeStateName(self,cnode,cstate):
        ### const char* GetNodeStateName_bn ( const node_bn*  node,   state_bn  state )
        self.n.GetNodeStateName_bn.argtypes = [ct.c_void_p, ct.c_int]
        #self.n.GetNodeStateName_bn.restype = ct.c_char_p
        stname = self.n.GetNodeStateName_bn(cnode,cstate)
        self.chkerr()
        return stname

    def GetNodeTitle(self,cnode):
        ### const char* GetNodeTitle_bn ( const node_bn*  node )
        self.n.GetNodeTitle_bn.argtypes = [ct.c_void_p]
        # self.n.GetNodeTitle_bn.restype = ct.c_char_p
        ctitle = self.n.GetNodeTitle_bn(cnode)
        self.chkerr()
        return ctitle

    def GetTestLogLoss(self,ctester,cnode):
        ### double GetTestLogLoss_bn ( tester_bn*  test,   node_bn*  node )
        self.n.GetTestLogLoss_bn.argtypes = [ct.c_void_p, ct.c_void_p]
        self.n.GetTestLogLoss_bn.restype = ct.c_double
        logloss = self.n.GetTestLogLoss_bn(ctester,cnode)
        self.chkerr()
        return logloss
    
    def GetTestConfusion(self,ctester,cnode,predState,actualState):
        ### double GetTestConfusion_bn ( tester_bn*  test,   node_bn*  node,   int  predictedState,   int  actualState )
        self.n.GetTestConfusion_bn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
        self.n.GetTestConfusion_bn.restype = ct.c_double
        confusion = self.n.GetTestConfusion_bn(ctester,cnode,predState,actualState)
        self.chkerr()
        return confusion
    
    def GetTestErrorRate(self,ctester,cnode):
        ### double GetTestErrorRate_bn ( tester_bn*  test,   node_bn*  node )
        self.n.GetTestErrorRate_bn.argtypes = [ct.c_void_p, ct.c_void_p]
        self.n.GetTestErrorRate_bn.restype = ct.c_double
        errrate = self.n.GetTestErrorRate_bn(ctester,cnode)
        self.chkerr()
        return errrate
    
    def GetTestQuadraticLoss(self,ctester,cnode):
        ### double GetTestQuadraticLoss_bn ( tester_bn*  test,   node_bn*  node )
        self.n.GetTestQuadraticLoss_bn.argtypes = [ct.c_void_p, ct.c_void_p]
        self.n.GetTestQuadraticLoss_bn.restype = ct.c_double
        quadloss = self.n.GetTestQuadraticLoss_bn(ctester,cnode)
        self.chkerr()
        return quadloss
        
    def GetVarianceOfReal(self,sensv,Vnode):
        ### double GetVarianceOfReal_bn ( sensv_bn*  sens,   const node_bn*  Vnode )
        self.n.GetVarianceOfReal_bn.argtypes = [ct.c_void_p, ct.c_void_p]
        self.n.GetVarianceOfReal_bn.restype = ct.c_double
        retvar = self.n.GetVarianceOfReal_bn(sensv,Vnode)
        self.chkerr()
        return retvar
        
    def LearnCPTs(self,learner,nodes,caseset,voodooPar):
        ### void LearnCPTs_bn ( learner_bn*  learner, const nodelist_bn*  nodes, const caseset_cs*  cases, double  degree )
        self.n.LearnCPTs_bn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_double]
        self.n.LearnCPTs_bn.restype = None
        self.n.LearnCPTs_bn(learner,nodes,caseset,voodooPar)
        self.chkerr()

    def LengthNodeList(self, nodelist):
        ### int LengthNodeList_bn ( const nodelist_bn*  nodes )
        self.n.LengthNodeList_bn.argtypes = [ct.c_void_p]
        self.n.LengthNodeList_bn.restype = ct.c_int
        res = self.n.LengthNodeList_bn(nodelist)
        self.chkerr()
        return res

    def LimitMemoryUsage(self, memlimit):
        ### double LimitMemoryUsage_ns ( double  max_mem,   environ_ns*  env )
        self.n.LengthNodeList_bn.argtypes = [ct.c_double, ct.c_void_p]
        self.n.LengthNodeList_bn.restype = ct.c_double
        self.n.LimitMemoryUsage_ns(memlimit, self.env)
        vprint(1,self.verboselvl,'set memory limit to ---> %f bytes' %memlimit)
        self.chkerr()
        
    def NewCaseset(self,name):
        ### caseset_cs* NewCaseset_cs ( const char*  name,   environ_ns*  env )
        self.n.NewCaseset_cs.argtypes = [ct.c_char_p, ct.c_void_p]
        self.n.NewCaseset_cs.restype = ct.c_void_p
        newcaseset = self.n.NewCaseset_cs(name,self.env)
        self.chkerr()
        return newcaseset

    def NewFileStreamer(self,infile):
        ### stream_ns* NewFileStream_ns ( const char*  filename,   environ_ns*  env,   const char*  access )
        self.n.NewFileStream_ns.argtypes = [ct.c_char_p, ct.c_void_p, ct.c_char_p]
        self.n.NewFileStream_ns.restype = ct.c_void_p
        streamer =  self.n.NewFileStream_ns(infile, self.env, None)
        self.chkerr()
        return streamer

    def NewLearner(self,method):
        ### learner_bn* NewLearner_bn ( learn_method_bn  method,   const char*  options,   environ_ns*  env )
        self.n.NewLearner_bn.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_void_p]
        self.n.NewLearner_bn.restype = ct.c_void_p
        newlearner = self.n.NewLearner_bn(method,None,self.env)
        self.chkerr()
        return newlearner

    def NewNet(self, netname):
        ### net_bn* NewNet_bn ( const char*  name,   environ_ns*  env )
        self.n.NewNet_bn.argtypes = [ct.c_char_p,ct.c_void_p]
        self.n.NewNet_bn.restype = ct.c_void_p
        newnet = self.n.NewNet_bn(netname,self.env)
        self.chkerr()
        return newnet
    
    def NewNetTester(self,test_nodes,unobs_nodes):
        ### tester_bn* NewNetTester_bn ( nodelist_bn*  test_nodes,   nodelist_bn*  unobsv_nodes,   int  tests )
        self.n.NewNetTester_bn.argtypes = [ct.c_void_p,ct.c_void_p, ct.c_int]
        self.n.NewNetTester_bn.restype = ct.c_void_p
        tester = self.n.NewNetTester_bn(test_nodes,unobs_nodes,ct.c_int(-1))
        self.chkerr()
        return tester

    def NewNodeList2(self,length,cnet):
        ### nodelist_bn* NewNodeList2_bn ( int  length,   const net_bn*  net )
        self.n.NewNodeList2_bn.argtypes = [ct.c_int,ct.c_void_p]
        self.n.NewNodeList2_bn.restype = ct.c_void_p
        nodelist = self.n.NewNodeList2_bn(length,cnet)
        self.chkerr()
        return nodelist
    
    def NewSensvToFinding(self,Qnode,Vnodes,what_find):
        ### sensv_bn* NewSensvToFinding_bn ( const node_bn*  Qnode,   const nodelist_bn*  Vnodes,   int  what_find )
        self.n.NewSensvToFinding_bn.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int]
        self.n.NewSensvToFinding_bn.restype = ct.c_void_p
        sensv = self.n.NewSensvToFinding_bn(Qnode,Vnodes,what_find)
        self.chkerr()
        return sensv

    def NthNode(self,nodelist,index_n):
        ### node_bn* NthNode_bn ( const nodelist_bn*  nodes,   int  index )
        self.n.NthNode_bn.argtypes = [ct.c_void_p,ct.c_int]
        self.n.NthNode_bn.restype = ct.c_void_p
        cnode = self.n.NthNode_bn(nodelist,index_n)
        self.chkerr()
        return cnode
        
    def ReadNet(self,streamer):
        ### net_bn ReadNet_bn ( stream_ns*  file,   int  options )
        self.n.ReadNet_bn.argtypes = [ct.c_void_p, ct.c_int]
        self.n.ReadNet_bn.restype = ct.c_void_p
        cnet = self.n.ReadNet_bn(streamer,ct.c_int(pnC.netica_const.NO_WINDOW))
        # check for errors
        self.chkerr()
        # reset the findings
        self.n.RetractNetFindings_bn(cnet)
        self.chkerr()
        return cnet                                   
                                   
    def RetractNetFindings(self,cnet):
        ### void RetractNetFindings_bn ( net_bn*  net )
        self.n.RetractNetFindings_bn.argtypes = [ct.c_void_p]
        self.n.RetractNetFindings_bn.restype = None
        self.n.RetractNetFindings_bn(cnet)
        self.chkerr()

    def ReviseCPTsByCaseFile(self,casStreamer,cnodes,voodooPar):
        ### void ReviseCPTsByCaseFile_bn ( stream_ns*  file, const nodelist_bn*  nodes, int  updating, double  degree )
        self.n.ReviseCPTsByCaseFile_bn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_double]
        self.n.ReviseCPTsByCaseFile_bn.restype = None
        self.n.ReviseCPTsByCaseFile_bn(casStreamer,cnodes,ct.c_int(0),voodooPar)
        self.chkerr()

    def SetLearnerMaxIters(self,learner,maxiters):
        ### int SetLearnerMaxIters_bn ( learner_bn*  learner,   int  max_iters )
        self.n.SetLearnerMaxIters_bn.argtypes = [ct.c_void_p, ct.c_int]
        self.n.SetLearnerMaxIters_bn.restype = ct.c_int
        self.n.SetLearnerMaxIters_bn(learner,maxiters)
        self.chkerr()    

    def SetLearnerMaxTol(self,learner,tol):
        ### double SetLearnerMaxTol_bn ( learner_bn*  learner,   double  log_likeli_tol )
        self.n.SetLearnerMaxTol_bn.argtypes = [ct.c_void_p, ct.c_double]
        self.n.SetLearnerMaxTol_bn.restype = ct.c_double
        self.n.SetLearnerMaxTol_bn(learner,tol)
        self.chkerr()         
        
    def SetNetAutoUpdate(self,cnet,belief_value):
        ### int SetNetAutoUpdate_bn (	net_bn*  net,   int  autoupdate )
        self.n.SetNetAutoUpdate_bn.argtypes = [ct.c_void_p, ct.c_int]
        self.n.SetNetAutoUpdate_bn.restype = ct.c_int
        self.n.SetNetAutoUpdate_bn(cnet,belief_value)
        self.chkerr()

    def SetNthNode(self, nodelist, position, cnode):
        ### void SetNthNode_bn ( nodelist_bn*  nodes,   int  index,   node_bn*  node )
        self.n.SetNthNode_bn.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
        self.n.SetNthNode_bn.restype = None
        self.n.SetNthNode_bn(nodelist, position, cnode)
        self.chkerr()

    def SetNodeLevels(self, cnode, clevels):
        ### void SetNodeLevels_bn ( node_bn*  node,   int  num_states,   const level_bn*  levels )
        self.n.SetNodeLevels_bn.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
        self.n.SetNodeLevels_bn.restype = None
        self.n.SetNodeLevels_bn(cnode, ct.c_int(len(clevels)-1), clevels.ctypes.data_as(ct.POINTER(ct.c_double)))
        self.chkerr()

    def TestWithCaseset(self, test, cases):
        ### void TestWithCaseset_bn ( tester_bn*  test,   caseset_cs*  cases )
        self.n.TestWithCaseset_bn.argtypes = [ct.c_void_p, ct.c_void_p]
        self.n.TestWithCaseset_bn.restype = None
        self.n.TestWithCaseset_bn(test, cases)
        self.chkerr()
        
    def WriteNet(self, cnet, filename_streamer):
        ### void WriteNet_bn ( const net_bn*  net,   stream_ns*  file )
        self.n.WriteNet_bn.argtypes = [ct.c_void_p, ct.c_void_p]
        self.n.WriteNet_bn.restype = None
        self.n.WriteNet_bn(cnet, filename_streamer)
        self.chkerr()

    ###################################
    # Other functions for Netica      #   
    ###################################

    def cartesian(self,arrays,out=None):   
        '''
        function to calculate the Cartesian sum of multiple arrays.
        This is used to provide the permutations (odometer style) of all
        the possible parent states when calculating experience.
        See: http://stackoverflow.com/questions/1208118/
        using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        '''

        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype
    
        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)
        m = int(n / arrays[0].size) # lets make sure its an int for python 3!
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            # recursive?
            self.cartesian(arrays[1:], out=out[0:m,1:])
            for j in np.arange(1, arrays[0].size):
                out[j*m:(j+1)*m,1:] = out[0:m,1:]
        return out
        
#################
# Error Classes #
#################
# -- can't open external file
class dllFail(Exception):
    def __init__(self,cplat):
        self.cplat = cplat
    def __str__(self):
        if "windows" in self.cplat.lower():
            return("\n\nCannot open Netica.dll.\nBe sure it's in the path")
        else:
            return("\n\nCannot open libnetica.so.\nBe sure it's in the path")
# -- can't initialize Netica
class NeticaInitFail(Exception):
    def __init__(self,msg):
        self.msg = msg
    def __str__(self):
        return("\n\nCannot initialize Netica. Netica message is:\n%s\n" 
               %(self.msg))
# -- can't close Netica
class NeticaCloseFail(Exception):
    def __init__(self,msg):
        self.msg = msg
    def __str__(self):
        return("\n\nCannot properly close Netica. Netica message is:\n%s\n" 
               %(self.msg))
# -- General Netica Exception
class NeticaException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
