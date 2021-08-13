# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:28:54 2020

@author: guo_h
"""

import datetime as dt
import pandas as pd 
import numpy as np
import time
from maxent import irl as irl_ME 
from deep_maxent import irl as irl_DME

if __name__ == '__main__':
    
    # iterational paras
    test_date = '0813'      # a time-flag for the result files
    DUID_list = ['STAN-1']    # a list for DUID to be analyzed, such as 'BW03','STAN-1', 'GSTONE4', 'DDPS1'
    test_time_list = ['T1']     # test flag, only for notes in filename

    # IRL parameter
    Para_trajLen = 14       # trajectory length
    epochs = 3000  # start at 5
    discount = 0.5
    learning_rate = 0.001  #

    # fundamental paras
    Para_startDate = '20171019'     # StartDate
    Para_endDate = '20180919'  # endDate
    Num_T = 48    # number of time periods in one day
    Para_disDemand = 5
    Para_disPrice = 10
    Para_disQuan = 10
    Para_disRevenue = 10
    Num_A = 30
    LR_damp_rate = 1
    LR_damp_period = 20
    irl_para = 'DME'    #'DME'(Deep MaxEnt) or 'ME'(MaxEnt)
    
    # Deep MaxEnt Parameters
    DME_structure = (2, 10, 4)  # node numbers in each layer
    DME_l1 = 0
    DME_l2 = 0
    DME_initialisation = "normal"


    for r in range(len(DUID_list)):
        #  input parameter
        DUID = DUID_list[r]
        test_time = test_time_list[r]

        # start time
        starttime = str(time.strftime("%Y%m%d %X", time.localtime()) )
        print(starttime+' '+test_date+test_time+' '+'DUID: '+str(DUID)+' starts!')

        # fundamental parameter
        File_dataInPath = 'test_data/MDP_information/'
        Para_startDateDT = dt.datetime.strptime(Para_startDate, "%Y%m%d")
        Para_endDateDT = dt.datetime.strptime(Para_endDate, "%Y%m%d")
        Num_D = Para_endDateDT.__sub__(Para_startDateDT).days + 1
        Para_trajWindow = int(Num_D/Para_trajLen)
        Para_trajNum = int(Para_trajWindow*Num_T) 
        test_para = test_date + test_time + '_' + irl_para
        file_outPath = 'test_data/reward_function/'
        file_name = file_outPath + test_para + '_' + DUID
        
        # read MDP inforamtion
        dataInPrefix = DUID + '_' + Para_startDate + '_' + Para_endDate + '_D' + str(Para_disDemand) + '_P' + str(
            Para_disPrice) + '_Q' + str(Para_disQuan) + '_A' + str(Num_A)
            
        feature_raw = pd.read_csv(File_dataInPath+dataInPrefix+'_feature.csv')
        state_raw = pd.read_csv(File_dataInPath+dataInPrefix+'_state.csv')
        action_raw = pd.read_csv(File_dataInPath+dataInPrefix+'_action.csv')
        
        # feature_matrix generate
        feature_matrix0 = np.array(feature_raw)
        feature_matrix1 = np.delete(feature_matrix0, [0, 1], axis=1)
        feature_matrix_max = np.max(feature_matrix1, axis=0)
        
        feature_matrix_mean = np.mean(feature_matrix1, axis=0)
        feature_matrix = (feature_matrix1-feature_matrix_mean)/feature_matrix_mean
        Num_S = len(feature_matrix1)
        
        # n_actions
        n_actions = Num_A + 1
        
        # trajectory
        line = 0
        trajectory_matrix = np.zeros([Para_trajNum, Para_trajLen, 2], dtype=int)
        for w in range(Para_trajWindow): 
            for l in range(Para_trajLen):
                for t in range(Num_T):
                    tr = w*Num_T + t
                    # build trajectory_matirx
                    trajectory_matrix[tr, l, 0] = state_raw.iloc[line, 2]-1
                    trajectory_matrix[tr, l, 1] = action_raw.iloc[line, 2]-1
                    line = line + 1                
        
        # transition_probabiltiy
        transition_raw = np.zeros([Num_S, n_actions, Num_S], dtype=float)
        for tr in range(Para_trajNum):
            for l in range(Para_trajLen-1):
                s0 = trajectory_matrix[tr, l, 0]
                a0 = trajectory_matrix[tr, l, 1]
                s1 = trajectory_matrix[tr, l+1, 0]
                transition_raw[s0, a0, s1] = transition_raw[s0, a0, s1] + 1
        
        transition_sum = transition_raw.sum(axis=2)
        transition_matrix = np.zeros([Num_S, n_actions, Num_S], dtype=float)
        for s0 in range(Num_S):
            for a0 in range(n_actions):
                if transition_sum[s0, a0] > 0 :
                    for s1 in range(Num_S):
                        transition_matrix[s0, a0, s1] = transition_raw[s0, a0, s1]/transition_sum[s0, a0]
        
        # parameter write
        fo_para = open(file_name+'_parameter.csv', "w")
        fo_para.write('DUID, '+DUID+'\n')
        fo_para.write('test number, '+test_para+'\n')
        fo_para.write('if Deep, '+irl_para+'\n')
        fo_para.write('disDemand, '+str(Para_disDemand)+'\n')
        fo_para.write('disPrice, '+str(Para_disPrice)+'\n')
        fo_para.write('disQuantity, '+str(Para_disQuan)+'\n')
        fo_para.write('disRevenue, '+str(Para_disRevenue)+'\n')
        fo_para.write('number of actions, '+str(Num_A)+'\n')
        fo_para.write('trajectory length, '+str(Para_trajLen)+'\n')
        fo_para.write('IRL_discount, '+str(discount)+'\n')
        fo_para.write('IRL_epochs, '+str(epochs)+'\n')
        fo_para.write('IRL_learning_rate, '+str(learning_rate)+'\n')
        fo_para.write('start_date, '+Para_startDate+'\n')   
        fo_para.write('end_date, '+Para_endDate+'\n') 
        fo_para.write('feature_max, '+str(feature_matrix_max[0])+', '+str(feature_matrix_max[1])+'\n')
        fo_para.write('feature_matrix_mean, '+str(feature_matrix_mean[0])+', '+str(feature_matrix_mean[1])+'\n')
        fo_para.write('learning rate damping rate, '+str(LR_damp_rate)+'\n') 
        fo_para.write('DME_structure, '+str(DME_structure)+'\n') 
        fo_para.write('DME_l1, '+str(DME_l1)+'\n') 
        fo_para.write('DME_l2, '+str(DME_l2)+'\n') 
        fo_para.write('DME_initialisation, '+DME_initialisation+'\n') 
        fo_para.write('start, '+starttime+'\n')
        fo_para.close() 
        
        
        # irl calculation
        if irl_para == 'ME':
            reward_vector, alpha, alpha_record, grad_record = irl_ME(feature_matrix, n_actions, discount, transition_matrix, trajectory_matrix,
                                                                    epochs, learning_rate, file_name,test_para, LR_damp_rate, LR_damp_period)
        
        if irl_para == 'DME':
            reward_vector, weights, biases = irl_DME(DME_structure, feature_matrix, n_actions, discount, transition_matrix,
                                                    trajectory_matrix, epochs, learning_rate, file_name,
                                                    DME_initialisation, DME_l1, DME_l2)

        
        # end time
        endtime = str(time.strftime("%Y%m%d %X", time.localtime()) )
        
        fo_para = open(file_name+'_parameter.csv',"a")
        fo_para.write('end, '+endtime+'\n')
        fo_para.close() 
        
        print(endtime+' '+'DUID: '+str(DUID)+' ends!')
