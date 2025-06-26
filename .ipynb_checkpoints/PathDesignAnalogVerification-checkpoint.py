import numpy as np
import sympy
from sympy import symbols, Eq, linear_eq_to_matrix
import random
import matplotlib.pyplot as plt
from collections import deque, defaultdict
# from pyeda.inter import expr, exprvar, expr2bdd
import networkx as nx
from matplotlib.colors import ListedColormap
import itertools
import pandas as pd
from sympy.logic.boolalg import SOPform
import re
import pickle
import datetime
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from itertools import product
import seaborn as sns
from collections import deque
import uuid
import copy
import math

class Bus:
    def __init__(self, MainBDD_For_GoldenModel, Topological_order):
        self.crossbar_designs = {}

        self.MainBDD_For_GoldenModel = MainBDD_For_GoldenModel
        self.Topological_order = Topological_order

        self.WorstCaseMaxZero = {}
        self.WorstCaseMinOne = {}
        
        self.OnesCurrentFromDesignMap = {}
        self.ZerosCurrentFromDesignMap = {}
        
        self.input_vars = self.getInputVaribles()

        self.Output = self.getOutputMap()

    def getNumberOfDesigns(self):
        return len(self.crossbar_designs)

    def getNumberOfCrossbars(self):
        return len({self.crossbar_designs[design_ID]['crossbar_design_instance'] for design_ID in self.crossbar_designs})
        
    def getOutputMap(self):
        nodesLiterals = [self.MainBDD_For_GoldenModel.nodes[node]['ExpressionRoot'] 
                         for node in self.MainBDD_For_GoldenModel.nodes 
                         if self.MainBDD_For_GoldenModel.nodes[node]['BipartitePart'] == 'U1' 
                         and self.MainBDD_For_GoldenModel.nodes[node]['ExpressionRoot'] is not None]
        
        return {nodesLiteral:0 for nodesLiteral in nodesLiterals}

    def getInputVaribles(self):
        nodesLiterals = [self.MainBDD_For_GoldenModel.nodes[node]['literal'] for node in self.MainBDD_For_GoldenModel.nodes if self.MainBDD_For_GoldenModel.nodes[node]['BipartitePart'] == 'U2']
        input_vars = set()
        for label in nodesLiterals:
            var = label.replace('~', '')
            if 'O' not in var:
                input_vars.add(var)
        input_vars = list(input_vars)
        return input_vars
        
    def connect(self, crossbar_design_instance, design_ID, DesignIdItemToWordLineInputMap, OutputLine_group_Map, processed_group_graph):
        self.crossbar_designs[design_ID] = {
            "crossbar_design_instance":crossbar_design_instance, 
            "DesignIdItemToWordLineInputMap":DesignIdItemToWordLineInputMap, 
            "OutputLine_group_Map":OutputLine_group_Map,
            "processed_group_graph":processed_group_graph
        }
        self.OnesCurrentFromDesignMap[design_ID] = []
        self.ZerosCurrentFromDesignMap[design_ID] = []

    def GoldenModel(self, input_assignment, graph=None):

        singleDesignCheck = True
        graph = self.MainBDD_For_GoldenModel
    
        # Helper: decide if a U2 node can be passed
        def is_passable_u2(node):
            literal = graph.nodes[node].get('literal', '')
            if literal.startswith('~') or literal.startswith('-'):
                var = literal[1:]
                return input_assignment.get(var) == 0
            else:
                return input_assignment.get(literal) == 1
    
        # Start from all nodes with in-degree 0
        start_nodes = [n 
                       for n in graph.nodes 
                       if graph.in_degree(n) == 0
                      ]
        visited_outputs = {graph.nodes[n].get('ExpressionRoot'):0 
                           for n in graph.nodes 
                           if graph.nodes[n].get('ExpressionRoot') is not None}

        # and graph.out_degree(n)==0 need to check this
        # print('len(visited_outputs)', len(visited_outputs))
        
        output_paths = {}  # NEW: tracks the path that triggers each output
    
        for start_node in start_nodes:
            stack = [(start_node, [start_node])]
    
            while stack:
                current_node, path = stack.pop()
                node_data = graph.nodes[current_node]
                # bipartite_type = node_data.get('BipartitePart')
                # literal = node_data.get('literal', '')
                expression_root = node_data.get('ExpressionRoot')
    
                if expression_root is not None:
                    visited_outputs[expression_root] = 1
                    output_paths[expression_root] = path
                    # continue
    
                for succ in graph.successors(current_node):
                    succ_data = graph.nodes[succ]
                    succ_type = succ_data.get('BipartitePart')
    
                    # U1 nodes are always passable
                    if succ_type == 'U1':
                        stack.append((succ, path + [succ]))
    
                    # U2 nodes need validation
                    elif succ_type == 'U2' and is_passable_u2(succ):
                        stack.append((succ, path + [succ]))
    
        # print("Computed Outputs:", visited_outputs)
        return visited_outputs, output_paths

    def CalibrateHRSValues(self, Optimisation=True):
        if(not Optimisation):
            all_combinations = list(itertools.product([0, 1], repeat=len(self.input_vars)))
            num_tests = len(all_combinations)
            input_assignments = [
                dict(zip(self.input_vars, combo)) for combo in all_combinations
            ]

            for i, input_assignment in enumerate(input_assignments):
                # Run model and golden model        
                golden_output = self.GoldenModel(input_assignment, self.MainBDD_For_GoldenModel)
    
                model_output = self.ExecuteDesignsInTopologicalOrder(input_assignment)

                #remove unwanted keys that are not part of golden outputs
                model_processed_output = {k: v for k, v in model_output.items() if k in golden_output}
                
                # Compare
                match = (model_processed_output == golden_output)

                # if(match):
            return

    def CalibrateHRSValues(self, Optimisation=True, max_iterations=6):
        design_count = 0
        for design_ID_group in self.crossbar_designs:
            crossbar_design_instance       = self.crossbar_designs[design_ID_group]['crossbar_design_instance']
            OutputLine_group_Map           = self.crossbar_designs[design_ID_group]['OutputLine_group_Map']
            DesignIdItemToWordLineInputMap = self.crossbar_designs[design_ID_group]["DesignIdItemToWordLineInputMap"]

            wordLineInputs = list(DesignIdItemToWordLineInputMap.values())
            
            maxZeroCurrent, minOneCurrent = crossbar_design_instance.FindOptimalHRSvalue(design_ID_group, OutputLine_group_Map, wordLineInputs, max_iterations)

            self.WorstCaseMaxZero[design_ID_group] = maxZeroCurrent
            self.WorstCaseMinOne[design_ID_group] = minOneCurrent

            print(f'{design_count} design is complete')
            design_count+=1
    
    def RunRandomTestCases(self, num_tests=10, RunAllTests=False):
        
        test_results = []

        n = len(self.input_vars)
        total_combinations = 2 ** n

        # Step 2: Generate all combinations if RunAllTests is True
        if RunAllTests or num_tests >= total_combinations:
            all_combinations = list(itertools.product([0, 1], repeat=len(self.input_vars)))
            num_tests = len(all_combinations)
            input_assignments = [
                dict(zip(self.input_vars, combo)) for combo in all_combinations
            ]
        else:
            # input_assignments = [
            #     {var: random.randint(0, 1) for var in self.input_vars}
            #     for _ in range(num_tests)
            # ]
            # Generate `num_tests` evenly spaced indices
            step = total_combinations / num_tests
            input_assignments = []
            for i in range(num_tests):
                index = int(i * step)
                # Convert index to binary string with padding, then to dict
                bin_string = format(index, f'0{n}b')
                combo = [int(bit) for bit in bin_string]
                input_assignments.append(dict(zip(self.input_vars, combo)))
            

        passed_test_case_count = 0 #Keep track of number of test cases passed
        
        # Step 3: Run tests
        for i, input_assignment in enumerate(input_assignments):
            # input_assignment = {var: random.randint(0, 1) for var in self.input_vars}
            # if(i+1 not in [8, 16, 556, 557, 558, 684, 686, 812, 814, 940, 942, 1068, 1070, 1196, 1198, 1324, 1326, 1452, 1454]):
            #     continue
            golden_output,_ = self.GoldenModel(input_assignment, self.MainBDD_For_GoldenModel)

            # Run model and golden model
            model_output, failureFlag = self.ExecuteDesignsInTopologicalOrder(input_assignment)

            if(failureFlag):
                print('failed on the test case of this...')
                print('input_assignments',input_assignment)
                print('stopped due to error')
                break
            
            #remove unwanted keys that are not part of golden outputs
            model_processed_output = {k: v for k, v in model_output.items() if k in golden_output}
            
            # Compare
            match = (model_processed_output == golden_output)
    
            test_results.append({
                'test_id': i + 1,
                'input': input_assignment,
                'model_output': model_output,
                'golden_output': golden_output,
                'match': match
            })
    
            # Optional: Print mismatches
            print('______________________________________')
            if not match:
                print(f"[❌] Test {i+1} Failed")
                print(f"Input:", {k: input_assignment[k] for k in sorted(input_assignment)})
                print(f"Model Output:", {k: model_output[k] for k in sorted(model_output)})
                print(f"Golden Output:", {k: golden_output[k] for k in sorted(golden_output)})
            else:
                passed_test_case_count += 1
                print(f"[✅] Test {i+1} Passed", model_processed_output)
            print('______________________________________')

            #debug statment break
            # if(i==10):
            #     break

        print(f"Number of test cases passed out of is {passed_test_case_count}/{i+1}")
    
        return test_results
    
    def ExecuteDesignsInTopologicalOrder(self, input_assignment):

        Output = self.Output.copy()
        
        design_ID_item_activation_set = set()

        failureFlag = False
        
        for design_ID_group in self.Topological_order:
            crossbar_design   = self.crossbar_designs[design_ID_group]["crossbar_design_instance"]
            OutputLine_group_Map    = self.crossbar_designs[design_ID_group]["OutputLine_group_Map"]
            DesignIdItemToWordLineInputMap = self.crossbar_designs[design_ID_group]["DesignIdItemToWordLineInputMap"]
            # print('DesignIdItemToWordLineInputMap', DesignIdItemToWordLineInputMap)

            wordLineInputs = []

            # for design_ID_group inputs that are starting designs having initial input current
            if(None in DesignIdItemToWordLineInputMap):
                for wordLineInput in DesignIdItemToWordLineInputMap[None]:
                    wordLineInputs.append(wordLineInput)
                
            design_ID,_ = design_ID_group
            
            # design_ID_item_lst = []
            # for design_ID_item in design_ID:
            #     if(design_ID_item not in DesignIdItemToWordLineInputMap):  #make sure design_ID_item is in the following group
            #         continue
            #     if(design_ID_item is None or design_ID_item in design_ID_item_activation_set):  #check if following word line needs to be activated
            #         for wordLineInput in DesignIdItemToWordLineInputMap[design_ID_item]:
            #             # print('wordLineInput_lst',wordLineInput_lst)
            #             # for wordLineInput in wordLineInput_lst:
            #             wordLineInputs.append(wordLineInput)  # Add mutiple wordline inputs of following design_ID
            #         design_ID_item_lst.append(design_ID_item)

            # print('wordLineInputs',wordLineInputs)

            if(wordLineInputs==[]):
                continue
            
            selector_lines = crossbar_design.ActivateSelectorLines(input_assignment, design_ID_item_activation_set)
            
            design_ID_group_output, outputCurrentMap = crossbar_design.Execute(design_ID_group, wordLineInputs, OutputLine_group_Map, selector_lines)

            #debug
            # print('wordLineInputs',wordLineInputs)
            # print('OutputLine_group_Map',OutputLine_group_Map)
            # print('design_ID_group_output', design_ID_group_output)
            # print()
            
            for outputLabel in design_ID_group_output:
                for wordLineInput in design_ID_group_output[outputLabel]:
                    if(len(outputLabel)>10):   #split_id_item output
                        if(design_ID_group_output[outputLabel][wordLineInput]):
                            design_ID_item_activation_set.add(outputLabel)
                    else:
                        Output[outputLabel] = max(design_ID_group_output[outputLabel][wordLineInput], Output[outputLabel])

                    outputCurrent = outputCurrentMap[outputLabel][wordLineInput]
                    if(design_ID_group_output[outputLabel][wordLineInput]):
                        self.OnesCurrentFromDesignMap[design_ID_group].append(outputCurrent)
                        if(self.WorstCaseMinOne[design_ID_group]-outputCurrent>1e-6):
                            print('failureFlag 1', self.WorstCaseMinOne[design_ID_group], outputCurrent)
                            failureFlag = True
                    else:
                        self.ZerosCurrentFromDesignMap[design_ID_group].append(outputCurrent)
                        if(outputCurrent-self.WorstCaseMaxZero[design_ID_group]>0):
                            failureFlag = True
                            print('failureFlag 0', self.WorstCaseMaxZero[design_ID_group], outputCurrent,outputLabel)
        return Output, failureFlag
            
#Program the crossbar on 1024x1024 designs (Dimentions Custom)
#Have bus serice activated in each design

#Set selector lines

#Execute functions that will run the subtree design which will Give results of that subtree

#Generate Random Test cases for a split bdd
#Verify the input and output in single design of crossbar

#Generate Random Test cases and get the output of testcases from main bdd
#Verify the input and output in full designs of crossbars

#Have a topological plot of the graph (Optional)

class PATH_Design_Analog_Verification:
    def __init__(self, Bus, Design_Map, CrossbarGridSize = 1024):
        self.Design_Map = Design_Map
        self.Bus = Bus

        self.RowOffSet = 0
        self.ColOffSet = 0

        self.SelectorLineLabels = []

        self.Crossbar = np.zeros((CrossbarGridSize, CrossbarGridSize), dtype=np.uint8)

        self.Crossbar_Execution = None

        self.ColOffSetForDesign_ID_group = {}

        self.DesignIdToWordLineInputMap = {}

        self.HRS = {design_ID:None for design_ID in self.Design_Map}

        for design_ID, processed_graph_map_value_map in self.Design_Map.items():
            processed_group_graph              = processed_graph_map_value_map['processed_group_graph']
            Crossbar_design                    = processed_graph_map_value_map['Crossbar_design']
            Selector_Lines_Map                 = processed_graph_map_value_map['Selector_Lines_Map']
            OutputLine_group_Map               = processed_graph_map_value_map['OutputLine_group_Map']
            DesignIdItemToWordLineInputMap     = processed_graph_map_value_map['DesignIdItemToWordLineInputMap']
            LongestPath                        = processed_graph_map_value_map['LongestPaths']
            OutputLine_group_selectorLines_Map = processed_graph_map_value_map['OutputLine_group_selectorLines_Map']
            
            DesignIdToWordLineInputMap, (ColStart, ColEnd) = self.ProgramCrossbar(Crossbar_design, Selector_Lines_Map, DesignIdItemToWordLineInputMap)
            # print(DesignIdToWordLineInputMap, (ColStart, ColEnd))

            self.DesignIdToWordLineInputMap[design_ID] = DesignIdToWordLineInputMap
            
            self.ColOffSetForDesign_ID_group[design_ID] = (ColStart, ColEnd)
            
            self.Bus.connect(self, design_ID, DesignIdToWordLineInputMap, OutputLine_group_Map, processed_group_graph)

            # break
            

        self.SelectorLinesOutputLabelsToBitlineIndex = {SelectorLineLabel:index for index, SelectorLineLabel in enumerate(self.SelectorLineLabels) if 'O' in SelectorLineLabel}

    def ProgramCrossbar(self, Crossbar_design, Selector_Lines_Map, DesignId_To_WordLineNumber_Map):
        
        Crossbar_design = np.array(Crossbar_design)  # Ensure it's a NumPy array
        
        num_rows, num_cols = Crossbar_design.shape

        row_start = self.RowOffSet
        row_end = self.RowOffSet + num_rows
        col_start = self.ColOffSet
        col_end = self.ColOffSet + num_cols

        # print(Crossbar_design.shape)
        # print(row_start, col_start)
        # print(row_end, col_end)
        # print()
        
        # Efficient slice assignment
        self.Crossbar[row_start:row_end, col_start:col_end] = Crossbar_design

        # Offset input row mapping
        # DesignIdToWordLineInputMap = {
        #     split_id: row + row_start
        #     for split_id, row in DesignId_To_WordLineNumber_Map.items()
        # }

        DesignIdToWordLineInputMap = {}
        for split_id, rows in DesignId_To_WordLineNumber_Map.items():
            if(split_id not in DesignIdToWordLineInputMap):
                DesignIdToWordLineInputMap[split_id]= []
            for j, row in enumerate(rows):
                DesignIdToWordLineInputMap[split_id].append(row + row_start)
                
        DesignIdToWordLineInputMap[None] = [row_start]
        
        # Update Row/Col offsets
        self.RowOffSet = row_end
        self.ColOffSet = col_end
        
        self.SelectorLineLabels.extend(Selector_Lines_Map)
        
        return DesignIdToWordLineInputMap, (col_start, col_end)

    def ActivateSelectorLines(self, InputAssignmentMap, design_ID_item_activation_set):
        #Selecting selector lines based on the InputAssignmentMap (Boolean literals)
        selector_lines = [0 for _ in self.SelectorLineLabels]
        for i, SelectorLineLabel in enumerate(self.SelectorLineLabels):
            if(len(SelectorLineLabel)>20):
                if(SelectorLineLabel in design_ID_item_activation_set): #this input selectorline
                    selector_lines[i] = 1
                    # print('came here',i,SelectorLineLabel)
                continue
                
            if('O' not in SelectorLineLabel):
                if('~'==SelectorLineLabel[0] and InputAssignmentMap[SelectorLineLabel[1:]]==0):
                    selector_lines[i] = 1
                elif(SelectorLineLabel in InputAssignmentMap and InputAssignmentMap[SelectorLineLabel]==1):
                    selector_lines[i] = 1
            else:
                selector_lines[i] = 1

        return selector_lines

    def TimeMultiplexCrossbar(self, Crossbar_, nonOutputBitline_indexes):
        # Crossbar = [row.copy() for row in Crossbar_]
        # for nonOutputBitline_index in nonOutputBitlines:
        #     for row_i in range(len(Crossbar)):
        #         Crossbar[row_i][nonOutputBitline_index] = 2
        Crossbar = Crossbar_.copy()
        Crossbar[:, nonOutputBitline_indexes] = 2
        
        return Crossbar

    def find_path_execution_in_crossbar(self, Crossbar, wordLineInputs, outputBitlineIndex):
        # Custom function to emulate an ordered set using a list
        def add_to_ordered_set(ordered_set, element):
            if element not in ordered_set:
                ordered_set.append(element)

        R_LRS = 1

        visited = set()
        
        # Stack for depth-first traversal
        Stack = []
        
        # Finding paths
        for wordLineInput in wordLineInputs:
            for j in range(len(Crossbar[wordLineInput])):
                if Crossbar[wordLineInput][j] == R_LRS:
                    visited.add((wordLineInput, j))
                    Stack.append([(wordLineInput, j), 'w'])  # Use a list for ordered visited nodes

        # print('Crossbar[0]',Crossbar[0])
        # print('Stack',Stack)
        # print('outputBitlineIndex',outputBitlineIndex)
        while Stack:
            [(path_i, path_j), last_curr] = Stack.pop(0)  # Pop from the queue (FIFO)
            for i in range(len(Crossbar)):
                if last_curr == 'w':
                    if Crossbar[i][path_j] == R_LRS and  i!=path_i and (i, path_j) not in visited:
                        visited.add((i, path_j))
                        Stack.append([(i, path_j), 'b'])
                elif last_curr == 'b':
                    if Crossbar[path_i][i] == R_LRS and i!=path_j and (path_i, i) not in visited:
                        visited.add((path_i, i))
                        Stack.append([(path_i, i), 'w'])
            # print('path_j', path_i, path_j)
            if(path_j==outputBitlineIndex):
                return True
        return False
        
    def Execute(self, design_ID_group, wordLineInputs, OutputLine_group_Map, selector_lines, testing_Individual_Design_Cases=False):

        # print('wordLineInputs', wordLineInputs)
        # print('OutputLine_group_Map', OutputLine_group_Map)

        # print('design_ID_group',design_ID_group)

        Output = {}
        OutputCurrentMap = {}

        #expand mask to not have any selector lines other than the design_id_group columns
        selector_lines_for_design = selector_lines.copy()

        col_start, col_end = self.ColOffSetForDesign_ID_group[design_ID_group]

        for selector_line_index in range(len(selector_lines_for_design)):
            if(selector_line_index not in range(col_start, col_end)):
                selector_lines_for_design[selector_line_index] = 0
        
        OutputLine_group_selectorLines_Map = self.Design_Map[design_ID_group]['OutputLine_group_selectorLines_Map']
        
        design_ID_List = []
        for outputLine in OutputLine_group_Map:
            
            outputBitlineIndex = self.SelectorLinesOutputLabelsToBitlineIndex[outputLine]

            # print('outputBitlineIndex',outputBitlineIndex, outputLine, wordLineInputs)

            # not sure
            nonOutputBitline_indexes = [
                self.SelectorLinesOutputLabelsToBitlineIndex[OutputLineLabel] 
                for OutputLineLabel in OutputLine_group_Map 
                if outputBitlineIndex!=self.SelectorLinesOutputLabelsToBitlineIndex[OutputLineLabel]
            ]
            # print('nonOutputBitline_indexes',nonOutputBitline_indexes)

            ###############################
            design_cols_of_output_label = set(OutputLine_group_selectorLines_Map[outputLine])
        
            #expand mask to not have any selector lines in design_id_group that are not relavent to the output
            selector_lines_for_output = selector_lines_for_design.copy()

            for col_index in range(len(selector_lines_for_output)):
                if(col_index-col_start not in design_cols_of_output_label):  #need to add col_start t0 design_cols_of_output_label
                    selector_lines_for_output[col_index] = 0

            n_cols = self.Crossbar.shape[1] # 1024
            mask   = np.zeros(n_cols, dtype=bool)      # all False
        
            # fill the first 162 positions with your selector condition
            mask[:len(selector_lines_for_output)] = (np.array(selector_lines_for_output) == 0)

            # Make the rest of the values in the mask to be true
            mask[len(selector_lines_for_output):] = True

            crossbar_excution = self.Crossbar.copy()

            # broadcast one assignment
            crossbar_excution[:, mask] = 2
            ###############################

            # print('a',len(mask))
            # print('b true 0',len([i for i,ma in enumerate(mask) if ma==True]))
            # print('c false 1',len([i for i,ma in enumerate(mask) if ma==False]))
            # print('selector_lines_for_design',sum(selector_lines_for_design))
            # print('selector_lines_for_output',sum(selector_lines_for_output))
            
            MultiplexedCrossbar = self.TimeMultiplexCrossbar(crossbar_excution, nonOutputBitline_indexes)

            # wordLineInputs
            # print('wordLineInputs',wordLineInputs)

            for wordLineInput in wordLineInputs:
            
                # code to execute crossbar
                PathCurrent = self.find_path_execution_in_crossbar(MultiplexedCrossbar, [wordLineInput], outputBitlineIndex)
    
                outputCurrent = self.find_path_current_execution_in_crossbar(design_ID_group, MultiplexedCrossbar, [wordLineInput], outputBitlineIndex)
            
                # self.VisuvaliseCrossbar(MultiplexedCrossbar)
                # print('---------------------------------------------------',outputCurrent)
                # print(wordLineInputs, outputBitlineIndex)
                # for row in MultiplexedCrossbar:
                #     print(list([int(num) for num in row]))
                # print('---------------------------------------------------')
                
                # print('PathCurrent',PathCurrent)
                # print('MultiplexedCrossbar', len(MultiplexedCrossbar), len(MultiplexedCrossbar[0]), wordLineInputs, outputBitlineIndex)

                if(OutputLine_group_Map[outputLine] not in OutputCurrentMap):
                    OutputCurrentMap[OutputLine_group_Map[outputLine]] = {}
                OutputCurrentMap[OutputLine_group_Map[outputLine]][wordLineInput] = outputCurrent

                if(OutputLine_group_Map[outputLine] not in Output):
                    Output[OutputLine_group_Map[outputLine]] = {}
                Output[OutputLine_group_Map[outputLine]][wordLineInput] = 1 if PathCurrent else 0

                
        return Output, OutputCurrentMap
    
    def find_path_current_execution_in_crossbar(self, design_ID, Crossbar, wordLineInputs, outputBitlineIndex, R_HRS=4e7):
        
        # print('wordLineInputs',wordLineInputs)
        # print('outputBitlineIndex',outputBitlineIndex)
        # Custom function to emulate an ordered set using a list
        def add_to_ordered_set(ordered_set, element):
            if element not in ordered_set:
                ordered_set.append(element)

        # R_Off = self.R_Off  # Very large (transistor off)
        # R_HRS = self.R_HRS_Map[design_ID]  # High-resistance state of the memory cell
        # R_LRS = self.R_LRS  # Low-resistance state

        # R_HRS = 5e5
        if(self.HRS[design_ID] is not None):
            R_HRS,_ = self.HRS[design_ID]

        # print('R_HRS',R_HRS)
        
        # R_HRS = 4e7
        R_LRS = 1000
        
        R_Line_Out = 200  # 200 ohms from each column node to GND
        # R_Not = 1e12  # Large resistance for non-output columns
        R_Not = float('inf') # Large resistance for non-output columns
        R_source   = 100  # Series resistor from 0.2 V source to row 0
        R_Off = 1e12
        
        # Voltage source for the first row (0.2V)
        Vsrc = 0.2  

        # for row in Crossbar:
        #     print(row)



        Resistance_matrix = np.where(Crossbar == 0, R_HRS, 
                                    np.where(Crossbar == 1, R_LRS, 
                                             np.where(Crossbar == 2, R_Off, Crossbar)))
        
        # Resistance_matrix = np.where(Crossbar_design == 1,
        #                        self.R_LRS,          # low-resistance state
        #                        self.R_HRS_Map[design_ID])          # high-resistance state

        # -----------------------------
        # 2) Prepare the crossbar data
        # -----------------------------
        crossbar_size = Resistance_matrix.shape[0]

        # -----------------------------
        # 3) Construct the KCL system
        #    We have #rows + #columns unknowns:
        #       Vr0, Vr1, ..., Vr(N-1), Vc0, Vc1, ..., Vc(N-1)
        # -----------------------------
        num_vars = 2 * crossbar_size  
        A = np.zeros((num_vars, num_vars))  # Coefficient matrix
        b = np.zeros(num_vars)  # Constant vector

        # -----------------------------
        # 3a) Row equations (KCL at each row node)
        # -----------------------------
        for i in range(crossbar_size):
            if i in wordLineInputs:
                #
                # For row 0, we have an incoming/outgoing current through R_source to the 0.2V supply.
                # The KCL for row 0 is:
                #   Σ_j (Vr0 - Vc_j)/R(i,j) + (Vr0 - 0.2)/R_source = 0
                #
                for j in range(crossbar_size):
                    Rij = Resistance_matrix[i, j]
                    A[i, i]                += 1.0 / Rij    # +1/Rij for Vr[i]
                    A[i, crossbar_size + j] -= 1.0 / Rij    # -1/Rij for Vc[j]

                # Now add the series resistor with the source (0.2 V):
                A[i, i] += 1.0 / R_source  # Coefficient for Vr0
                
                # Move the known source voltage part to the RHS
                b[i] = (Vsrc / R_source)
    
            else:
                # For row i (i > 0), normal crossbar KCL with no direct voltage source:
                #   Σ_j (Vri - Vcj)/R(i,j) = 0
                for j in range(crossbar_size):
                    Rij = Resistance_matrix[i, j]
                    A[i, i]                += 1.0 / Rij    # Coefficient for Vr_i
                    A[i, crossbar_size + j] -= 1.0 / Rij   # Coefficient for Vc_j
                # b[i] remains 0
    
        # -----------------------------
        # 3b) Column equations (KCL at each column node)
        #       If the column is in output_bitlines => goes to GND through R_Line_Out
        #       Otherwise => goes to GND through R_Not
        # -----------------------------
        # print('output_bitlines',output_bitlines)
        for j in range(crossbar_size):
            if j == outputBitlineIndex:
                R_ground = R_Line_Out  # If it's an output bitline, connects to ground through R_Line_Out
            else:
                R_ground = R_Not  # If not an output bitline, connect to ground through R_Not

            # The KCL for column j is:
            #    Vc_j / R_ground + Σ_i (Vc_j - Vr_i)/R(i,j) = 0
            #
            # Expand:
            #    (Vc_j / R_ground) + Σ_i (Vc_j / R(i,j) - Vr_i / R(i,j)) = 0
            # => (1/R_ground + Σ_i (1/R(i,j))) * Vc_j  - Σ_i(1/R(i,j)) * Vr_i = 0
            
            A[crossbar_size + j, crossbar_size + j] = 1 / R_ground  # Self term
            for i in range(crossbar_size):
                Rij = Resistance_matrix[i][j]
                A[crossbar_size + j, crossbar_size + j] += 1 / Rij  # Self term
                A[crossbar_size + j, i] -= 1 / Rij  # Interaction with row

        # -----------------------------
        # 4) Solve the system A x = b
        #    where x = [Vr_0, ..., Vr_(N-1), Vc_0, ..., Vc_(N-1)]
        # -----------------------------
        solution = np.linalg.solve(A, b)
    
        # Extract row voltages Vr and column voltages Vc
        Vr = solution[:crossbar_size]*1000  # Row voltages
        Vc = solution[crossbar_size:]*1000  # Column voltages
    
        # Compute currents through each resistor
        currentInDevices = (Vr[:, None] - Vc[None, :]) / Resistance_matrix
    
        # -----------------------------
        # 5) Compute output currents
        #    The current from each output column j is simply
        #    I_out(j) = Vc_j / R_Line_Out (if j is an output bitline)
        # -----------------------------
        CurrentOutput = Vc[outputBitlineIndex] / R_Line_Out
        # print('CurrentOutput', CurrentOutput, outputBitlineIndex)
        
        return CurrentOutput

        
    def VisuvaliseCrossbar(self, initialisedCrossbar):

        colors = ["red", "blue", "black"]
        if(max(initialisedCrossbar[0])!=2):
            colors = colors[:-1]
        custom_cmap = ListedColormap(colors)
        initialisedCrossbar = np.array(initialisedCrossbar)
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(initialisedCrossbar, cmap=custom_cmap, aspect="auto")

        # Annotate the heatmap with 'LRS', 'HRS', or 'Off'
        for i in range(initialisedCrossbar.shape[0]):
            for j in range(initialisedCrossbar.shape[1]):
                if initialisedCrossbar[i, j] == 1:
                    text = "LRS"        # Same label for value=1
                    font_color = "white"
                elif initialisedCrossbar[i, j] == 2:
                    text = "Off"        # New label for value=2
                    font_color = "white"
                elif initialisedCrossbar[i, j] == 2:
                    text = "On"        # New label for value=2
                    font_color = "green"
                else:  # Assume any other value (including 0) is HRS
                    text = "HRS"
                    font_color = "black"
        
                ax.text(
                    j, i,
                    text,
                    ha="center",
                    va="center",
                    color=font_color,
                    fontsize=10,
                    weight="bold"
                )

        # Add gridlines for cell borders
        ax.set_xticks(np.arange(initialisedCrossbar.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(initialisedCrossbar.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Set labels and title
        ax.set_xlabel("Columns", fontsize=12, weight="bold")
        ax.set_ylabel("Rows", fontsize=12, weight="bold")
        ax.set_title("Crossbar Resistive States", fontsize=14, weight="bold")

        #shif this to functional functions
        # xlabelsOfLiterals ['a1', 'a0', '~a1', '~a1', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']

        xlabelsOfLiterals = self.SelectorLineLabels
        # Pad with 'buffer' if needed
        num_columns = initialisedCrossbar.shape[1]
        if len(xlabelsOfLiterals) < num_columns:
            xlabelsOfLiterals += ['NA'] * (num_columns - len(xlabelsOfLiterals))
        
        # Remove default ticks
        ax.set_xticks(np.arange(initialisedCrossbar.shape[1]))
        ax.set_yticks(np.arange(initialisedCrossbar.shape[0]))
        ax.set_xticklabels(xlabelsOfLiterals, fontsize=10)
        ax.set_yticklabels([f"{j+1}" for j in range(initialisedCrossbar.shape[1])], fontsize=10)

        # Show the heatmap
        plt.colorbar(heatmap, label="Resistive State", orientation="vertical")
        plt.tight_layout()
        plt.show()

    def CreateCrossbarDesignFromLongestPath(self, LongestPath, crossbar_design_size, crossbar_size):
        crossbar = np.full(crossbar_size, 2)
        path_cols = {y for (_, y) in LongestPath}
        path_rows = {x for (x, _) in LongestPath}
        
        crossbar[:,list(path_cols)] = 0
        
        for (x, y) in LongestPath:
            crossbar[x][y] = 1

        # count_j = 0
        # for col_j in range(len(crossbar)):
        #     if(col_j not in path_cols):
        #         crossbar[:,col_j] = 0
        #         count_i = 0
        #         for row_i in range(len(crossbar)):
        #             if(row_i not in path_rows):
        #                 crossbar[row_i][col_j] = 1
        #                 path_rows.add(row_i)
        #                 count_i+=1
        #                 if(count_i==2):
        #                     break
        #         path_cols.add(col_j)
        #         count_j+=1
        #         if(count_j==2):
        #             break

        return crossbar

    def CreateCrossbarDesignFromDesign(self, crossbar_size, Crossbar_design, EnabledSelectorLines):
        crossbar = np.array(Crossbar_design, dtype=np.uint8)  # Efficient copy with correct type
    
        # Resize or pad if Crossbar_design is smaller than crossbar_size
        if crossbar.shape != crossbar_size:
            full_crossbar = np.zeros(crossbar_size, dtype=np.uint8)
            full_crossbar[:crossbar.shape[0], :crossbar.shape[1]] = crossbar
            crossbar = full_crossbar
    
        # Set entire columns to 2 (disabled) where selector is not enabled
        all_columns = np.arange(crossbar.shape[1])
        # print('all_columns',all_columns)
        disabled_columns = np.setdiff1d(all_columns, EnabledSelectorLines)
        # print('disabled_columns',disabled_columns)
        crossbar[:, disabled_columns] = 2
    
        return crossbar

    def ModifyCrossbarforZeroCurrent(self, crossbar, inputRow, outputCol):

        mod_crossbar = crossbar.copy()
        rows, cols = len(mod_crossbar), len(mod_crossbar[0])
        R_LRS = 1
        all_paths = []
        
        rows_used = set()
        cols_used = set()
        add_rows = 1  #make it divisible by 2

        includedCutoffPaths = set()
        path_present = False

        # print('WordLineInputs_zero',WordLineInputs_zero, len(includedCutoffPaths))
        
        def dfs(position, visited, current_path, last_type, mainSplit=True):
            nonlocal add_rows, path_present
            row, col = position
            
            rows_used.add(row)
            cols_used.add(col)
            # print('current_path',current_path[-1], last_type)
            # print('col',col)
    
            # Check if reached output column
            if col == outputCol and last_type == 'w':
    
                # Break the path in the middle
                d = len(current_path)
                mid_index = (d // 2)
                if(mid_index%2==0 and d>1):
                    mid_index+=1

                flag = True
                pre_foundIndex = None
                if(not mainSplit):
                    for i,cell in enumerate(current_path):
                        if(cell in includedCutoffPaths and i<1):
                            path_present = True
                            break
                        elif(cell in includedCutoffPaths and i>1):
                            pre_foundIndex = i-1
                            flag=False
                            break
                    
                if(flag):
                    # print('flag',pre_foundIndex)
                    # print('mid_index',mid_index, current_path[mid_index-1], current_path[mid_index])
                    mid_row, mid_col = current_path[mid_index]
                    mod_crossbar[mid_row][mid_col] = 0  # Disable the path at midpoint
                else:
                    # print('pre_foundIndex',pre_foundIndex,len(includedCutoffPaths))
                    pre_foundIndex_row, pre_foundIndex_col = current_path[pre_foundIndex]
                    mod_crossbar[pre_foundIndex_row][pre_foundIndex_col] = 0

                for cell in current_path:
                    includedCutoffPaths.add(cell)
                # print('mid_row, mid_col', mid_row, mid_col)
                return path_present

            if last_type == 'w':
                # From wordline → traverse column (same col, all rows)
                for r in range(rows):
                    if r != row and mod_crossbar[r][col] == R_LRS and (r, col) not in visited:
                        visited.add((r, col))
                        current_path.append((r, col))
                        dfs((r, col), visited, current_path, 'b')
                        current_path.pop()
                        visited.remove((r, col))
            elif last_type == 'b':
                # From bitline → traverse row (same row, all cols)
                for c in range(cols):
                    if c != col and mod_crossbar[row][c] == R_LRS and (row, c) not in visited:
                        visited.add((row, c))
                        current_path.append((row, c))
                        dfs((row, c), visited, current_path, 'w')
                        current_path.pop()
                        visited.remove((row, c))
                        
        # Start DFS from all LRS cells in input rows
        for c in range(cols):
            if mod_crossbar[inputRow][c] == R_LRS:
                start = (inputRow, c)
                # print('Starting dfs...')
                dfs(start, {start}, [start], 'w')

        WordLineInputs = [inputRow]
        # WordLineInputs = []
        # for WordLineInput in WordLineInputs_zero:
        #     foundPath = self.find_path_execution_in_crossbar(mod_crossbar, [WordLineInput], outputCol)
        #     if(foundPath):
        #         for c in range(cols):
        #             if mod_crossbar[WordLineInput][c] == R_LRS:
        #                 start = (WordLineInput, c)
        #                 path_present = False
        #                 path_present = dfs(start, {start}, [start], 'w', mainSplit=False)
        #                 # print('path_present',path_present)
        #         if(not path_present):
        #             WordLineInputs.append(WordLineInput)
        #     else:
        #         WordLineInputs.append(WordLineInput)

        # missing = list(set(WordLineInputs_zero)-set(WordLineInputs))
        # print('missing',missing)

        # print('WordLineInputs chance',WordLineInputs)
        return mod_crossbar

    def ModifyCrossbarforZeroCurrent2(self, crossbar_size, disconnected_cells, EnabledSelectorLines):
        crossbar = np.zeros(crossbar_size, dtype=np.uint8)
    
        for disconnected_cell in disconnected_cells:
            row_i, col_j = disconnected_cell
            crossbar[row_i][col_j] = 1
    
        # Set entire columns to 2 (disabled) where selector is not enabled
        all_columns = np.arange(crossbar.shape[1])
        # print('all_columns',all_columns)
        disabled_columns = np.setdiff1d(all_columns, EnabledSelectorLines)
        # print('disabled_columns',disabled_columns)
        crossbar[:, disabled_columns] = 2
    
        return crossbar

    def Activecells(self, Crossbar_zero_mod):
        active_cells = []
        for row_i in range(len(Crossbar_zero_mod)):
            for col_j in range(len(Crossbar_zero_mod[0])):
                if(Crossbar_zero_mod[row_i][col_j]==1):
                    active_cells.append((row_i, col_j))
        return active_cells
                                            
    def FindOptimalHRSvalue(self, design_ID_group, OutputLine_group_Map, wordLineInputs, max_iterations=6):

        print('-------------------------------------------------------------------')
        print('design_ID_group',design_ID_group)
        
        Guardbound_Threshold = 1e-3
        best_guardBound = None

        Crossbar_design                    = self.Design_Map[design_ID_group]['Crossbar_design']
        OutputLine_group_selectorLines_Map = self.Design_Map[design_ID_group]['OutputLine_group_selectorLines_Map']
        OutputLine_group_Map               = self.Design_Map[design_ID_group]['OutputLine_group_Map']
        Selector_Lines_Map                 = {label:index for index,label in enumerate(self.Design_Map[design_ID_group]['Selector_Lines_Map'])}

        ############### Prepare for finding minimum Ones current ####################

        self.CrossbarOneCache = {}
        
        LongestPaths = self.Design_Map[design_ID_group]['LongestPaths']
        # print('LongestPaths',LongestPaths)

        crossbar_design_size=(len(Crossbar_design), len(Crossbar_design[0]))
        crossbar_size=(self.Crossbar.shape[0], self.Crossbar.shape[1])

        for LongestPath in LongestPaths:
            # Create Crossbar design from LongestPath
            Crossbar_one = self.CreateCrossbarDesignFromLongestPath(LongestPath, 
                                                                crossbar_design_size=crossbar_design_size,
                                                                crossbar_size=crossbar_size)
        
            self.CrossbarOneCache[tuple(LongestPath)] = Crossbar_one
        
            # outputBitlineIndex_one = LongestPath[-1][1]

        # print('LongestPath 1',LongestPath)
        # print('foundPath 1',foundPath)
        # for row in Crossbar_one:
        #     print(row,'a')

        ##################### prepare for Finding maximum Zeros current ####################

        self.CrossbarZeroCache = {}
        self.CrossbarZeroCache1 = {}
        
        WordLineInput_zero = self.DesignIdToWordLineInputMap[design_ID_group][None][0]

        WordLineInput_zero = 0
        
        Crossbar_disabledOutputsToInputs = self.Design_Map[design_ID_group]['Crossbar_disabledOutputsToInputs']
        
        for outputLabel in OutputLine_group_Map:

            outputBitlineIndex_zero       = Selector_Lines_Map[outputLabel]

            # outputBitlineIndex_zero = self.SelectorLinesOutputLabelsToBitlineIndex[outputLabel]
            
            # print('outputBitlineIndex_zero', outputBitlineIndex_zero, outputLabel, WordLineInput_zero)
            
            EnabledSelectorLines = OutputLine_group_selectorLines_Map[outputLabel]

            Crossbar_zero = self.CreateCrossbarDesignFromDesign(
                                            crossbar_size,
                                            Crossbar_design,
                                            EnabledSelectorLines)

            Crossbar_zero_mod = self.ModifyCrossbarforZeroCurrent(Crossbar_zero, WordLineInput_zero, outputBitlineIndex_zero)

            # print('Activecells',set(self.Activecells(Crossbar_zero_mod)))

            # print('WordLineInput_zero',WordLineInput_zero)

            foundPath = self.find_path_execution_in_crossbar(Crossbar_zero_mod, [WordLineInput_zero], outputBitlineIndex_zero)

            # print('foundPath 0', foundPath,'============================================')
            if foundPath:
                raise RuntimeError(f"Unexpected path found for design_ID_group={design_ID_group}, "
                                   f"outputLabel={outputLabel}, WordLineInput={WordLineInput_zero}")

            cache_key = (design_ID_group, outputLabel)
            self.CrossbarZeroCache[cache_key] = Crossbar_zero_mod, WordLineInput_zero

            disconnected_cells = Crossbar_disabledOutputsToInputs[outputLabel]['disconnected_cells']

            # print('disconnected_cells',disconnected_cells, outputLabel)
            # print('disconnected_cells',{col_j for _,col_j in disconnected_cells})
            # print('EnabledSelectorLines',EnabledSelectorLines)
            # print()
            
            Crossbar_zero_mod = self.ModifyCrossbarforZeroCurrent2(crossbar_size, disconnected_cells, EnabledSelectorLines)

            
            foundPath = self.find_path_execution_in_crossbar(Crossbar_zero_mod, [WordLineInput_zero], outputBitlineIndex_zero)
            
            # print('foundPath 0', foundPath,'============================================')
            if foundPath:
                raise RuntimeError(f"Unexpected path found for design_ID_group={design_ID_group}, "
                                   f"outputLabel={outputLabel}, WordLineInput={WordLineInput_zero}")

            cache_key = (design_ID_group, outputLabel)
            self.CrossbarZeroCache1[cache_key] = Crossbar_zero_mod, WordLineInput_zero
        
        
        ######################## Binary Search for R_HRS ########################
        low = 0
        high = 1e10
        best_R_HRS = None
        GuardBound = None
        # max_iterations = 2
        iteration = 0

        OnesCurrents = []
    
        while high - low > Guardbound_Threshold and iteration < max_iterations:
            mid = (low + high) // 2
            R_HRS = mid

            for LongestPath in LongestPaths:
                WordLineInput_one = LongestPath[0][0]
                
                Crossbar_one = self.CrossbarOneCache[tuple(LongestPath)]
                
                outputBitlineIndex_one = LongestPath[-1][1]
                
                OnesCurrent = self.find_path_current_execution_in_crossbar(
                    design_ID_group, Crossbar_one, [WordLineInput_one], outputBitlineIndex_one, R_HRS=R_HRS
                )
                
                OnesCurrents.append(OnesCurrent)

            OnesCurrent = min(OnesCurrents)

            print('start zero')
            # ZerosCurrent calculation
            ZerosCurrent_lst = []
            ZerosCurrent_lst1 = []
            for outputLabel in OutputLine_group_Map:
                outputBitlineIndex_zero       = Selector_Lines_Map[outputLabel]
                # outputBitlineIndex_zero = self.SelectorLinesOutputLabelsToBitlineIndex[outputLabel]
                
                Crossbar_zero_mod, WordLineInputs_curr = self.CrossbarZeroCache.get((design_ID_group, outputLabel))
                ZerosCurrent = self.find_path_current_execution_in_crossbar(
                    design_ID_group, Crossbar_zero_mod, [WordLineInputs_curr], outputBitlineIndex_zero, R_HRS=R_HRS)
                ZerosCurrent_lst.append(ZerosCurrent)

                Crossbar_zero_mod, WordLineInputs_curr = self.CrossbarZeroCache1.get((design_ID_group, outputLabel))
                ZerosCurrent = self.find_path_current_execution_in_crossbar(
                    design_ID_group, Crossbar_zero_mod, [WordLineInputs_curr], outputBitlineIndex_zero, R_HRS=R_HRS)
                ZerosCurrent_lst1.append(ZerosCurrent)

            ZerosCurrent = max(ZerosCurrent_lst)
            ZerosCurrent1 = max(ZerosCurrent_lst1)
            print('ZerosCurrent',ZerosCurrent)
            print('ZerosCurrent1',ZerosCurrent1)

            ZerosCurrent = ZerosCurrent1

            guardBound = OnesCurrent - ZerosCurrent

            # print('design_ID_group',design_ID_group)
            # print('OnesCurrent',OnesCurrent,"|",'ZerosCurrent',ZerosCurrent)
            print('guardBound', guardBound)
            print('R_HRS',R_HRS)

            if((best_R_HRS is None or R_HRS <= best_R_HRS) and guardBound > Guardbound_Threshold):
                best_R_HRS = R_HRS
                best_guardBound = guardBound
                best_OnesCurrent = OnesCurrent
                best_ZerosCurrent = ZerosCurrent
            
            if guardBound > Guardbound_Threshold:
                high = mid  # Try smaller R_HRS to find minimal satisfying value
            else:
                low = mid  # Increase R_HRS
    
            iteration += 1

        print('design_ID_group', design_ID_group)
        print('ZerosCurrent',ZerosCurrent)
        print('OnesCurrent',OnesCurrent)
        print('best_R_HRS:', best_R_HRS, 'best_guardBound:', best_guardBound)
        print()
        self.HRS[design_ID_group] = (best_R_HRS, best_guardBound)

        if(best_guardBound==None):
            raise ValueError("best_guardBound is None. Expected a valid value.")

        return best_ZerosCurrent, best_OnesCurrent
            

    def RunRandomTestCasesOnEachDesign(self, num_tests=10):

        design_IDs = list(self.Bus.crossbar_designs.keys())
        testNum = 0

        for z, design_ID in enumerate(design_IDs):
            print("Design:",z,design_ID)
            crossbar_design = self.Bus.crossbar_designs[design_ID]["crossbar_design_instance"]
            wordLineInput   = self.Bus.crossbar_designs[design_ID]["wordLineInput"]
            OutputLine_group_Map  = self.Bus.crossbar_designs[design_ID]["OutputLine_group_Map"]

            DesignBDDgraph = self.Design_Map[design_ID]['processed_group_graph']
        
            #Create inputs for each bdd design
            # Step 1: Extract unique positive variable names from SelectorLineLabels
            input_vars = set()
            SelectorLineLabels = self.SelectorLineLabels.copy()
            for label in SelectorLineLabels:
                var = label.replace('~', '')
                if 'O' not in var:
                    input_vars.add(var)
            input_vars = list(input_vars)
        
            # Step 3: Generate and test random input assignments
            test_results = []
            
            for _ in range(num_tests):
                # create cases for those inputs
                input_assignment = {var: random.randint(0, 1) for var in input_vars}
        
                # Copy the main crossbar design to execution crossbar to run executions
                self.Crossbar_Execution = self.Crossbar.copy()
        
                #Selecting selector lines based on the input_assignment (Boolean literals)
                selector_lines = [0 for _ in self.SelectorLineLabels]
                for i, SelectorLineLabel in enumerate(self.SelectorLineLabels):
                    if('O' not in SelectorLineLabel):
                        if('~'==SelectorLineLabel[0] and input_assignment[SelectorLineLabel[1:]]==0):
                            selector_lines[i] = 1
                        elif(SelectorLineLabel in input_assignment and input_assignment[SelectorLineLabel]==1):
                            selector_lines[i] = 1
                    else:
                        selector_lines[i] = 1
        
                #Setting selectorlines in execution crossbar
                for col_j, selector_line in enumerate(selector_lines):
                    if(not selector_line):
                        for row_i in range(len(self.Crossbar_Execution)):
                            self.Crossbar_Execution[row_i][col_j] = 2
        
                #Create outputMap
                self.Output = {}
            
                # Run model and golden model
                model_output = self.Execute(design_ID, wordLineInput, OutputLine_group_Map, testing_Individual_Design_Cases=True)  #Run each of those tests in its own designs
        
                # print('model_output', model_output)
                
                golden_output = self.GoldenModel(input_assignment, DesignBDDgraph) #Run them in the golden model

                # print('golden_output', golden_output)
        
                # Compare
                match = (model_output == golden_output)
        
                test_results.append({
                    'design_ID': design_ID,
                    'test_id': testNum + 1,
                    'input': input_assignment,
                    'model_output': model_output,
                    'golden_output': golden_output,
                    'match': match
                })
        
                # Optional: Print mismatches
                if not match:
                    print(f"[❌] Test {testNum+1} Failed")
                    print(f"design_ID: {design_ID}")
                    print(f"Input:", {k: input_assignment[k] for k in sorted(input_assignment)})
                    print(f"Model Output:", {k: model_output[k] for k in sorted(model_output)})
                    print(f"Golden Output:", {k: golden_output[k] for k in sorted(golden_output)})
                else:
                    print(f"[✅] Test {testNum+1} Passed", model_output)
                
                testNum+=1
            print()
    
        return test_results