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

class Bus:
    def __init__(self, MainBDD_For_GoldenModel, Topological_order):
        self.crossbar_designs = {}

        self.MainBDD_For_GoldenModel = MainBDD_For_GoldenModel
        self.Topological_order = Topological_order

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
        
    def connect(self, crossbar_design_instance, design_ID, DesignIdItemToWordLineInputMap, OutputLine_group_Map):
        self.crossbar_designs[design_ID] = {
            "crossbar_design_instance":crossbar_design_instance, 
            "DesignIdItemToWordLineInputMap":DesignIdItemToWordLineInputMap, 
            "OutputLine_group_Map":OutputLine_group_Map
        }

    def GoldenModel(self, input_assignment, graph=None):

        singleDesignCheck = True
        if(graph is None):
            singleDesignCheck = False
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
        start_nodes = [n for n in graph.nodes if graph.in_degree(n) == 0]
        visited_outputs = {graph.nodes[n].get('ExpressionRoot'):0 for n in graph.nodes if graph.nodes[n].get('ExpressionRoot') is not None}
        if(singleDesignCheck):
            split_outputs_map = {}
            for n in graph.nodes:
                if graph.nodes[n].get('split_id') is not None and graph.nodes[n].get('split_id').split()[0]=='leaf':
                    successors = list(graph.successors(n))

                    for successor in successors:
                        if('O' in graph.nodes[successor].get('literal')):
                            split_outputs_map[graph.nodes[n].get('split_id').split()[1]] = graph.nodes[successor].get('literal')
                            visited_outputs[graph.nodes[successor].get('literal')] = 0
    
        for start_node in start_nodes:
            stack = [(start_node, [start_node])]
    
            while stack:
                current_node, path = stack.pop()
                node_data = graph.nodes[current_node]
                bipartite_type = node_data.get('BipartitePart')
                literal = node_data.get('literal', '')
                expression_root = node_data.get('ExpressionRoot')
    
                if expression_root is not None:
                    visited_outputs[expression_root] = 1
                    continue
                if(singleDesignCheck):
                    if node_data.get('split_id') is not None and node_data.get('split_id').split()[0] == 'leaf':
                        output_node_label = split_outputs_map[node_data.get('split_id').split()[1]]
                        visited_outputs[output_node_label] = 1
    
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
        return visited_outputs

    def RunRandomTestCases(self, num_tests=10, RunAllTests=False):
        
        test_results = []

        # Step 2: Generate all combinations if RunAllTests is True
        if RunAllTests:
            all_combinations = list(itertools.product([0, 1], repeat=len(self.input_vars)))
            num_tests = len(all_combinations)
            input_assignments = [
                dict(zip(self.input_vars, combo)) for combo in all_combinations
            ]
        else:
            input_assignments = [
                {var: random.randint(0, 1) for var in self.input_vars}
                for _ in range(num_tests)
            ]

        passed_test_case_count = 0 #Keep track of number of test cases passed
        
        # Step 3: Run tests
        for i, input_assignment in enumerate(input_assignments):
            # input_assignment = {var: random.randint(0, 1) for var in self.input_vars}
            # if(i+1 not in [8, 16, 556, 557, 558, 684, 686, 812, 814, 940, 942, 1068, 1070, 1196, 1198, 1324, 1326, 1452, 1454]):
            #     continue
            golden_output = self.GoldenModel(input_assignment, self.MainBDD_For_GoldenModel)

            # Run model and golden model
            model_output = self.ExecuteDesignsInTopologicalOrder(input_assignment)
            
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
        
        for design_ID_group in self.Topological_order:
            crossbar_design   = self.crossbar_designs[design_ID_group]["crossbar_design_instance"]
            OutputLine_group_Map    = self.crossbar_designs[design_ID_group]["OutputLine_group_Map"]
            DesignIdItemToWordLineInputMap = self.crossbar_designs[design_ID_group]["DesignIdItemToWordLineInputMap"]
            # print('DesignIdItemToWordLineInputMap', DesignIdItemToWordLineInputMap)

            wordLineInputs = []

            # for design_ID_group inputs that are starting designs having initial input current
            if(None in DesignIdItemToWordLineInputMap):
                wordLineInputs.append(DesignIdItemToWordLineInputMap[None])
                
            design_ID,_ = design_ID_group
            # print()
            # print('design_ID',design_ID)
            for design_ID_item in design_ID:
                if(design_ID_item not in DesignIdItemToWordLineInputMap):  #make sure design_ID_item is in the following group
                    continue
                if(design_ID_item is None or design_ID_item in design_ID_item_activation_set):  #check if following word line needs to be activated
                    wordLineInputs.append(DesignIdItemToWordLineInputMap[design_ID_item])  # Add wordline inputs of following design_ID
            # print('wordLineInputs',wordLineInputs)

            if(wordLineInputs==[]):
                continue
            
            selector_lines = crossbar_design.ActivateSelectorLines(input_assignment)
            
            design_ID_group_output = crossbar_design.Execute(design_ID_group, wordLineInputs, OutputLine_group_Map, selector_lines)

            #debug
            # print('design_ID_group_output', design_ID_group_output)
            # print()
            
            for outputLabel in design_ID_group_output:
                if(len(outputLabel)>10):   #split_id_item output
                    if(design_ID_group_output[outputLabel]):
                        design_ID_item_activation_set.add(outputLabel)
                else:
                    Output[outputLabel] = design_ID_group_output[outputLabel]
        
        return Output
            

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

        self.OnesCurrentFromDesignMap = {design_ID:[] for design_ID in self.Design_Map}
        self.ZerosCurrentFromDesignMap = {design_ID:[] for design_ID in self.Design_Map}

        for design_ID, processed_graph_map_value_map in self.Design_Map.items():
            processed_group_graph              = processed_graph_map_value_map['processed_group_graph']
            Crossbar_design                    = processed_graph_map_value_map['Crossbar_design']
            Selector_Lines_Map                 = processed_graph_map_value_map['Selector_Lines_Map']
            OutputLine_group_Map               = processed_graph_map_value_map['OutputLine_group_Map']
            DesignIdItemToWordLineInputMap     = processed_graph_map_value_map['DesignIdItemToWordLineInputMap']
            LongestPath                        = processed_graph_map_value_map['LongestPath']
            OutputLine_group_selectorLines_Map = processed_graph_map_value_map['OutputLine_group_selectorLines_Map']
            
            DesignIdToWordLineInputMap, (ColStart, ColEnd) = self.ProgramCrossbar(Crossbar_design, Selector_Lines_Map, DesignIdItemToWordLineInputMap)
            # print(DesignIdToWordLineInputMap, (ColStart, ColEnd))

            self.ColOffSetForDesign_ID_group[design_ID] = (ColStart, ColEnd)
            
            self.Bus.connect(self, design_ID, DesignIdToWordLineInputMap, OutputLine_group_Map)

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
        DesignIdToWordLineInputMap = {
            split_id: row + row_start
            for split_id, row in DesignId_To_WordLineNumber_Map.items()
        }
        if(DesignIdToWordLineInputMap == {}):
            DesignIdToWordLineInputMap[None] = row_start
        
        # Update Row/Col offsets
        self.RowOffSet = row_end
        self.ColOffSet = col_end
        
        self.SelectorLineLabels.extend(Selector_Lines_Map)
        
        return DesignIdToWordLineInputMap, (col_start, col_end)

    def ActivateSelectorLines(self, InputAssignmentMap):
        #Selecting selector lines based on the InputAssignmentMap (Boolean literals)
        selector_lines = [0 for _ in self.SelectorLineLabels]
        for i, SelectorLineLabel in enumerate(self.SelectorLineLabels):
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
        
        # Stack for depth-first traversal
        Stack = []
        
        # Finding paths
        for wordLineInput in wordLineInputs:
            for j in range(len(Crossbar[wordLineInput])):
                if Crossbar[wordLineInput][j] == R_LRS:
                    Stack.append([(wordLineInput, j), [(wordLineInput, j)], 'w'])  # Use a list for ordered visited nodes

        # print('Crossbar[0]',Crossbar[0])
        # print('Stack',Stack)
        # print('outputBitlineIndex',outputBitlineIndex)
        while Stack:
            [(path_i, path_j), visited, last_curr] = Stack.pop()  # Pop from the stack (LIFO)
            for i in range(len(Crossbar)):
                if last_curr == 'w':
                    if Crossbar[i][path_j] == R_LRS and (i, path_j) not in visited:
                        new_visited = visited.copy()
                        add_to_ordered_set(new_visited, (i, path_j))
                        Stack.append([(i, path_j), new_visited, 'b'])
                elif last_curr == 'b':
                    if Crossbar[path_i][i] == R_LRS and (path_i, i) not in visited:
                        new_visited = visited.copy()
                        add_to_ordered_set(new_visited, (path_i, i))
                        Stack.append([(path_i, i), new_visited, 'w'])
            # print('path_j', path_i, path_j)
            if(path_j==outputBitlineIndex):
                # print('outputBitlineIndex',outputBitlineIndex)
                # print('visited',visited)
                return True
        return False
        
    def Execute(self, design_ID_group, wordLineInputs, OutputLine_group_Map, selector_lines, testing_Individual_Design_Cases=False):

        # print('wordLineInputs', wordLineInputs)
        # print('OutputLine_group_Map', OutputLine_group_Map)

        Output = {}

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

            # print('outputBitlineIndex',outputBitlineIndex, OutputLine_group_Map[outputLine])

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

            # self.VisuvaliseCrossbar(MultiplexedCrossbar)
            
            # code to execute crossbar
            PathCurrent = self.find_path_execution_in_crossbar(MultiplexedCrossbar, wordLineInputs, outputBitlineIndex)
            # print('PathCurrent',PathCurrent)
            # print('MultiplexedCrossbar', len(MultiplexedCrossbar), len(MultiplexedCrossbar[0]), wordLineInputs, outputBitlineIndex)

            outputCurrent = self.find_path_current_execution_in_crossbar(design_ID_group, MultiplexedCrossbar, wordLineInputs, outputBitlineIndex)
            
            if(PathCurrent):
                Output[OutputLine_group_Map[outputLine]] = 1
                self.OnesCurrentFromDesignMap[design_ID_group].append(outputCurrent)
            else:
                Output[OutputLine_group_Map[outputLine]] = 0
                self.ZerosCurrentFromDesignMap[design_ID_group].append(outputCurrent)

        return Output
    
    def find_path_current_execution_in_crossbar(self, design_ID, Crossbar, wordLineInputs, outputBitlineIndex):
        # Custom function to emulate an ordered set using a list
        def add_to_ordered_set(ordered_set, element):
            if element not in ordered_set:
                ordered_set.append(element)

        # R_Off = self.R_Off  # Very large (transistor off)
        # R_HRS = self.R_HRS_Map[design_ID]  # High-resistance state of the memory cell
        # R_LRS = self.R_LRS  # Low-resistance state

        R_HRS = 4e7
        R_LRS = 2000
        
        R_Line_Out = 200  # 200 ohms from each column node to GND
        R_Not = 1e10  # Large resistance for non-output columns
        R_source   = 100  # Series resistor from 0.2 V source to row 0
        R_Off = 1e10
        
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