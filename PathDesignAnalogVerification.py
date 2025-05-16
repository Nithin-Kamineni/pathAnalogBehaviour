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
    def __init__(self):
        self.crossbar_designs = {}
        self.crossbar_wordLineInputs_memories = {}

        self.Topological_order = None

    def connect(self, crossbar_design_instance, design_ID, DesignIdItemToWordLineInputMap, OutputLine_Map, PrerequisiteTrees):
        self.crossbar_designs[design_ID] = {
            "crossbar_design_instance":crossbar_design_instance, 
            "DesignIdItemToWordLineInputMap":DesignIdItemToWordLineInputMap, 
            "OutputLine_Map":OutputLine_Map,
            "PrerequisiteTrees":PrerequisiteTrees,   #Pre-requsite designs to process the current design(Fixed)
        }

        # What input word lines need to be active 
        self.crossbar_wordLineInputs_memories[design_ID] = {DesignIdItem:False for DesignIdItem in DesignIdItemToWordLineInputMap}

    def setTopologicalOrder(self, Topological_order):
        self.Topological_order = Topological_order
    
    def send_signal(self, design_ID_item, signal=True):
        #find all the design_IDs that are having the design_ID_item
        design_ID_lst = [design_ID for design_ID in self.crossbar_designs if design_ID_item in design_ID]
        
        for design_ID in design_ID_lst:
            #design_ID is frozenset
            self.crossbar_wordLineInputs_memories[design_ID][design_ID_item] = (signal or self.crossbar_wordLineInputs_memories[design_ID][design_ID_item])
                

    def ResetExecution(self):
        # Reseting inputs to all world lines
        for design_ID in self.crossbar_wordLineInputs_memories:
            for design_ID_item in design_ID:
                self.crossbar_wordLineInputs_memories[design_ID][design_ID_item] = False

    def ExecuteDesignsInTopologicalOrder(self):
        for design_ID in self.Topological_order:
            crossbar_design   = self.crossbar_designs[design_ID]["crossbar_design_instance"]
            OutputLine_Map    = self.crossbar_designs[design_ID]["OutputLine_Map"]
            PrerequisiteTrees = self.crossbar_designs[design_ID]["PrerequisiteTrees"]
            DesignIdItemToWordLineInputMap = self.crossbar_designs[design_ID]["DesignIdItemToWordLineInputMap"]

            wordLineInputs = []
            if(design_ID==frozenset({})):
                wordLineInputs.append(0)
            else:
                for design_ID_item in design_ID:
                    if(self.crossbar_wordLineInputs_memories[design_ID][design_ID_item]):  #check if following word line needs to be activated
                        wordLineInputs.append(DesignIdItemToWordLineInputMap[design_ID_item])  # Add wordline inputs of following design_ID
                # print('wordLineInputs',wordLineInputs)
            crossbar_design.Execute(design_ID, wordLineInputs, OutputLine_Map)
            

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
    def __init__(self, Bus, Design_Map, MainBDD_For_GoldenModel, CrossbarGridSize = 1024):
        self.Design_Map = Design_Map
        self.Bus = Bus

        # resistor parameters
        self.R_Off      = 4e9  # Very large (transistor off)
        self.R_LRS      = 2000  # Low-resistance state
        self.R_Line_Out = 200  # 200 ohms from each column node to GND
        self.R_Not      = 1e10  # Large resistance for non-output columns
        self.R_source   = 100  # Series resistor from 0.2 V source to row 0

        self.R_HRS_Map  = {design_ID:2e6 for design_ID in self.Design_Map}  # DesignId: resistance... High-resistance state of the memory cell
        self.R_HRS      = self.R_HRS_Map[frozenset({})]
        
        # Voltage source for the first row (0.2V)
        self.Vsrc = 0.2

        self.RowOffSet = 0
        self.ColOffSet = 0

        self.OnesCurrentFromDesignMap = {design_ID:[] for design_ID in self.Design_Map}
        self.ZerosCurrentFromDesignMap = {design_ID:[] for design_ID in self.Design_Map}

        self.MainBDD_For_GoldenModel = MainBDD_For_GoldenModel

        self.SelectorLineLabels = []

        self.Crossbar = np.zeros((CrossbarGridSize, CrossbarGridSize), dtype=np.float32)

        self.Crossbar_Execution = None

        AllPrerequisiteTrees = {}
        for design_ID, processed_graph_map_value_map in self.Design_Map.items():
            AllPrerequisiteTrees[design_ID] = processed_graph_map_value_map['PrerequisiteTrees']

        self.Topological_order = self.Flatten_prerequisite_graph(AllPrerequisiteTrees)
        self.Bus.setTopologicalOrder(self.Topological_order)

        for design_ID, processed_graph_map_value_map in self.Design_Map.items():
            processed_graph          = processed_graph_map_value_map['processed_graph']
            Crossbar_design          = processed_graph_map_value_map['Crossbar_design']
            Selector_Lines_Map       = processed_graph_map_value_map['Selector_Lines_Map']
            OutputLine_Map           = processed_graph_map_value_map['OutputLine_Map']
            DesignIdItemToWordLineInputMap = processed_graph_map_value_map['DesignIdItemToWordLineInputMap']
            LongestPath              = processed_graph_map_value_map['LongestPath']
            PrerequisiteTrees        = processed_graph_map_value_map['PrerequisiteTrees']

            self.R_HRS_Map[design_ID] = 2e6

            DesignIdToWordLineInputMap = self.ProgramCrossbar(design_ID, Crossbar_design, Selector_Lines_Map, DesignIdItemToWordLineInputMap)
            
            self.Bus.connect(self, design_ID, DesignIdToWordLineInputMap, OutputLine_Map, PrerequisiteTrees)

        self.SelectorLinesOutputLabelsToBitlineIndex = {SelectorLineLabel:index for index, SelectorLineLabel in enumerate(self.SelectorLineLabels) if 'O' in SelectorLineLabel}

    def Flatten_prerequisite_graph(self, AllPrerequisiteTrees):
        G = nx.DiGraph()
    
        for design_id, prereq_sets in AllPrerequisiteTrees.items():
            G.add_node(design_id)  # Ensure the node exists
    
            for prereq_set in prereq_sets:
                G.add_edge(prereq_set, design_id)  # prereq_set must come before design_id
    
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Cycle detected in dependency graph!")

        return list(nx.topological_sort(G))

    def ProgramCrossbar(self, design_ID, Crossbar_design, Selector_Lines_Map, DesignId_To_WordLineNumber_Map):
        
        Crossbar_design = np.asarray(Crossbar_design, dtype=np.float32)  # Ensure it's a NumPy array

        # 2) Replace 1 → R_LRS, 0 → R_HRS in a vectorised way
        # Crossbar_design = np.where(Crossbar_design == 1,
        #                        self.R_LRS,          # low-resistance state
        #                        self.R_HRS_Map[design_ID])          # high-resistance state
        
        num_rows, num_cols = Crossbar_design.shape

        row_start = self.RowOffSet
        row_end = self.RowOffSet + num_rows
        col_start = self.ColOffSet
        col_end = self.ColOffSet + num_cols

        # Efficient slice assignment
        self.Crossbar[row_start:row_end, col_start:col_end] = Crossbar_design

        # Offset input row mapping
        DesignIdToWordLineInputMap = {
            split_id: row + row_start
            for split_id, row in DesignId_To_WordLineNumber_Map.items()
        }
        
        # Update Row/Col offsets
        self.RowOffSet = row_end
        self.ColOffSet = col_end
        
        self.SelectorLineLabels.extend(Selector_Lines_Map)
        
        return DesignIdToWordLineInputMap

    def ActivateSelectorLines(self, InputAssignmentMap):

        # Copy the main crossbar design to execution crossbar to run executions
        self.Crossbar_Execution = self.Crossbar.copy()

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

        #Setting selectorlines in execution crossbar
        # for col_j, selector_line in enumerate(selector_lines):
        #     if(not selector_line):
        #         for row_i in range(len(self.Crossbar_Execution)):
        #             self.Crossbar_Execution[row_i][col_j] = 2

        n_cols = self.Crossbar_Execution.shape[1]  # 1024
        mask   = np.zeros(n_cols, dtype=bool)      # all False
        
        # fill the first 162 positions with your selector condition
        mask[:len(selector_lines)] = (np.array(selector_lines) == 0)

        # broadcast one assignment
        self.Crossbar_Execution[:, mask] = 2

        #Create a Output dictionary for storing the output result
        self.Output = {}

        for design_ID in self.Design_Map:
            for outputLabel in self.Design_Map[design_ID]['OutputLine_Map']:
                if len(self.Design_Map[design_ID]['OutputLine_Map'][outputLabel]) < 10:
                    self.Output[self.Design_Map[design_ID]['OutputLine_Map'][outputLabel].split()[0]]=0

        # Resetting crossbars for first execution
        self.Bus.ResetExecution()
        
        #Sending signal to run the first crossbar after setting selector lines in programed crossbar
        self.Bus.ExecuteDesignsInTopologicalOrder()

        return self.Output

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

    def find_path_current_execution_in_crossbar(self, design_ID, Crossbar, wordLineInputs, outputBitlineIndex):
        # Custom function to emulate an ordered set using a list
        def add_to_ordered_set(ordered_set, element):
            if element not in ordered_set:
                ordered_set.append(element)

        R_Off = self.R_Off  # Very large (transistor off)
        R_HRS = self.R_HRS_Map[design_ID]  # High-resistance state of the memory cell
        R_LRS = self.R_LRS  # Low-resistance state

        R_HRS = 2e6
        R_LRS = 2000
        
        R_Line_Out = 200  # 200 ohms from each column node to GND
        R_Not = 1e10  # Large resistance for non-output columns
        R_source   = 100  # Series resistor from 0.2 V source to row 0
        
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
        print('CurrentOutput', CurrentOutput, outputBitlineIndex)
        
        return CurrentOutput
        
        # # Stack for depth-first traversal
        # Stack = []
        
        # # Finding paths
        # for wordLineInput in wordLineInputs:
        #     for j in range(len(Crossbar[wordLineInput])):
        #         if Crossbar[wordLineInput][j] == R_LRS:
        #             Stack.append([(wordLineInput, j), [(wordLineInput, j)], 'w'])  # Use a list for ordered visited nodes

        # # print('Crossbar[0]',Crossbar[0])
        # # print('Stack',Stack)
        # # print('outputBitlineIndex',outputBitlineIndex)
        # while Stack:
        #     [(path_i, path_j), visited, last_curr] = Stack.pop()  # Pop from the stack (LIFO)
        #     for i in range(len(Crossbar)):
        #         if last_curr == 'w':
        #             if Crossbar[i][path_j] == R_LRS and (i, path_j) not in visited:
        #                 new_visited = visited.copy()
        #                 add_to_ordered_set(new_visited, (i, path_j))
        #                 Stack.append([(i, path_j), new_visited, 'b'])
        #         elif last_curr == 'b':
        #             if Crossbar[path_i][i] == R_LRS and (path_i, i) not in visited:
        #                 new_visited = visited.copy()
        #                 add_to_ordered_set(new_visited, (path_i, i))
        #                 Stack.append([(path_i, i), new_visited, 'w'])
        #     # print('path_j', path_i, path_j)
        #     if(path_j==outputBitlineIndex):
        #         # print('outputBitlineIndex',outputBitlineIndex)
        #         # print('visited',visited)
        #         return True
        # return False
        
    def Execute(self, design_ID, wordLineInputs, OutputLine_Map, testing_Individual_Design_Cases=False):

        # print('wordLineInputs', wordLineInputs)
        # print('OutputLine_Map', OutputLine_Map)

        design_ID_graph = self.Design_Map[design_ID]['processed_graph']
        LiteralsInputs = [wordLineInput for wordLineInput in self.Design_Map[design_ID]['DesignIdItemToWordLineInputMap'].values()]
        
        Golden_Unprocessed_OutputMap = self.GoldenModel(self.input_assignment, design_ID_graph, LiteralsInputs, OutputLine_Map)
        
        # Get all outputs from design_ID_graph
        Unprocessed_OutputMap = {}
        for outputLabel in OutputLine_Map:
            Unprocessed_OutputMap[OutputLine_Map[outputLabel]]=0

        design_ID_List = []
        for outputLine in OutputLine_Map:
            outputBitlineIndex = self.SelectorLinesOutputLabelsToBitlineIndex[outputLine]

            # print('outputBitlineIndex',outputBitlineIndex, OutputLine_Map[outputLine])

            # not sure
            nonOutputBitline_indexes = [
                self.SelectorLinesOutputLabelsToBitlineIndex[OutputLineLabel] 
                for OutputLineLabel in OutputLine_Map 
                if outputBitlineIndex!=self.SelectorLinesOutputLabelsToBitlineIndex[OutputLineLabel]
            ]

            # print('nonOutputBitline_indexes',nonOutputBitline_indexes)

            MultiplexedCrossbar = self.TimeMultiplexCrossbar(self.Crossbar_Execution, nonOutputBitline_indexes)

            # code to logically execute crossbar
            FoundPath = self.find_path_execution_in_crossbar(MultiplexedCrossbar, wordLineInputs, outputBitlineIndex)
            print()
            print('FoundPath',FoundPath, outputLine, OutputLine_Map[outputLine])

            # code to logically execute crossbar
            PathCurrent = self.find_path_current_execution_in_crossbar(design_ID, MultiplexedCrossbar, wordLineInputs, outputBitlineIndex)
            # print('PathCurrent',PathCurrent)

            if(FoundPath):
                self.OnesCurrentFromDesignMap[design_ID].append(PathCurrent)
            else:
                self.ZerosCurrentFromDesignMap[design_ID].append(PathCurrent)
            
            # isSplitNode = OutputLine_Map[outputLine].split()[0]=="leaf"
            isSplitNode = len(OutputLine_Map[outputLine])>18

            if(FoundPath):
                # print(isSplitNode, OutputLine_Map, outputLine)
                if(isSplitNode):
                    design_ID_List.append(OutputLine_Map[outputLine])
                else:
                    self.Output[OutputLine_Map[outputLine]] = 1
                Unprocessed_OutputMap[OutputLine_Map[outputLine]] = 1
            else:
                if(isSplitNode):
                    pass

        print(1)
        print('Unprocessed_OutputMap', Unprocessed_OutputMap)
        print()
        print('Golden_Unprocessed_OutputMap', Golden_Unprocessed_OutputMap)
        print()

        #Sending output to the testbench
        if(testing_Individual_Design_Cases):
            return self.Output
            
        # Debug print
        # print('design_ID_List',design_ID_List)
        # if(len(design_ID_List)>0):
        #     print('============================')
        #     design_ID_item = design_ID_List[0]
        #     design_IDs = [design_ID for design_ID in self.Design_Map if design_ID_item in design_ID]
        #     print('design_IDs',design_IDs)
        #     for design_ID in design_IDs:
        #         DesignBDDgraph = self.Design_Map[design_ID]['processed_graph']
        #         DesignIdItemToWordLineInputMap = self.Design_Map[design_ID]['DesignIdItemToWordLineInputMap']
        #         Crossbar_design          = self.Design_Map[design_ID]['Crossbar_design']
        #         Selector_Lines_Map = self.Design_Map[design_ID]['Selector_Lines_Map']
                
        #         for node in DesignBDDgraph.nodes:
        #             if(design_ID_item in DesignBDDgraph.nodes[node]['in_split_id']):
        #                 print('DesignBDDgraph', DesignBDDgraph.nodes[node]['literal'])
        #         print('DesignIdItemToWordLineInputMap', DesignIdItemToWordLineInputMap[design_ID_item])

        #         for row in Crossbar_design:
        #             print(row)
        #         print(Selector_Lines_Map)
                
        #     print('============================')
            
            
        # send signals to bus
        for design_ID in design_ID_List:
            self.Bus.send_signal(design_ID, signal = True)
            
        # print(design_ID_List)

    def GoldenModel(self, input_assignment, graph=None, LiteralsInputs=None, OutputLine_Map=None):
        
        singleDesignCheck = True
        
        if(graph is None):
            # singleDesignCheck = False
            graph = self.MainBDD_For_GoldenModel
            
            # Start from all nodes with in-degree 0
            start_nodes = [n for n in graph.nodes if graph.in_degree(n) == 0]
            visited_outputs = {graph.nodes[n].get('ExpressionRoot'):0 for n in graph.nodes if graph.nodes[n].get('ExpressionRoot') is not None}
        else:
            # Start from nodes in LiteralsInputs
            start_nodes = [n for n in graph.nodes if graph.nodes[n]['literal'] in LiteralsInputs]
            visited_outputs = {OutputLine_Map[OutputLine_label]:0 for OutputLine_label in OutputLine_Map}
    
        # Helper: decide if a U2 node can be passed
        def is_passable_u2(node):
            literal = graph.nodes[node].get('literal', '')
            if literal.startswith('~') or literal.startswith('-'):
                var = literal[1:]
                return input_assignment.get(var) == 0
            else:
                return input_assignment.get(literal) == 1
    
        
        
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
                if(expression_root is None):
                    expression_root = node_data.get('out_split_id')
                
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
        # Step 1: Extract unique positive variable names from SelectorLineLabels
        input_vars = set()
        SelectorLineLabels = self.SelectorLineLabels.copy()
        for label in SelectorLineLabels:
            var = label.replace('~', '')
            if 'O' not in var:
                input_vars.add(var)
        input_vars = list(input_vars)
        
        test_results = []

        # Step 2: Generate all combinations if RunAllTests is True
        if RunAllTests:
            all_combinations = list(itertools.product([0, 1], repeat=len(input_vars)))
            num_tests = len(all_combinations)
            input_assignments = [
                dict(zip(input_vars, combo)) for combo in all_combinations
            ]
        else:
            input_assignments = [
                {var: random.randint(0, 1) for var in input_vars}
                for _ in range(num_tests)
            ]

        passed_test_case_count = 0 #Keep track of number of test cases passed
        
        # Step 3: Run tests
        for i, input_assignment in enumerate(input_assignments):
            # input_assignment = {var: random.randint(0, 1) for var in input_vars}
            boundary_indices = [0, len(input_assignments) - 1, 5, 39]
            if i not in boundary_indices:
                continue

            self.input_assignment = input_assignment
            
            # Run model and golden model
            print('input_assignment', input_assignment)
            model_output = self.ActivateSelectorLines(input_assignment)
            golden_output = self.GoldenModel(input_assignment)

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
            if(i>=20):
                break

        print(f"Number of test cases passed out of is {passed_test_case_count}/{i+1}")
    
        return test_results

    def RunRandomTestCasesOnEachDesign(self, num_tests=10):

        design_IDs = list(self.Bus.crossbar_designs.keys())
        testNum = 0

        for z, design_ID in enumerate(design_IDs):
            print("Design:",z,design_ID)
            crossbar_design = self.Bus.crossbar_designs[design_ID]["crossbar_design_instance"]
            wordLineInput   = self.Bus.crossbar_designs[design_ID]["wordLineInput"]
            OutputLine_Map  = self.Bus.crossbar_designs[design_ID]["OutputLine_Map"]

            DesignBDDgraph = self.Design_Map[design_ID]['processed_graph']
        
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
                model_output = self.Execute(design_ID, wordLineInput, OutputLine_Map, testing_Individual_Design_Cases=True)  #Run each of those tests in its own designs
        
                # print('model_output', model_output)

                self.input_assignment = input_assignment
                
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