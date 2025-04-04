import numpy as np
import sympy
from sympy import symbols, Eq, linear_eq_to_matrix
import random
import matplotlib.pyplot as plt
from collections import deque, defaultdict
# from pyeda.inter import expr, exprvar, expr2bdd
from dd.autoref import BDD
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

    def connect(self, crossbar_design_instance, design_ID, wordLineInput, OutputLine_Map):
        self.crossbar_designs[design_ID] = {"crossbar_design_instance":crossbar_design_instance, "wordLineInput":wordLineInput, "OutputLine_Map":OutputLine_Map}

    def send_signal(self, design_ID):
        if design_ID in self.crossbar_designs:
            crossbar_design = self.crossbar_designs[design_ID]["crossbar_design_instance"]
            wordLineInput   = self.crossbar_designs[design_ID]["wordLineInput"]
            OutputLine_Map  = self.crossbar_designs[design_ID]["OutputLine_Map"]
            crossbar_design.Execute(wordLineInput, OutputLine_Map)
        else:
            print(f"tried to send to unknown design_ID: {design_ID}")

#Program the crossbar on 1024x1024 designs (Dimentions Custom)
#Have bus serice activated in each design

#Set selector lines

#Execute functions that will run the subtree design which will Give results of that subtree

#Generate Random Test cases for a split bdd
#Verify the input and output in single design of crossbar

#Generate Random Test cases and get the output of testcases from main bdd
#Verify the input and output in full designs of crossbars

#Have a topological plot of the graph (Optional)

class PATH_Design_Logic_Verification:
    def __init__(self, Bus, Design_Map, CrossbarGridSize = 1024):
        self.Design_Map = Design_Map
        self.Bus = Bus

        self.RowOffSet = 0
        self.ColOffSet = 0

        self.SelectorLineLabels = []

        self.Crossbar = [[0 for _ in range(CrossbarGridSize)] for _ in range(CrossbarGridSize)]

        self.Crossbar_Execution = self.Crossbar

        for design_ID, processed_graph_map_value_map in self.Design_Map.items():
            processed_graph    = processed_graph_map_value_map['processed_graph']
            
            Crossbar_design    = processed_graph_map_value_map['Crossbar_design']
            Selector_Lines_Map = processed_graph_map_value_map['Selector_Lines_Map']
            OutputLine_Map     = processed_graph_map_value_map['OutputLine_Map']
            
            LongestPath        = processed_graph_map_value_map['LongestPath']
            
            wordLineInput = self.ProgramCrossbar(Crossbar_design, Selector_Lines_Map)
            
            self.Bus.connect(self, design_ID, wordLineInput, OutputLine_Map)

        self.SelectorLinesOutputLabelsToBitlineIndex = {SelectorLineLabel:index for index, SelectorLineLabel in enumerate(self.SelectorLineLabels) if 'O' in SelectorLineLabel}

    def ProgramCrossbar(self, Crossbar_design, Selector_Lines_Map):
        wordLineInput = self.RowOffSet

        for row_i in range(len(Crossbar_design)):
            for col_j in range(len(Crossbar_design[row_i])):
                if(Crossbar_design[row_i][col_j]==1):
                    self.Crossbar[row_i + self.RowOffSet][col_j + self.ColOffSet] = 1
        
        self.RowOffSet = row_i + self.RowOffSet + 1
        self.ColOffSet = col_j + self.ColOffSet + 1
        
        self.SelectorLineLabels.extend(Selector_Lines_Map)
        return wordLineInput

    def ActivateSelectorLines(self, InputAssignmentMap):

        # Copy the main crossbar design to execution crossbar to run executions
        self.Crossbar_Execution = [row.copy() for row in self.Crossbar]

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
        for col_j, selector_line in enumerate(selector_lines):
            if(not selector_line):
                for row_i in range(len(self.Crossbar_Execution)):
                    self.Crossbar_Execution[row_i][col_j] = 2

        #Create a Output dictionary for storing the output result
        self.Output = {}
        
        #Sending signal to run the first crossbar after setting selector lines in programed crossbar
        self.Bus.send_signal(None)

        print('self.Output', self.Output)

    def TimeMultiplexCrossbar(self, Crossbar_, nonOutputBitlines):
        Crossbar = [row.copy() for row in Crossbar_]
        for nonOutputBitline_index in nonOutputBitlines:
            for row_i in range(len(Crossbar)):
                Crossbar[row_i][nonOutputBitline_index] = 2
        return Crossbar

    def find_path_execution_in_crossbar(self, Crossbar, wordLineInput, outputBitline):
        # Custom function to emulate an ordered set using a list
        def add_to_ordered_set(ordered_set, element):
            if element not in ordered_set:
                ordered_set.append(element)

        R_LRS = 1
        
        # Stack for depth-first traversal
        Stack = []
        
        # Finding paths
        for j in range(len(Crossbar[wordLineInput])):
            if Crossbar[wordLineInput][j] == R_LRS:
                Stack.append([(wordLineInput, j), [(wordLineInput, j)], 'w'])  # Use a list for ordered visited nodes

        # print('Crossbar[0]',Crossbar[0])
        print('Stack',Stack)
        # print('outputBitline',outputBitline)
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
            if(path_j==outputBitline):
                print('outputBitline',outputBitline)
                print('visited',visited)
                return True
        return False
        
    def Execute(self, wordLineInput, OutputLine_Map):

        print('wordLineInput', wordLineInput)
        print('OutputLine_Map', OutputLine_Map)

        design_ID_List = []
        for outputLine in OutputLine_Map:
            outputBitline = self.SelectorLinesOutputLabelsToBitlineIndex[outputLine]

            print('outputBitline',outputBitline, OutputLine_Map[outputLine])

            nonOutputBitlines = [self.SelectorLinesOutputLabelsToBitlineIndex[OutputLineLabel] for OutputLineLabel in OutputLine_Map if outputBitline!=self.SelectorLinesOutputLabelsToBitlineIndex[OutputLineLabel]]

            # print('nonOutputBitlines', nonOutputBitlines)
            MultiplexedCrossbar = self.TimeMultiplexCrossbar(self.Crossbar_Execution, nonOutputBitlines)

            # self.VisuvaliseCrossbar(MultiplexedCrossbar)
            
            # code to execute crossbar
            foundPath = self.find_path_execution_in_crossbar(MultiplexedCrossbar, wordLineInput, outputBitline)
            print('foundPath',foundPath)
            if(foundPath):
                if(OutputLine_Map[outputLine].split()[0]=="leaf"):
                    design_ID_List.append(OutputLine_Map[outputLine].split()[1])
                else:
                    self.Output[OutputLine_Map[outputLine]] = 1
            else:
                if(OutputLine_Map[outputLine].split()[0]=="leaf"):
                    pass
                else:
                    self.Output[OutputLine_Map[outputLine]] = 0

        print('design_ID_List',design_ID_List)
        # send signals to bus
        for design_ID in design_ID_List:
            self.Bus.send_signal(design_ID)
        
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

    def IdealCurrentPath(self, literal_value_map):
        
        # literal_value_map = {'a':0, 'b':0, 'cin':1}
        # print('literal_value_map',literal_value_map)
        
        #bitline labels according to crossbar
        # self.Bitlines = ['a', '~a', '~a', 'a', 'b', '~b', '~b', 'b', 'b', '~cin', 'cin', 'O14', 'O15']
        # self.OutputLine_Map = {'O14': 'cout', 'O15': 'sum0'}
        # self.End_Bitline_Output_Line = 'O16'

        # print('literal_value_map', literal_value_map)
        bitlines = self.Bitlines

        R_LRS = 1
        R_HRS = 0
        R_Off = 2

        IdealOutputs = {}
        IdealPathsOfCurrents = []

        OutputPaths = {}

        End_Bitline_Output_Cell = None
        #finding output paths
        for col,bitline in enumerate(bitlines):
            if(self.End_Bitline_Output_Line == bitline):
                for row in range(len(self.CrossbarResistiveStates)-1,-1,-1):
                    if(self.CrossbarResistiveStates[row][col]==R_LRS):
                        End_Bitline_Output_Cell = (row, col)
            elif('O' in bitline):
                OutputPaths[self.OutputLine_Map[bitline]] = []

        # print('End_Bitline_Output_Cell',End_Bitline_Output_Cell)
        # print('OutputLine_Map',self.OutputLine_Map)
        
        for clockCycle, outputLineLabel in enumerate(self.OutputLine_Map):
            Resistance_matrix = [row.copy() for row in self.CrossbarResistiveStates]
            SelectorLines = [0 for _ in range(len(Resistance_matrix[0]))]
            for i,bitline in enumerate(bitlines):
                literal = bitline
                negation = literal.startswith('~')
                if(negation):
                    literal = literal[1:]
    
                if(outputLineLabel == literal):
                    SelectorLines[i] = 1
                elif(self.End_Bitline_Output_Line == literal):
                    SelectorLines[i] = 1
                elif(literal in literal_value_map and literal_value_map[literal]==0):
                    if(negation):
                        SelectorLines[i] = 1
                    else:
                        SelectorLines[i] = 0
                elif(literal in literal_value_map and literal_value_map[literal]==1):
                    if(negation):
                        SelectorLines[i] = 0
                    else:
                        SelectorLines[i] = 1
    
            # print('outputLineLabel',outputLineLabel)
            # print('SelectorLines',SelectorLines)
            
            #Swiching off the bitlines
            for col, SelectorLine in enumerate(SelectorLines):
                if(SelectorLine==0):
                    for row in range(len(Resistance_matrix)):
                        Resistance_matrix[row][col] = R_Off
    
            # Custom function to emulate an ordered set using a list
            def add_to_ordered_set(ordered_set, element):
                if element not in ordered_set:
                    ordered_set.append(element)

            literal_value_map_lst = [{'a0': 1, 'b0': 1, 'cin': 0},{'a0': 0, 'b0': 1, 'cin': 0},{'a0': 0, 'b0': 1, 'cin': 1}]
                    
            # Stack for depth-first traversal
            Stack = []
    
            # Finding paths
            for j in range(len(Resistance_matrix[0])):
                if Resistance_matrix[0][j] == R_LRS:
                    Stack.append([(0, j), [(0, j)], 'w'])  # Use a list for ordered visited nodes

            # print('Resistance_matrix[0]',Resistance_matrix[0])
            # print('Stack',Stack)
            while Stack:
                [(path_i, path_j), visited, last_curr] = Stack.pop()  # Pop from the stack (LIFO)
                for i in range(len(Resistance_matrix)):
                    if last_curr == 'w':
                        if Resistance_matrix[i][path_j] == R_LRS and (i, path_j) not in visited:
                            new_visited = visited.copy()
                            add_to_ordered_set(new_visited, (i, path_j))
                            Stack.append([(i, path_j), new_visited, 'b'])
                    elif last_curr == 'b':
                        if Resistance_matrix[path_i][i] == R_LRS and (path_i, i) not in visited:
                            new_visited = visited.copy()
                            add_to_ordered_set(new_visited, (path_i, i))
                            Stack.append([(path_i, i), new_visited, 'w'])
                if((path_i, path_j) == End_Bitline_Output_Cell):
                    OutputPaths[self.OutputLine_Map[outputLineLabel]] = visited

            # print('OutputPaths2',OutputPaths)

            #display paths
            label = self.OutputLine_Map[outputLineLabel]
            temp_path = OutputPaths[label]
            if(len(temp_path)>0):
                IdealPathsOfCurrents.append({"literal_value_map":literal_value_map, "label":label, "OutputPath": temp_path, "lengthOfDevices":len(temp_path)})
                IdealOutputs[label]=1
                # print(f"{label}=1")
            else:
                IdealOutputs[label]=0
                # print(f"{label}=0")
                if(literal_value_map in literal_value_map_lst):
                    print('literal_value_map',literal_value_map)
                    print(f"{label}=0")
                    # self.VisuvaliseCrossbar(initialisedCrossbar=Resistance_matrix)
        
        return IdealOutputs, IdealPathsOfCurrents
        
    def Verify_All_Ideal_Paths_In_Crossbar(self, checkWithOriginal=False, checkWithBDD=False):
        """
        Iterates through all rows of the truth table, inputs values into 
        the IdealCurrentPath function, and compares outputs.
        """
        dfs = []
        if(checkWithOriginal):
            # Ensure the truth table is generated
            if not hasattr(self, "TruthTable") or self.OriginalTruthTable is None:
                print("Error: TruthTable not generated. Call GetTruthTables() first.")
                return
            dfs.append(self.OriginalTruthTable)
        if(checkWithBDD):
            # Ensure the truth table is generated
            if not hasattr(self, "TruthTable") or self.BDDTruthTable is None:
                print("Error: TruthTable not generated. Call GetTruthTables() first.")
                return
            dfs.append(self.BDDTruthTable)
    
        print("Verifying all ideal paths in the crossbar...")

        for df in dfs:

            mismatches = 0  # Track number of mismatches
    
            input_columns = [col for col in df.columns if col not in self.outputs]  # Variables
            output_columns = [col for col in df.columns if col in self.outputs]  # Expressions (functions)

            # print('df.columns',df.columns)
            # print('input_columns',input_columns)
            # print('output_columns',output_columns)
    
            self.AllIdealPathOfCurrent = {}
            
            # Iterate over each row in the truth table
            for idx, row in df.iterrows():
                # if(idx%1000!=0):
                #     continue
                # print(idx)
                input_assignment = {var: int(row[var]) for var in input_columns}  # Convert inputs to dictionary
                expected_outputs = {expr: int(row[expr]) for expr in output_columns}  # Expected output values
    
                # Compute output using IdealCurrentPath function
                computed_outputs, IdealPathsOfCurrents = self.IdealCurrentPath(input_assignment)

                # Compare computed outputs with expected outputs
                for expr in output_columns:
                    if computed_outputs.get(expr) != expected_outputs[expr]:
                        print(f"Mismatch at row {idx}: Inputs {input_assignment}, "
                              f"Expected {expected_outputs}, Got {computed_outputs}")
                        
                        mismatches += 1
                        
                totlengthTemp = []
    
                # print('input_assignment', idx, input_assignment)   #debugprint
                
                maxLengthOfDevices = 0
                for IdealPathsOfCurrent in IdealPathsOfCurrents:
                    literal_value_map = IdealPathsOfCurrent["literal_value_map"]
                    label = IdealPathsOfCurrent["label"]
                    OutputPath = IdealPathsOfCurrent["OutputPath"]
                    lengthOfDevices = IdealPathsOfCurrent["lengthOfDevices"]
    
                    if(frozenset(literal_value_map.items()) not in self.AllIdealPathOfCurrent):
                        self.AllIdealPathOfCurrent[frozenset(literal_value_map.items())] = {}
                    self.AllIdealPathOfCurrent[frozenset(literal_value_map.items())][label]  = {"OutputPath":OutputPath, "lengthOfDevices":lengthOfDevices}
                    # self.AllIdealPathOfCurrent[frozenset(literal_value_map.items())][label]  = {"lengthOfDevices":lengthOfDevices}
    
                    totlengthTemp.append(lengthOfDevices)
                    # print(label, OutputPath)
    
                # print('sum totlengthTemp',sum(totlengthTemp))
                # if(len(totlengthTemp)):
                #     print('avg totlengthTemp',sum(totlengthTemp)/len(totlengthTemp))
                # print('expected_outputs',idx, expected_outputs)
                # print('computed_outputs', computed_outputs)
                
            print(f"Crossbar verification completed. Total mismatches: {mismatches}")
            
            return mismatches