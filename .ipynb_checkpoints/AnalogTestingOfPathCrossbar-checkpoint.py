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
import math

def four_sig_truncate(num):
    """
    Truncates a positive float 'num' to exactly 4 significant digits.
    For example:
      1.357371568 -> '1.357'
      4.69340734797519e-6 -> '4.693e-06'
    """
    if num == 0:
        return "0"  # edge case
    
    # Get sign and make num positive if it isn't
    sign = "-" if num < 0 else ""
    num = abs(num)
    
    # Find base-10 exponent (e.g., 1.357... => exp=0, 4.693e-6 => exp ~ -5, etc.)
    exp = math.floor(math.log10(num))
    
    # Scale to get leading digit(s) (mantissa >= 1 and < 10)
    # e.g., for 1.357..., mantissa ~ 1.357
    # for 4.693e-6, mantissa ~ 4.693
    mantissa = num / (10**exp)
    
    # We want 4 significant digits total.
    # Multiply by 10^(4 - 1) = 10^3 = 1000. Then truncate.
    # This will keep exactly 4 digits before the decimal, effectively.
    mantissa_scaled = int(mantissa * 1000)  # truncation, no rounding
    mantissa_trunc = mantissa_scaled / 1000.0
    
    # If mantissa_trunc became >= 10 (very edge case), adjust
    if mantissa_trunc >= 10:
        mantissa_trunc /= 10
        exp += 1
    
    # If exponent is in [–3, 3], we might prefer normal notation (like '1.234') 
    # but to stick to a consistent style, let's always use e-notation if exp != 0:
    if exp == 0:
        # Just attach the sign
        return f"{sign}{mantissa_trunc}"
    else:
        return f"{sign}{mantissa_trunc}e{exp:+d}"

# Example usage
val1 = 1.3573715686798096
val2 = 4.69340734797519e-6

class AnalogTestingOfPathCrossbar:
    def __init__(self, Crossbar, Bitlines, TruthTable, OutputLine_Map, AllIdealPathOfCurrent=None, CrossbarLongPaths=None, WorstCaseCrossbarPaths=None, End_Bitline_Output_Line=None):
        self.Crossbar = Crossbar
        self.Bitlines = Bitlines
        self.End_Bitline_Output_Index = Bitlines.index(End_Bitline_Output_Line)
        self.TruthTable = TruthTable
        self.OutputLabelsToBitlineIndexMap = {OutputLine_Map[key]:Bitlines.index(key) for key in OutputLine_Map}  #imp
        self.OutputBitlineIndexMapToLabels = {Bitlines.index(key):OutputLine_Map[key] for key in OutputLine_Map}
        self.AllIdealPathOfCurrent = AllIdealPathOfCurrent
        self.AdderNumDevicesToCurrents_Map = {}
        
        self.CrossbarLongPaths = CrossbarLongPaths
        self.WorstCaseCrossbarPaths = WorstCaseCrossbarPaths
        
        self.tempMatrix = None

    def initialiseCrossbar(self, input_assignment={}, outputSelectorLine=None):
        CrossbarCopy = [row.copy() for row in self.Crossbar]
        # self.Bitline

        selectorLinesDeactive = []
        selectorLinesActive = {i for i in range(len(CrossbarCopy))}
        
        for col_j, bitline in enumerate(self.Bitlines):
            literal = bitline
            
            negation = True if literal[0]=='~' else False
            if(negation):
                literal=literal[1:]
            
            flag=True
            if('O' in literal):
                if(outputSelectorLine==col_j):
                    flag=False
                elif(self.End_Bitline_Output_Index==col_j):
                    flag=False
            elif(literal in input_assignment and input_assignment[literal]==0):
                if(negation):
                    flag=False
            elif(literal in input_assignment and input_assignment[literal]==1):
                if(not negation):
                    flag=False
            
            if(flag):
                selectorLinesActive.remove(col_j)
                for row_i in range(len(CrossbarCopy)):
                    CrossbarCopy[row_i][col_j] = 2
                selectorLinesDeactive.append(col_j)
        print('input_assignment',input_assignment, 'outputSelectorLine',outputSelectorLine)
        print('selectorLinesActive',selectorLinesActive)
            
        return CrossbarCopy
    
    def SufficientCase_initialiseCrossbar(self, output, output_col=None, Path=None):
        if(output==0):  # 0 -> outpulines  #nithin
            R_LRS = 1
            R_off = 2
            CrossbarCopy = [row.copy() for row in self.Crossbar]

            active_selector_lines = {i for i in range(len(CrossbarCopy[0]))}
            deactive_selector_lines = set()

            #make other selectors lines off other than output_col and end_output_col
            for col_j, bitline in enumerate(self.Bitlines):
                if(col_j!=self.End_Bitline_Output_Index and col_j!=output_col and 'O' in bitline):
                    active_selector_lines.remove(col_j)
                    deactive_selector_lines.add(col_j)

            #Get the end cell of the outputline
            end_cell = None
            for row_i in range(len(CrossbarCopy)):
                if(CrossbarCopy[row_i][output_col] == R_LRS):
                    end_cell = (row_i, output_col)
                    break
            
            # Queue holds states of the form ((row,col), mode)
            # mode = 'w' or 'b', as in your existing logic
            queue = deque()
            queue.append((end_cell, 'b'))
                         
            visited = set()     # to mark ((row,col), mode) as visited
            visited.add(end_cell)
            print('end_cell',end_cell)

            children = {}

            # BFS
            found = False
            end_state = False
        
            while queue:
                (row, col), mode = queue.popleft()

                # print('row,col,mode',row,col,mode)
                # print('visited', visited)
                # Check if we've reached the end cell
                if row==0:
                    found = True     
                    end_state = ((row, col), mode)
                    break
        
                if mode == 'w':
                    # Move "across" the row
                    for r in range(len(CrossbarCopy)):
                        if (CrossbarCopy[r][col] == R_LRS):
                            next_state = ((r, col), 'b')
                            if((row, col) not in children):
                                children[(row, col)] = []
                            children[(row, col)].append(next_state[0])
                            if next_state[0] not in visited:
                                visited.add(next_state[0])
                                queue.append(next_state)
                
                elif mode=='b': # mode == 'b'
                    # Move "up" or "down" the column
                    for c in range(len(CrossbarCopy[row])):
                        if (CrossbarCopy[row][c] == R_LRS):
                            next_state = ((row, c), 'w')
                            if((row, col) not in children):
                                children[(row, col)] = []
                            children[(row, col)].append(next_state[0])
                            if next_state[0] not in visited:
                                visited.add(next_state[0])
                                queue.append(next_state)
                # if((row, col) in children):
                #     print('children[(row, col)]1', children[(row, col)])

            # print('found',found)
            # print('children', children)

            ############################################################

            print('deactive_selector_lines1',deactive_selector_lines)

            flag=True
            for row_i,col_j in children[end_cell]:
                if(col_j != end_cell[1]):
                    print('row_i, col_j',row_i, col_j)
                    print('children[(row_i,col_j)]',children[(row_i,col_j)])
                    for child_i, child_j in children[(row_i,col_j)]:
                        if((child_i, child_j)!=(row_i, col_j)):
                            CrossbarCopy[child_i][child_j] = 0
                        elif(flag):
                            CrossbarCopy[child_i][child_j] = 0
                            flag=False
                    
            print('deactive_selector_lines2',deactive_selector_lines)
            
            
            for col_j in deactive_selector_lines:
                # print('self.Bitlines[col_j]', self.Bitlines[col_j])
                if(col_j in active_selector_lines):
                    active_selector_lines.remove(col_j)
                for row_i in range(len(CrossbarCopy)):
                    CrossbarCopy[row_i][col_j] = 2

            #deactivate remaining rows
            for col_j in range(self.End_Bitline_Output_Index+1, len(CrossbarCopy[0])):
                # print('self.Bitlines[col_j]', self.Bitlines[col_j])
                active_selector_lines.remove(col_j)
                for row_i in range(len(CrossbarCopy)):
                    CrossbarCopy[row_i][col_j] = 2

            print('active_selector_lines',active_selector_lines)
                
            
        if(output==1):  # 1 -> Paths
            CrossbarCopy = [[0 for _ in row] for row in self.Crossbar]
            selectorLines = set()
            if(Path):
                for cell in Path:
                    CrossbarCopy[cell[0]][cell[1]]=1
                    selectorLines.add(cell[1])
                for col_j in range(len(CrossbarCopy[0])):
                    if(col_j not in selectorLines):
                        for row_i in range(len(CrossbarCopy)):
                            CrossbarCopy[row_i][col_j]=2
        return CrossbarCopy

    def NodalAnalysis(self, Initialized_Crossbar, variable_HRS):
        """
        Performs a nodal (KCL) analysis on the given Initialized_Crossbar
        using variable_HRS as the high-resistance state (HRS).
    
        Returns:
            Output_current_map (dict): Maps each output line label to its computed current.
        """
        # for row in Initialized_Crossbar:
        #     print(row)
        # Define resistor parameters
        R_Off = 4e9  # Very large (transistor off)
        R_HRS = variable_HRS  # High-resistance state of the memory cell
        R_LRS = 2000  # Low-resistance state
        R_Line_Out = 200  # 200 ohms from each column node to GND
        R_Not = 1e10  # Large resistance for non-output columns
        R_source   = 100  # Series resistor from 0.2 V source to row 0
        
        # Voltage source for the first row (0.2V)
        Vsrc = 0.2  
    
        # -----------------------------
        # 2) Prepare the crossbar data
        # -----------------------------
        crossbar_size = len(Initialized_Crossbar)
        Initialized_Crossbar = np.array(Initialized_Crossbar)
    
        # Build the final resistance matrix:
        #    0 => R_HRS
        #    1 => R_LRS
        #    2 => R_Off
        Resistance_matrix = np.where(Initialized_Crossbar == 0, R_HRS, 
                                    np.where(Initialized_Crossbar == 1, R_LRS, 
                                             np.where(Initialized_Crossbar == 2, R_Off, Initialized_Crossbar)))
    
        # -----------------------------
        # 3) Construct the KCL system
        #    We have #rows + #columns unknowns:
        #       Vr0, Vr1, ..., Vr(N-1), Vc0, Vc1, ..., Vc(N-1)
        # -----------------------------
        num_vars = 2 * crossbar_size  
        A = np.zeros((num_vars, num_vars))  # Coefficient matrix
        b = np.zeros(num_vars)  # Constant vector
    
        # Define row equations (KCL)
        # for i in range(crossbar_size):
        #     if i == 0:
        #         # First row is directly connected to the voltage source
        #         A[i, i] = 1
        #         b[i] = Vsrc
        #     else:
        #         for j in range(crossbar_size):
        #             Rij = Resistance_matrix[i][j]
        #             A[i, i] += 1 / Rij  # Self term
        #             A[i, crossbar_size + j] -= 1 / Rij  # Interaction with column

        # -----------------------------
        # 3a) Row equations (KCL at each row node)
        # -----------------------------
        for i in range(crossbar_size):
            if i == 0:
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
            if j == self.End_Bitline_Output_Index:
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
        CurrentOutput = Vc[self.End_Bitline_Output_Index] / R_Line_Out

        # print('Vr', Vr)
        # print('Vc', Vc)

        # print('Output_current_map',Output_current_map)

        # print('Vr',Vr)
        # print('Vc',Vc)
        # print('---------')
    
        return CurrentOutput
        
    def Iteration_Of_Each_Row_In_the_TruthTable(self, variable_HRS):

        output_cols = set(self.OutputLabelsToBitlineIndexMap.keys())
        input_columns = [col for col in self.TruthTable.columns if col not in output_cols]  # Variables
        output_columns = [col for col in self.TruthTable.columns if col in output_cols]  # Expressions (functions)

        # self.OutputLabelsToBitlineIndexMap = {'cout': 11, 'sum0': 12}

        zerosCurrents, onesCurrents = [], []
        # Iterate over each row in the truth table
        for idx, row in self.TruthTable.iterrows():
            input_assignment = {var: int(row[var]) for var in input_columns}  # Convert inputs to dictionary {'a0':0, 'b0':0, 'cin':1}
            expected_outputs = {expr: int(row[expr]) for expr in output_columns}  # Expected output values

            start_time_iteration = time.time()

            currentOutputs = {}
            for label in self.OutputLabelsToBitlineIndexMap:
                outputSelectorLine = self.OutputLabelsToBitlineIndexMap[label]
            
                #setting selector lines -> crossbar matrix
                Initialized_Crossbar = self.initialiseCrossbar(input_assignment, outputSelectorLine)
                
                #running nodal analysis -> for each output give current in output line
                currentOutput = self.NodalAnalysis(Initialized_Crossbar, variable_HRS)

                currentOutputs[label] = currentOutput

            print("Duration:", four_sig_truncate(time.time()-start_time_iteration),"sec")

            # print('input_assignment',idx, input_assignment)
            
            # check min and max curretns at expected_outputs using this AnalogPathTest.OutputLabelsToBitlineIndexMap
            for label in expected_outputs:
                outputCurrent = currentOutputs[label]
                if(expected_outputs[label]==0):
                    print('000000', label, outputCurrent)
                    zerosCurrents.append(outputCurrent)
                elif(expected_outputs[label]==1):
                    print('111111', label, outputCurrent)
                    onesCurrents.append(outputCurrent)
                    numberofDevices = self.AllIdealPathOfCurrent[frozenset(input_assignment.items())][label]["lengthOfDevices"]
                    if(numberofDevices not in self.AdderNumDevicesToCurrents_Map):
                        self.AdderNumDevicesToCurrents_Map[numberofDevices] = []
                    self.AdderNumDevicesToCurrents_Map[numberofDevices].append(outputCurrent)
            # print('expected_outputs',idx,expected_outputs)

        guardBand = min(onesCurrents) - max(zerosCurrents)

        print('min(onesCurrents)1',min(onesCurrents))
        print('max(zerosCurrents)1',max(zerosCurrents))
        # print('guardBand1',guardBand)?\
        
        return {"guardBand":guardBand, "zerosCurrents":zerosCurrents, "onesCurrents":onesCurrents}

    def get_expected_output(self, input_assignment):
        """
        Given an input_assignment dict like:
            {'a0': 0, 'b0': 1, 'cin': 0}
        or:
            {'a0': 0, 'a1': 0, 'b0': 0, 'b1': 1, 'cin': 0}
        etc.
        Return a dict of the form:
            {'sum0': ..., 'sum1': ..., ..., 'cout': ...}
        representing the bitwise sum of a and b, plus cin.
        """
        # Gather all bit indices for a and b.
        a_indices = [int(k[1:]) for k in input_assignment if k.startswith('a')]
        b_indices = [int(k[1:]) for k in input_assignment if k.startswith('b')]
    
        # If there are no bits for a or b (unlikely in practice), just return something empty.
        if not a_indices and not b_indices:
            return {'sum0': 0, 'cout': input_assignment.get('cin', 0)}
    
        max_index = max(a_indices + b_indices)  # highest bit position found
    
        # Convert the bits from the dictionary into integer form for 'a' and 'b'.
        a_val = 0
        b_val = 0
        for i in range(max_index + 1):
            a_bit = input_assignment.get(f'a{i}', 0)
            b_bit = input_assignment.get(f'b{i}', 0)
            # Shift bit i into the correct position
            a_val |= (a_bit << i)
            b_val |= (b_bit << i)
    
        # Get cin (carry in), defaulting to 0 if not present
        cin = input_assignment.get('cin', 0)
    
        # Perform the addition
        total = a_val + b_val + cin
    
        # Convert the sum back into individual bits
        result = {}
        for i in range(max_index + 1):
            result[f'sum{i}'] = (total >> i) & 1
    
        # Extract the carry-out (this is the bit above the highest position)
        cout = (total >> (max_index + 1)) & 1
        result['cout'] = cout
    
        return result

    def Iteration_Of_sufficient_Rows_In_the_TruthTable(self, variable_HRS):
        output_cols = list(self.OutputLabelsToBitlineIndexMap.values())

        zerosCurrents, onesCurrents = [], []

        # print("max I in zero")
        #worst case maximum current in zero
        # print('output_cols',output_cols)

        R_LRS = 1

        outputSelectorLines = []
        for col_j, bitline in enumerate(self.Bitlines):
            if('O' in bitline and col_j!=self.End_Bitline_Output_Index):
                for row_i in range(len(self.Crossbar)):
                    if(self.Crossbar[row_i][col_j] == R_LRS):
                        outputSelectorLines.append(col_j)
                        break

        for outputSelectorLine in outputSelectorLines:
                            
            Initialized_Crossbar = self.SufficientCase_initialiseCrossbar(output=0, output_col=outputSelectorLine)
            
            maxZerocurrent = self.NodalAnalysis(Initialized_Crossbar, variable_HRS)
            # print(self.Bitlines[outputSelectorLine],maxZerocurrent)
            zerosCurrents.append(maxZerocurrent)
        
        #########################################
        
        # print("min I in one")#niti
        #worst case minimum current in one
        
        for keymap in self.CrossbarLongPaths:
            literal_CrossbarLongPaths = self.CrossbarLongPaths[keymap]
            longPath = literal_CrossbarLongPaths
            break

        keymap = list(keymap)
        keymap.sort(key= lambda x:x[0])
        # print('key',{key:value for key,value in keymap})
        # longPath = [(0, 9), (2, 9), (2, 5), (6, 5), (6, 3), (8, 3), (8, 12), (9, 12), (9, 13)]
        # longPath = [(0, 9), (2, 9), (2, 5), (6, 5), (6, 3), (8, 3), (8, 12), (9, 12), (9, 13), 
        #             (1,0), (4,0), (1, 6), (0,6)
        #            ]
        # print('longPath',longPath)
        #setting selector lines -> crossbar matrix
        Initialized_Crossbar = self.SufficientCase_initialiseCrossbar(Path=longPath, output=1)
        #running nodal analysis -> current in each bitlines
        minOneCurrent = self.NodalAnalysis(Initialized_Crossbar, variable_HRS)
        onesCurrents.append(minOneCurrent)
        
        #------------------------------------------------------
        
        guardBand = min(onesCurrents) - max(zerosCurrents)

        # print('min(onesCurrents)2',min(onesCurrents))
        # print('max(zerosCurrents)2',max(zerosCurrents))
        # print('guardBand2',guardBand)
        
        return {"guardBand":guardBand, "zerosCurrents":zerosCurrents, "onesCurrents":onesCurrents}
        

    def Finding_optimal_HRS_using_binarysearch(self, high_variable_HRS=2e9, iterations=2, optimisation=False):
        """
        Perform a binary search to find an optimal high-resistance value (HRS)
        that yields a positive guardValue returned by 'Iteration_Of_Each_Row_In_the_TruthTable'.
        
        Params:
            high_variable_HRS (float): Upper bound for HRS (defaults to 2e9).
            iterations (int): Number of binary search iterations.
    
        Returns:
            float: The last HRS value that produced a positive guardValue, if any.
        """
        lower_bound = 0.0
        upper_bound = high_variable_HRS
    
        guard_values = []
        resistance_values = []
        zerosCurrents_lst = []
        onesCurrents_lst = []
    
        for i in range(iterations):
            # Midpoint of the current lower and upper bounds
            resistance_value = (lower_bound + upper_bound) / 2

            print('resistance_value',resistance_value, i)

            start_time_all_iterations = time.time()

            if(optimisation):
                guardbandMap = self.Iteration_Of_sufficient_Rows_In_the_TruthTable(resistance_value)
            else:
                guardbandMap = self.Iteration_Of_Each_Row_In_the_TruthTable(resistance_value)

            guardband, zerosCurrents, onesCurrents = guardbandMap['guardBand'], guardbandMap['zerosCurrents'], guardbandMap['onesCurrents']
            print('guardBand',guardband, 'Total Duration =',four_sig_truncate(time.time()-start_time_all_iterations),"sec")
            # Decide which way to move in the binary search
            if guardband > 0:
                # If guardband is positive, it means we can try smaller high-resistance
                upper_bound = resistance_value
                guard_values.append(guardband)
                resistance_values.append(resistance_value)
                zerosCurrents_lst.append(zerosCurrents)
                onesCurrents_lst.append(onesCurrents)
            else:
                # Otherwise, we need to go higher
                lower_bound = resistance_value

        return {"guardbands":guard_values, "resistance_values":resistance_values, "zerosCurrents_lst":zerosCurrents_lst, "onesCurrents_lst":onesCurrents_lst, "minOne":min(onesCurrents), "maxZero":max(zerosCurrents)}