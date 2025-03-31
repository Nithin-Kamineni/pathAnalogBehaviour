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



class PATH:
    def __init__(self, CrossbarGridSize = 16):
        self.CrossbarGridSize = CrossbarGridSize
        self.Crossbar_sizes = [2**i for i in range(3,16)]

        self.pre_bool_expressions = None
        self.pre_varibles_lst = None
        self.nodeID_map = None

        self.filename = None
        self.model_name = None
        self.inputs = []
        self.outputs = []

        self.OriginalTruthTable = None
        self.TruthTable = None
        self.BDDTruthTable = None
        
        self.BDD = None
        self.Graph = None
        self.Expressions = None
        self.NodeIDMap = None
        self.InputNode = None
        self.GraphProcessPhase = None
        self.MainNodesCounter = 0
        
        self.CrossbarResistiveStates = None
        self.Bitlines = None
        self.SBDD_dimentions = None
        self.End_Bitline_Output_Line = None

        self.BDDlongPaths = None
        self.CrossbarLongPaths = None

        self.TreeMapInNodes = {}
        self.xlabelsOfLiterals_map = {}
        self.OutputLine_Map = {}
        self.AllIdealPathOfCurrent = {}
        self.CountOfLongestPath = 0

        self.leaf_count = {}
        self.distance_dict = {}
        self.last_split_node = None
        self.prefix_path = []

    def parse_file_to_NetworkXGraph(self, filename):
        """ Reads the file and extracts nodes, variables, and outputs. """
        self.filename = filename
        with open(self.filename, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]

        unprocessed_lines = []
        for line in lines:
            if line.startswith('.model'):
                self.model_name = line.split()[1:]
            elif line.startswith('.inputs'):
                self.inputs = line.replace(".inputs", "").replace(";", "").strip().split()
            elif line.startswith('.outputs'):
                self.outputs = line.replace(".outputs", "").replace(";", "").strip().split()
            elif line.startswith('.bdd') or line.startswith('.order') or line.startswith('.inputs') or line.startswith('.outputs'):
                continue  # Ignore section marker
            elif line.startswith('.end'):
                break  # Stop parsing at .end
            else:
                unprocessed_lines.append(line)
        self._parse_bdd_lines(unprocessed_lines)

    def _parse_bdd_lines(self, lines):
        """ Parses a BDD node definition line and assigns node colors. """
        outputLiteral = {}
        self.NodeIDMap = {}
        self.Expressions = {}
        self.TreeMapInNodes = {}
        
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) == 5:
                node_id, high_child_id, low_child_id, var, output = parts  # Root node
            elif len(parts) == 4:
                node_id, high_child_id, low_child_id, var = parts
                output = None  # Some nodes may not specify an explicit output
            else:
                return  # Skip malformed lines
    
            # Convert to integers where applicable
            node_id = int(node_id, 16) if re.match(r'^[0-9a-fA-F]+$', node_id) else int(node_id)
            low_child_id = int(low_child_id, 16) if low_child_id != "-1" else -1
            high_child_id = int(high_child_id, 16) if high_child_id != "-1" else -1
            
            if(var=='0'):
                outputLiteral[node_id] = var
            elif(var=='1'):
                outputLiteral[node_id] = var
            else:
                # Store node structure in TreeMapInNodes
                self.TreeMapInNodes[str(node_id)] = {
                    "variable": var,
                    "low": low_child_id,
                    "high": high_child_id,
                    "negation": False,  # Assuming no negation flag in file
                }
    
            # Store reference count in NodeIDMap
            if node_id not in self.NodeIDMap:
                self.NodeIDMap[str(node_id)] = [0, var]
            self.NodeIDMap[str(node_id)][0] += 1
    
            # Track root nodes per output variable
            if output:
                self.Expressions[output] = str(node_id)

    def TruthTable_to_BooleanExpression(self, df, outputs):

        # Need to validate
        self.OriginalTruthTable = df
        self.outputs = outputs

        def get_minterm_index(row):
            minterm_str = ""
            for col in df.columns:
                # if("F" not in col):
                if(col not in outputs):
                    minterm_str = minterm_str + row[col]
            return int(minterm_str, 2)

        expressions = {}
        # variables = [symbols(col) for col in df.columns if 'F' not in col]
        variables = [symbols(col) for col in df.columns if col not in outputs]
        varibles_str_lst = []

        for col in df.columns:
            # if('F' in col):
            if(col in outputs):
                minterms_col = [get_minterm_index(row) for idx, row in df.iterrows() if row[col] == '1']
                expressions[col] = str(SOPform(variables, minterms_col))
            else:
                varibles_str_lst.append(col)

        self.pre_varibles_lst = varibles_str_lst
        self.pre_bool_expressions = expressions

        for expression in expressions:
            print(expression,':',expressions[expression])

    def SetBooleanExpressionsAndVaribles(self, variables=None, expressions=None, outputs=None, OriginalTruthTable=None):
        if(variables!=None):
            self.pre_varibles_lst = variables
        if(expressions!=None):
            self.pre_bool_expressions = expressions
        if(outputs!=None):
            self.outputs = outputs
        if OriginalTruthTable is not None:
            self.OriginalTruthTable = OriginalTruthTable
        
    def BooleanExpresions_to_BDD(self):
        variables = self.pre_varibles_lst    
        expressions = self.pre_bool_expressions
        
        # Initialize the BDD manager
        self.BDD = BDD()
        # self.BDD.configure(reordering=True)
        
        # Declare variables
        self.BDD.declare(*variables)
        # self.BDD.declare('a', 'b', 'c', 'd')
        # self.BDD.declare('a0', 'b0', 'cin',)
        
        # Define Boolean expressions
        self.expressions = expressions
        # self.expressions = {
        #     'P0': 'a & b',
        #     'P1': '(a & b) ^ (c & d)',
        #     'P2': '(a & c) ^ ((a & b) & (c & d))',
        #     'P3': '(a & c) & ((a & b) & (c & d))'
        # }
        
        # self.expressions = {'c_out': '(a0 & b0) | (a0 & cin) | (b0 & cin)', 's_0': 'a0 ^ b0 ^ cin'}
        self.Expressions = {expression: self.BDD.add_expr(self.expressions[expression]) for expression in self.expressions}
        
        self.BDD.collect_garbage()
        # self.BDD.reorder()

        self.TreeMapInNodes = {}
        self.NodeIDMap = {}
        for rootKey in self.Expressions:
            # queue = [(self.Expressions[rootKey], self.Expressions[rootKey].negated)]
            queue = [(self.Expressions[rootKey], False)]
            while(queue):
                curr_ele, neg = queue.pop(0)
                [lvl, node1, node2] = self.BDD.succ(curr_ele)

                # Store current node structure in TreeMapInNodes
                lowValue = None
                highValue = None

                neg1 = not neg if curr_ele.negated else neg

                neg2 = not neg if curr_ele.negated else neg
                
                if node1.var:
                    lowValue = str(node1)
                else:
                    lowValue = str(node1)
                    # lowValue =  node1.negated if neg1 else not node1.negated
                    
                if node2.var:
                    highValue = str(node2)
                else:
                    highValue = str(node2)
                    # highValue = node2.negated if neg2 else not node2.negated

                self.TreeMapInNodes[str(curr_ele)] = {
                    "variable": curr_ele.var,
                    "negation": curr_ele.negated,
                    "low": lowValue,
                    "high": highValue
                }
                
                if(str(curr_ele) not in self.NodeIDMap):
                    self.NodeIDMap[str(curr_ele)]=[0, curr_ele.var]
                self.NodeIDMap[str(curr_ele)][0]+=1
                
                if node1.var!=None:
                    queue.append((node1, neg1))
                else:
                    if(str(node1) not in self.NodeIDMap):
                        self.NodeIDMap[str(node1)]=[0, '0' if node1.negated else '1']
                    self.NodeIDMap[str(node1)][0]+=1
                
                if node2.var!=None:
                    queue.append((node2, neg2))
                else:
                    if(str(node2) not in self.NodeIDMap):
                        self.NodeIDMap[str(node2)]=[0, '0' if node2.negated else '1']
                    self.NodeIDMap[str(node2)][0]+=1

    def BDD_to_NetworkXGraph(self):
        # Initialize an undirected graph
        self.Graph = nx.DiGraph()
        ExpressionsRev = {str(self.Expressions[key]):key for key in self.Expressions}

        # print('---------------------------------')
        # #debug code by nithin
        # print('self.NodeIDMap', self.NodeIDMap)
        # print()
        # print('ExpressionsRev', ExpressionsRev)
        # print()
        # print('self.TreeMapInNodes', self.TreeMapInNodes)
        # print('---------------------------------')
        #adding nodes
        for id_str in self.NodeIDMap:
            literal = self.NodeIDMap[id_str][1]
            ExpressionRoot = None
            if(id_str in ExpressionsRev):
                ExpressionRoot = ExpressionsRev[id_str]
                literal = literal + '('+str(ExpressionsRev[id_str])+')'
                
            attributes = {'ID': id_str, 'literal': literal, 'ExpressionRoot': ExpressionRoot, 'BipartitePart':None}

            # Add nodes with attributes to the graph
            self.Graph.add_node(id_str, **attributes)

        for rootKey in self.TreeMapInNodes:
            node1 = self.TreeMapInNodes[rootKey]['low']
            node2 = self.TreeMapInNodes[rootKey]['high']
                
            self.Graph.add_edge(str(rootKey), str(node1), label='0')
            self.Graph.add_edge(str(rootKey), str(node2), label='1')

        self.GraphProcessPhase = "0. BDD creation"

    def Verify_BDD_to_NetworkXGraph(self, withExpression=False, withOriginalTruthTable=False, generateFromBDD=False):
        """
        Verifies if the BDD truth table representation matches the expected truth table.
        """

        outputs = self.outputs
        dfs = []
        
        if(withOriginalTruthTable):
            if(self.OriginalTruthTable is not None):
                dfs.append(self.OriginalTruthTable)
                # print('self.OriginalTruthTable.columns',self.OriginalTruthTable.columns)
            else:
                print("No Original Truth table(user given truth table) exists")

        if(withExpression):
            if(self.pre_bool_expressions and self.pre_varibles_lst):
                self.GetTruthTables()
                dfs.append(self.TruthTable)
            else:
                print("No  Evaluated Truth table exists")

        self.inputs.sort()
        if(generateFromBDD):
            combinations = list(itertools.product([0, 1], repeat=len(self.inputs)))
            self.BDDTruthTable = pd.DataFrame(combinations, columns=self.inputs)

        flag = True
        for i, df in enumerate(dfs):
            # Extract input and output column names from the dataframe
            input_columns = [col for col in df.columns if col not in outputs]  # Variables
            output_columns = [col for col in df.columns if col in outputs]  # Expressions (functions)

            # Iterate over each row in the dataframe (each row represents an input assignment)
            for idx, row in df.iterrows():
                input_assignment = {var: int(row[var]) for var in input_columns}  # Convert inputs to dictionary
                expected_outputs = {expr: int(row[expr]) for expr in output_columns}  # Expected output values

                
                computed_outputs = {}  # Store computed values from BDD traversal

                # Evaluate each expression using the BDD
                for expr in output_columns:
                    if expr not in self.Expressions:
                        print(f"Error: Expression {expr} not found in BDD.")
                        continue
                    
                    current_node = str(self.Expressions[expr])  # Start traversal from the root node of the expression
                    
                    while True:
                        # Get node attributes
                        if current_node not in self.Graph.nodes:
                            print(f"Error: Node {current_node} not found in graph.")
                            return None
                
                        node_data = self.Graph.nodes[current_node]
                        # print(f"Visiting Node {current_node}: {node_data}")

                        # If it's a terminal node, return the computed output (0 or 1)
                        if node_data["literal"] in {"0", "1"}:
                            computed_outputs[expr] = int(node_data["literal"])
                            break
                        
                        # Extract the literal (decision variable)
                        literal = node_data["literal"].split('(')[0]  # Extracts 'a' from 'a(y)'

                        # Determine next node based on input assignment
                        if literal in input_assignment:
                            next_node = None
                            if input_assignment[literal] == 0:
                                next_node = list(self.Graph.successors(current_node))[0]  # Low branch
                            else:
                                next_node = list(self.Graph.successors(current_node))[1]  # High branch
                            
                            current_node = next_node  # Move to the next node
                        else:
                            print(f"Error: Variable '{literal}' not in input assignment.")
                            return None

                # print('input_assignment', input_assignment)
                # print('expected_outputs', expected_outputs)
                # Compare computed outputs with expected outputs
                for expr in output_columns:
                    if computed_outputs.get(expr) != expected_outputs[expr]:
                        print(f"Mismatch at row {idx}: Inputs {input_assignment}, "
                              f"Expected {expected_outputs}, Got {computed_outputs}")
                        flag=False

            if(flag):
                print("BDD has no issues")

        if(generateFromBDD):
            input_columns = [col for col in self.BDDTruthTable.columns]  # Variables
            output_columns = []
            computed_outputs_map = {output:[] for output in self.outputs}  # Store computed values from BDD traversal
            
            # Iterate over each row in the dataframe (each row represents an input assignment)
            for idx, row in self.BDDTruthTable.iterrows():

                input_assignment = {var: int(row[var]) for var in input_columns}  # Convert inputs to dictionary
    
                # Evaluate each expression using the BDD
    
                current_nodes = [n for n, deg in self.Graph.in_degree() if deg == 0]
                for current_node in current_nodes:                    
                    output_label = self.Graph.nodes[current_node]["literal"].split('(')[1].rstrip(')')
                    
                    while True:
                        # Get node attributes
                        if current_node not in self.Graph.nodes:
                            print(f"Error: Node {current_node} not found in graph.")
                            return None
                
                        node_data = self.Graph.nodes[current_node]
                        # print(f"Visiting Node {current_node}: {node_data}")
        
                        # If it's a terminal node, return the computed output (0 or 1)
                        if node_data["literal"] in {"0", "1"}:
                            computed_outputs_map[output_label].append(int(node_data["literal"]))
                            break
                        
                        # Extract the literal (decision variable)
                        literal = node_data["literal"].split('(')[0]  # Extracts 'a' from 'a(y)'
        
                        # Determine next node based on input assignment
                        if literal in input_assignment:
                            next_node = None
                            if input_assignment[literal] == 0:
                                next_node = list(self.Graph.successors(current_node))[0]  # Low branch
                            else:
                                next_node = list(self.Graph.successors(current_node))[1]  # High branch
                            
                            current_node = next_node  # Move to the next node

            # print(self.BDDTruthTable.shape,len(computed_outputs))
            for output in self.outputs:
                self.BDDTruthTable[output] = computed_outputs_map[output]

        print("BDD verification completed.")
            
    def GraphPreprocessing(self):
        #Re-label the edges
        for u, v, data in self.Graph.edges(data=True):
            # Retrieve the parent node's literal
            parent_literal = self.Graph.nodes[u].get('literal').split('(')[0]
            
            # Ensure the parent_literal is valid (not None) and the edge has a label
            if parent_literal and 'label' in data:
                # Update the edge label based on the parent node's literal
                if data['label'] == '0':
                    # For '0', add a negation (~) to the parent's literal
                    data['label'] = f"~{parent_literal}"
                elif data['label'] == '1':
                    # For '1', use the parent's literal directly
                    data['label'] = parent_literal

        # Remove the node with literal='0' and id='@-1' along with its connections
        nodes_to_remove = [node for node, data in self.Graph.nodes(data=True) if data.get('literal') == '0']
        for node in nodes_to_remove:
            self.Graph.remove_node(node)


        #Invert all edges in the graph 
        inverted_graph = nx.DiGraph()
        inverted_graph.add_nodes_from(self.Graph.nodes(data=True))
        inverted_graph.add_edges_from([(v, u, data) for u, v, data in self.Graph.edges(data=True)])
        self.Graph = inverted_graph

        # Store the root node (in RootNode) after inversion        
        self.InputNode = [node for node in self.Graph.nodes if self.Graph.nodes[node].get('literal')=='1' and self.Graph.in_degree(node) == 0][0]

        #Re-label the nodes
        Counter = 1
        queue = [self.InputNode]
        visited = set()
        while queue:
            current_node = queue.pop(0)

            if current_node in visited:
                continue
            visited.add(current_node)

            # Update node labels for the current node's
            self.Graph.nodes[str(current_node)]['literal'] = str(Counter)
            Counter+=1

            for _, target_node, edge_data in self.Graph.out_edges(str(current_node), data=True):
                queue.append(target_node)

        # Wordline nodes counter
        self.MainNodesCounter = Counter
                
        self.GraphProcessPhase = "1. Graph pre-processed"

    def GraphTransformation(self):
        # First mark all the old nodes with attribute as BipartitePart='U1'
        for node in self.Graph.nodes:
            self.Graph.nodes[node]['BipartitePart'] = 'U1'
    
        # Initialize a list to store new edges transformed into nodes
        new_nodes = []
        edge_counter = 1  # Counter for unique IDs for new nodes
    
        # Iterate through all edges in the graph
        for u, v, data in list(self.Graph.edges(data=True)):
            # Create a new node for the edge
            new_node_id = f"EdgeNode_{edge_counter}"
            edge_label = data.get('label', 'NoLabel')
            new_node_attributes = {
                'ID': new_node_id,
                'literal': edge_label,
                'BipartitePart': 'U2'
            }
    
            # Add the new node to the graph
            self.Graph.add_node(new_node_id, **new_node_attributes)
    
            # Connect the new node to the original source and target nodes
            self.Graph.add_edge(u, new_node_id, label='')
            self.Graph.add_edge(new_node_id, v, label='')
    
            # Remove the original edge
            self.Graph.remove_edge(u, v)
    
            # Keep track of the newly created node
            new_nodes.append(new_node_id)
    
            edge_counter += 1
    
        # Update the graph process phase
        self.GraphProcessPhase = "2. Graph Transformation"


    def GraphCompression(self):
        # Create a dictionary to store U2 node literals as keys and input node literals as values in a list
        compression_dict = {}

        # Iterate through all U2 nodes
        for node in self.Graph.nodes:
            if self.Graph.nodes[node].get('BipartitePart') == 'U2':
                # Get the literal of the current U2 node
                u2_literal = self.Graph.nodes[node].get('literal')
                u2_id = self.Graph.nodes[node].get('ID')

                # Collect input node literals (U1) connected to this U2 node
                if(u2_literal not in compression_dict):
                    compression_dict[u2_literal] = {}

                if(u2_id not in compression_dict[u2_literal]):
                    compression_dict[u2_literal][u2_id] = []
                
                for predecessor in self.Graph.predecessors(node):
                    if self.Graph.nodes[predecessor].get('BipartitePart') == 'U1':
                        compression_dict[u2_literal][u2_id].append(self.Graph.nodes[predecessor].get('literal'))

        # Merge U2 nodes with the same literal if they have the same input edges
        for u2_literal, nodes in compression_dict.items():
            merged_inputs = {}
            for node_id, inputs in nodes.items():
                inputs_tuple = tuple(sorted(inputs))  # Sort to handle duplicate edge inputs
                if inputs_tuple not in merged_inputs:
                    merged_inputs[inputs_tuple] = node_id
                else:
                    # Merge this node into the existing one
                    existing_node_id = merged_inputs[inputs_tuple]

                    # Redirect all outgoing edges from the current node to the existing node
                    for _, successor, edge_data in list(self.Graph.out_edges(node_id, data=True)):
                        self.Graph.add_edge(existing_node_id, successor, **edge_data)

                    # Remove the current node
                    self.Graph.remove_node(node_id)

        #Add the output nodes to the BDD
        
        # Track the next available EdgeNode index
        edge_node_index = max(
            [int(node.split("_")[-1]) for node in self.Graph.nodes if "EdgeNode_" in str(node)],
            default=13  # Assuming EdgeNode_13 is the last used
        ) + 1
        
        self.OutputLine_Map = {}
        outputNodes = []
        
        for node in list(self.Graph.nodes):
            node_data = self.Graph.nodes[node]
            # print('node_data',node_data)
            if node_data.get("ExpressionRoot") is not None:  # If ExpressionRoot is not None
                new_edge_node_id = f"EdgeNode_{edge_node_index}"  # Create a new EdgeNode
        
                # Add the new node with '1' as the literal and 'U2' as BipartitePart
                self.Graph.add_node(new_edge_node_id, ID=new_edge_node_id, literal="O"+str(edge_node_index), BipartitePart="U2")
                
                # Connect the original node to the new EdgeNode
                self.Graph.add_edge(node, new_edge_node_id)

                self.OutputLine_Map["O"+str(edge_node_index)] = self.Graph.nodes[node].get('ExpressionRoot')

                outputNodes.append(new_edge_node_id)

                edge_node_index += 1  # Increment for the next node
        
        #create an ID for End outut node that is going to be grounded  #nithin2
        end_output_edge_node_id = f"EdgeNode_{edge_node_index}"

        intermediate_node_ID = self.MainNodesCounter
        
        # Add the intermediate node as number and 'U1' as BipartitePart
        self.Graph.add_node(intermediate_node_ID, ID=intermediate_node_ID, literal=intermediate_node_ID, BipartitePart="U1")

        #Storing endoutput bitlines label
        self.End_Bitline_Output_Line = "O"+str(edge_node_index)
        
        # Add the end output node and 'U2' as BipartitePart
        self.Graph.add_node(end_output_edge_node_id, ID=end_output_edge_node_id, literal="O"+str(edge_node_index), BipartitePart="U2")

        # connect the intermediate node to the end output node
        self.Graph.add_edge(intermediate_node_ID, end_output_edge_node_id)
        
        for i, outputNode in enumerate(outputNodes):
            
            # Connect the output nodes to intermediate node
            self.Graph.add_edge(outputNode, intermediate_node_ID)

        # Update the graph process phase
        self.GraphProcessPhase = "3. Graph Compression"

    def CrossbarRelalization(self):

        self.colMap, self.rowMap, counter, main_node_counter = {}, {}, 0, 0
        for node in self.Graph.nodes:
            if(self.Graph.nodes[node]['BipartitePart']=='U2'):
                self.colMap[self.Graph.nodes[node]['ID']] = counter
                counter += 1
            if(self.Graph.nodes[node]['BipartitePart']=='U1'):
                self.rowMap[self.Graph.nodes[node]['ID']] = int(self.Graph.nodes[node]['literal'])-1
                main_node_counter += 1

        #start of setting the crossbar size
        max_CrossbarGridSize = 0
        for i,(u, v, data) in enumerate(self.Graph.edges(data=True)):
            if(self.Graph.nodes[u]['BipartitePart']=='U2'):
                row_i = int(self.Graph.nodes[v]['literal'])-1
                col_j = self.colMap[self.Graph.nodes[u]['ID']]
            else:
                row_i = int(self.Graph.nodes[u]['literal'])-1
                col_j = self.colMap[self.Graph.nodes[v]['ID']]
            max_CrossbarGridSize = max(max_CrossbarGridSize, row_i, col_j)
            
        for i,Crossbar_size in enumerate(self.Crossbar_sizes):
            # if(Crossbar_size>=max_CrossbarGridSize):
            if(Crossbar_size>=max_CrossbarGridSize and Crossbar_size>=self.CrossbarGridSize):
                self.CrossbarGridSize = Crossbar_size
                break
        print("Suitable Crossbar size:", self.CrossbarGridSize)

        self.SBDD_dimentions = f"{counter} x {main_node_counter}"
        
        self.CrossbarResistiveStates = [[0 for _ in range(self.CrossbarGridSize)] for _ in range(self.CrossbarGridSize)]
        #end of setting crossbar size
        
        self.Bitlines = []
        for i,(u, v, data) in enumerate(self.Graph.edges(data=True)):
            if(self.Graph.nodes[u]['BipartitePart']=='U2'):
                row_i = int(self.Graph.nodes[v]['literal'])-1
                col_j = self.colMap[self.Graph.nodes[u]['ID']]
                
            else:
                row_i = int(self.Graph.nodes[u]['literal'])-1
                col_j = self.colMap[self.Graph.nodes[v]['ID']]

            self.Bitlines = [f"C{i+1}" for i in range(self.CrossbarGridSize)]
            for col in self.colMap:
                self.Bitlines[self.colMap[col]] = self.Graph.nodes[col]['literal']
            
            self.CrossbarResistiveStates[row_i][col_j] = 1

        self.CrossbarLongPaths = {}

        # print('len(self.BDDlongPaths)',len(self.BDDlongPaths))

        for path in self.BDDlongPaths:
            CrossbarPath = []
            SelectorlinesActiveLiterals = []
            # print('self.Bitlines',self.Bitlines)
            for j in range(len(path)-1):
                if(j%2==0):
                    node1, node2 = path[j],path[j+1]
                else:
                    node2, node1 = path[j],path[j+1]
                row_index, col_index = self.rowMap[node1], self.colMap[node2]
                # print('row_index',row_index)
                # print('col_index',col_index)
                SelectorlinesActiveLiterals.append(self.Bitlines[col_index])
                CrossbarPath.append((row_index, col_index))

            LiteralMap = {}
            # print('SelectorlinesActiveLiterals', SelectorlinesActiveLiterals)
            for literal in SelectorlinesActiveLiterals:
                if(literal[0]=="O"):
                    continue
                elif(literal[0]=="~"):
                    LiteralMap[literal[1:]] = 0
                else:
                    LiteralMap[literal] = 1
                    
            self.CrossbarLongPaths[frozenset(LiteralMap.items())] = CrossbarPath
            
        self.GraphProcessPhase = "4. Crossbar Realization"
    
    def DisplayEdgesInNetworkXGraph(self):
        # Iterate through all edges in the graph
        for i,(u, v, data) in enumerate(self.Graph.edges(data=True)):
            # Retrieve the edge label; default to 'No label' if not present
            edge_label = data.get('label', 'No label')
            print(f"{next(iter(self.NodeIDMap[u][1]))[0]}({u}) -[{edge_label}]-> {next(iter(self.NodeIDMap[v][1]))[0]}({v})")

    def VisuvaliseNetworkXGraph(self, bipartite=False):
        # Initialize position dictionary
        pos = {}
        
        # Parameters for positioning
        HORIZONTAL_SPACING = 1.0  # Horizontal distance between nodes
        VERTICAL_SPACING = 3.0    # Vertical distance between layers
        
        # Separate nodes into categories
        root_nodes = [node for node in self.Graph.nodes if self.Graph.in_degree(node) == 0]
        leaf_nodes = [node for node in self.Graph.nodes if self.Graph.out_degree(node) == 0]
        intermediate_nodes = [node for node in self.Graph.nodes if node not in root_nodes + leaf_nodes]

        if(bipartite):
            # Separate nodes by BipartitePart
            u1_nodes = [node for node in self.Graph.nodes if self.Graph.nodes[node].get('BipartitePart') == 'U1']
            u2_nodes = [node for node in self.Graph.nodes if self.Graph.nodes[node].get('BipartitePart') == 'U2']
    
            # Assign positions to U1 nodes (left column)
            for i, node in enumerate(u1_nodes):
                pos[node] = (0, (i * VERTICAL_SPACING))
            
            # Assign positions to U2 nodes (right column)
            for i, node in enumerate(u2_nodes):
                pos[node] = (2, (i * VERTICAL_SPACING))

        else:
            # Assign positions to ExpressionRoot nodes (top layer)
            for i, node in enumerate(root_nodes):
                pos[node] = (i * HORIZONTAL_SPACING, 0)
                # print((i * HORIZONTAL_SPACING, 0))
            
            # Assign positions to Intermediate nodes based on BFS layers
            # Start BFS from all ExpressionRoot nodes
            layers = list(nx.bfs_layers(self.Graph, root_nodes))
            
            for lvl, nodes in enumerate(layers):
                # print("Layer:")
                for i, node in enumerate(nodes):
                    if node in intermediate_nodes:  # Only position intermediate nodes
                        # print(next(iter(idSet[node][1]))[0], end=",")
                        pos[node] = (i * HORIZONTAL_SPACING, -VERTICAL_SPACING * (lvl))
                        # print((i * HORIZONTAL_SPACING, -VERTICAL_SPACING * (lvl)))
                # print("\n")
    
            leafnodeYaxis = len(layers)
            if(self.Graph.nodes[layers[0][0]].get('literal')=="1"):
                leafnodeYaxis = leafnodeYaxis - 1
            
            # print("Literals")
            # Assign positions to Literal nodes (bottom layer)
            for i, node in enumerate(leaf_nodes):
                pos[node] = (i * HORIZONTAL_SPACING, -VERTICAL_SPACING * (leafnodeYaxis))
                # print((i * HORIZONTAL_SPACING, -VERTICAL_SPACING * (leafnodeYaxis)))
        
        # Draw nodes with labels
        node_colors = []
        for node in self.Graph.nodes:
            if self.Graph.nodes[node].get('ExpressionRoot') is not None:
                node_colors.append('orange')
            elif self.Graph.nodes[node].get('literal') == '0':
                node_colors.append('red')
            elif self.Graph.nodes[node].get('literal') == '1':
                node_colors.append('green')
            else:
                node_colors.append('lightblue')  # Default color for intermediate nodes

        nx.draw(self.Graph, pos, with_labels=False, node_color=node_colors, node_size=3000, font_size=10, arrows=True, arrowstyle='->', arrowsize=20)
        
        # Draw literals as node labels
        node_labels = nx.get_node_attributes(self.Graph, 'literal')
        nx.draw_networkx_labels(self.Graph, pos, labels=node_labels, font_size=12, font_color='black')
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.Graph, 'label')
        nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels=edge_labels, font_color='red')
        
        # Display the graph
        plt.title(self.GraphProcessPhase if self.GraphProcessPhase is not None else "Graph with Node Attributes and Edge Labels")
        plt.show()

    def VisuvaliseCrossbar(self, initialisedCrossbar=None):
        if(initialisedCrossbar==None):
            crossbar_matrix = np.array(self.CrossbarResistiveStates)
            crop=True
        else:
            crossbar_matrix = np.array(initialisedCrossbar)
            crop=False

        colors = ["red", "blue", "black"]
        if(crop):
            colors = colors[:-1]
        custom_cmap = ListedColormap(colors)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(crossbar_matrix, cmap=custom_cmap, aspect="auto")

        # Annotate the heatmap with 'LRS', 'HRS', or 'Off'
        for i in range(crossbar_matrix.shape[0]):
            for j in range(crossbar_matrix.shape[1]):
                if crossbar_matrix[i, j] == 1:
                    text = "LRS"        # Same label for value=1
                    font_color = "white"
                elif crossbar_matrix[i, j] == 2 and self.CrossbarResistiveStates[i][j] == 0:
                    text = "Off"        # New label for value=2
                    font_color = "white"
                elif crossbar_matrix[i, j] == 2 and self.CrossbarResistiveStates[i][j] == 1:
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
        ax.set_xticks(np.arange(crossbar_matrix.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(crossbar_matrix.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Set labels and title
        ax.set_xlabel("Columns", fontsize=12, weight="bold")
        ax.set_ylabel("Rows", fontsize=12, weight="bold")
        ax.set_title("Crossbar Resistive States", fontsize=14, weight="bold")

        xlabelsOfLiterals = [f"C{i+1}" for i in range(crossbar_matrix.shape[0])]
        for col in self.colMap:
            xlabelsOfLiterals[self.colMap[col]] = self.Graph.nodes[col]['literal']

        #shif this to functional functions
        # xlabelsOfLiterals ['a1', 'a0', '~a1', '~a1', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']
        self.xlabelsOfLiterals_map = {}
        for i,xlabelsOfLiteral in enumerate(xlabelsOfLiterals):
            
            only_label = xlabelsOfLiteral if xlabelsOfLiteral[0]!='~' else xlabelsOfLiteral[1:]
            negation = False if xlabelsOfLiteral[0]!='~' else True
            
            if(only_label not in self.xlabelsOfLiterals_map):
                self.xlabelsOfLiterals_map[only_label] = []
            self.xlabelsOfLiterals_map[only_label].append((i, negation))

        # Remove default ticks
        ax.set_xticks(np.arange(crossbar_matrix.shape[1]))
        ax.set_yticks(np.arange(crossbar_matrix.shape[0]))
        ax.set_xticklabels(xlabelsOfLiterals, fontsize=10)
        ax.set_yticklabels([f"{j+1}" for j in range(crossbar_matrix.shape[1])], fontsize=10)

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
                if(idx%1000!=0):
                    continue
                print(idx)
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

    def GetTruthTables(self):
        variables = list(self.pre_varibles_lst)
        truth_table = []
        # Generate all combinations of variable assignments (0 and 1)
        for values in itertools.product([0, 1], repeat=len(variables)):
            assignment = dict(zip(variables, values))
            row = {var: val for var, val in assignment.items()}
            
            for expr_name, expr in self.pre_bool_expressions.items():
                # Evaluate the expression in the context of the current assignment
                row[expr_name] = eval(expr, {}, assignment)
            
            truth_table.append(row)

        self.TruthTable = pd.DataFrame(truth_table)

        return self.TruthTable

    def LongestpathInTreeAndCrossbar(self):
        # 1) Identify the start node (in-degree = 0)
        start_node = [n for n, deg in self.Graph.in_degree() if deg == 0][0]
        # print("Start nodes (in-degree = 0):", self.Graph.nodes[start_node])

        def dag_longest_path_lengths(G, start_node):
            """
            Returns a dict of the longest path length from 'start_node' to each node in the DAG 'G'.
            Raises an error if 'G' is not a DAG.
            """
            # 1) Check if G is a DAG:
            if not nx.is_directed_acyclic_graph(G):
                raise ValueError("Longest path computation is only straightforward for DAGs. "
                                 "Your graph has cycles, so the problem is NP-hard.")
        
            # 2) Initialize all distances to -∞, except start_node at 0
            distances = {node: float('-inf') for node in G.nodes()}
            distances[start_node] = 0

            # track parents to reconstruct path
            parents = {node: None for node in G.nodes()}
        
            # 3) Process nodes in topological order to find longest distances
            for u in nx.topological_sort(G):
                for v in G.successors(u):
                    candidate_dist = distances[u] + 1
                    if candidate_dist > distances[v]:
                        distances[v] = candidate_dist
                        parents[v] = u

            # 4) Identify the farthest node
            farthest_node = max(distances, key=distances.get)
            longest_distance = distances[farthest_node]

            # 5) Reconstruct ONE longest path from `start_node` to `farthest_node`
            path = []
            current = farthest_node
            while current is not None:
                path.append(current)
                current = parents[current]
            path.reverse()  # because we built it from farthest_node back to start_node
        
            return distances, path, longest_distance
            
        
        # 2) Compute the distance from start_node to all other nodes
        # Compute distances and retrieve ONE longest path
        distance_dict, single_longest_path, longest_distance = dag_longest_path_lengths(self.Graph, start_node)

        self.BDDlongPaths = [single_longest_path]