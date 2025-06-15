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

class ParentCountError(Exception):
    """Raised when a U2 node does not have exactly one U1 parent."""
    pass

class PATH:
    def __init__(self, CrossbarGridSize = 16):

        self.HeightThereshold = 8
        self.SizeThereshold = 64

        self.pre_bool_expressions = None
        self.pre_varibles_lst = None

        self.filename = None
        self.model_name = None
        self.inputs = []
        self.outputs = []

        self.OriginalTruthTable = None
        self.BDDTruthTable = None
        
        self.Graph = None
        self.Expressions = None
        self.NodeIDMap = None
        self.InputNode = None
        self.GraphProcessPhase = None

        self.TreeMapInNodes = {}

        self.Included_NodeIdToDesignIdMap = {}
        
        self.output_node_index = 0
        
        self.Processed_height_constraint_graphs_Map = {}
        self.Processed_group_graphs_Map = {}
        self.Included_nodes = set()

        self.OutputLine_Map = {}
        self.Topological_order = []  #design_ID_groups - order of execution

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

    def SetBooleanExpressionsAndVaribles(self, variables=None, expressions=None, outputs=None, OriginalTruthTable=None):
        if(variables!=None):
            self.pre_varibles_lst = variables
        if(expressions!=None):
            self.pre_bool_expressions = expressions
        if(outputs!=None):
            self.outputs = outputs
        if OriginalTruthTable is not None:
            self.OriginalTruthTable = OriginalTruthTable

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

        ##############################
        # Make a copy of the current node list because we'll modify the graph
        nodes_to_remove = []
        for node in self.Graph.nodes:
            if 'ID' not in self.Graph.nodes[node]:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.Graph.remove_node(node)

        ##########################

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

        # # Update the graph process phase
        self.GraphProcessPhase = "3. Graph Compression"

    # def get_longest_distances_from_root(self, graph, root, threshold):

    #     inclusion_nodes = set([root])  # your "Visited" set for threshold logic
    
    #     # Initialize longest distances
    #     longest_distances = {node: float('-inf') for node in graph.nodes}
    #     longest_distances[root] = 0

    #     # BFS queue
    #     queue = deque([root])
        
    #     # for node in nx.topological_sort(graph):
    #     while queue:
    #         node = queue.popleft()
    #         for neighbor in graph.successors(node):
    #             LongDistance = max(
    #                 longest_distances[neighbor],
    #                 longest_distances[node] + 1
    #             )
    #             if(LongDistance<=threshold):
    #                 longest_distances[neighbor] = LongDistance
    #                 inclusion_nodes.add(neighbor)
    #                 queue.append(neighbor)
    #             else:
    #                 if(neighbor not in inclusion_nodes):
    #                     longest_distances[neighbor] = LongDistance

    #     longest_distances_temp={}
    #     for dist in longest_distances:
    #         node_literal = graph.nodes[dist]['literal']
    #         longest_distances_temp[node_literal] = longest_distances[dist]
            
    #     print('longest_distances2',longest_distances_temp)
        
    #     return longest_distances

    def getDistanceFromRoot(self, graph):
        # find the node with in-degee 0
        root_candidates = [n for n in graph.nodes if graph.in_degree(n) == 0]
        if not root_candidates:
            raise ValueError("No root node found with in-degree 0")

        # Initialize distances
        longest_distances = {node: float('-inf') for node in graph.nodes}
        for root in root_candidates:
            longest_distances[root] = 0
        
        # Process in topological order
        for node in nx.topological_sort(graph):
            for neighbor in graph.successors(node):
                if longest_distances[neighbor] < longest_distances[node] + 1:
                    longest_distances[neighbor] = longest_distances[node] + 1
    
        # Set the 'distance' attribute for each node
        for node, dist in longest_distances.items():
            graph.nodes[node]['distance'] = dist
    
        return graph

    def get_full_top_graph(self, top_graph_nodes):
        """
        Get the subgraph containing all nodes on paths between the top U1 wordLine nodes.
        """

        
        # Include all nodes reachable from or reaching to any top-level U1 node
        nodes_to_include = set(top_graph_nodes)
        
        graph = self.Graph
        
        # Optional debug: sanity check for stray ancestors
        # for node in nodes_to_include:
        #     for parent in graph.predecessors(node):
        #         if parent not in nodes_to_include:
        #             print(f"Warning: Node {node} has external parent {parent} not in subgraph.")
    
        return graph.subgraph(nodes_to_include).copy()

    def find_split_nodes(self, top_graph):
        """
        Identify U1 & U2 nodes in top_graph that connect to nodes outside top_graph.
        """
        full_graph = self.Graph
        top_nodes_set = set(top_graph.nodes)
        pred_split_nodes_map, succ_split_nodes_map = {}, {}
        visited = set()

        # Step 1: Find all root nodes (in-degree 0 in top_graph and U1)
        roots = [
            node for node in top_graph.nodes
            if top_graph.in_degree(node) == 0 and top_graph.nodes[node].get("BipartitePart") == "U1"
        ]
        roots_set = set(roots)
        queue = deque(roots)
        visited.update(roots)

        # print('roots',roots)
        # print('top_graph in func', [top_graph.nodes[n]['literal'] for n in top_graph.nodes])

        while queue:
            next_layer = deque()

            #traverse through nodes in first layer
            while queue:
                node = queue.popleft()

                #root nodes always have prequisites to communicate
                if node in roots_set:
                    for pred in full_graph.predecessors(node):
                        if pred not in top_nodes_set:
                            if(node not in pred_split_nodes_map):
                                pred_split_nodes_map[node] = []
                            pred_split_nodes_map[node].append(pred)

                for succ in full_graph.successors(node):
                    if succ not in top_nodes_set:
                        if(node not in succ_split_nodes_map):
                            succ_split_nodes_map[node] = []
                        succ_split_nodes_map[node].append(succ)

                # Add successors in top_graph to next layer
                for neighbor in top_graph.successors(node):
                    if neighbor not in visited:
                        next_layer.append(neighbor)
                        visited.add(neighbor)

            # Move to next BFS layer
            queue = next_layer

        # print('pred split_nodes in func',[top_graph.nodes[n]['literal'] for n in pred_split_nodes_map])
        # print('succ split_nodes in func',[top_graph.nodes[n]['literal'] for n in succ_split_nodes_map])
        return pred_split_nodes_map, succ_split_nodes_map

    def find_output_label_nodes(self, top_graph):
        """
        Identify U1 nodes in top_graph that connect to nodes outside top_graph.
        """
        return [{"node_id":node, "literal":top_graph.nodes[node]['literal'], "ExpressionRoot":top_graph.nodes[node]['ExpressionRoot']} for node in top_graph.nodes if top_graph.nodes[node].get('ExpressionRoot', None) != None]

    def add_output_node(self, graph, split_node):
        """
        Add a new output node to the graph connected from split_node. Returns modified graph and the new node ID.
        """
        # Track the next available EdgeNode index
        edge_node_ids = [int(str(node).split("_")[-1]) for node in graph.nodes if str(node).startswith("EdgeNode_")]
        
        edge_node_index = max(edge_node_ids) + 1 if edge_node_ids else 1

        FinalLeafNode = False
        SplitLeafNode = False
        
        node_data = graph.nodes[split_node]
        if node_data.get("ExpressionRoot") is not None:
            FinalLeafNode = True
        else:
            SplitLeafNode = True

        if FinalLeafNode or SplitLeafNode:  # If ExpressionRoot is not None

            literal = "O"+str(self.output_node_index)

            new_edge_node_id = f"EdgeNode_{edge_node_index}"  # Create a new EdgeNode
            
            # Add the new node with '1' as the literal and 'U2' as BipartitePart
            graph.add_node(new_edge_node_id, ID=new_edge_node_id, literal=literal, BipartitePart="U2", split_id=None, in_split_id=set(), out_split_id=None)
            
            # Connect the original node to the new EdgeNode
            graph.add_edge(split_node, new_edge_node_id)

        endingNodeLabel = graph.nodes[split_node].get('ExpressionRoot') if FinalLeafNode else None
        
        self.output_node_index += 1
    
        return graph, new_edge_node_id, literal, endingNodeLabel

    def add_output_nodes(self, graph, split_node):
        """
        Add a new output node to the graph connected from split_node. Returns modified graph and the new node ID.
        """

        #Add U1 (Word Line) output node
        literal = max([int(graph.nodes[node]['literal'].split('(')[0]) for node in graph if graph.nodes[node]['BipartitePart'] == 'U1'])+1
        id_str = f"{literal}_ID"
        U1_node_attributes = {'ID': id_str, 'literal': str(literal), 'ExpressionRoot': None, 'BipartitePart':'U1', 'split_id':None, 'in_split_id':set(), 'out_split_id':None}
        graph.add_node(id_str, **U1_node_attributes)
        
        #Add U2 (Bit Line) output node
        
        # Track the next available EdgeNode index
        edge_node_ids = [int(str(node).split("_")[-1]) for node in graph.nodes if str(node).startswith("EdgeNode_")]
        edge_node_index = max(edge_node_ids) + 1 if edge_node_ids else 1

        FinalLeafNode = False
        SplitLeafNode = False
        
        node_data = graph.nodes[split_node]
        if node_data.get("ExpressionRoot") is not None:
            FinalLeafNode = True
        else:
            SplitLeafNode = True

        if FinalLeafNode or SplitLeafNode:  # If ExpressionRoot is not None

            literal = "O"+str(self.output_node_index)

            new_edge_node_id = f"EdgeNode_{edge_node_index}"  # Create a new EdgeNode
            
            # Add the new node with '1' as the literal and 'U2' as BipartitePart
            graph.add_node(new_edge_node_id, ID=new_edge_node_id, literal=literal, BipartitePart="U2", split_id=None, in_split_id=set(), out_split_id=None)
            
            # Connect the original node to the u1_node
            graph.add_edge(split_node, id_str)

            # Connect the u1_node to the  new EdgeNode
            graph.add_edge(id_str, new_edge_node_id)
            

        endingNodeLabel = graph.nodes[split_node].get('ExpressionRoot') if FinalLeafNode else None
        
        self.output_node_index += 1
    
        return graph, new_edge_node_id, literal, endingNodeLabel

    def add_expression_output_node(self, graph, output_node):
        """
        Add a new output node to the graph connected from output_node. Returns modified graph and the new node ID.
        """
        # Track the next available EdgeNode index
        edge_node_ids = [int(str(node).split("_")[-1]) for node in graph.nodes if str(node).startswith("EdgeNode_")]
        edge_node_index = max(edge_node_ids) + 1 if edge_node_ids else 1

        literal = "O"+str(self.output_node_index)

        new_edge_node_id = f"EdgeNode_{edge_node_index}"  # Create a new EdgeNode

        endingNodeLabel = graph.nodes[output_node].get('ExpressionRoot')
        
        # Add the new node with '1' as the literal and 'U2' as BipartitePart
        graph.add_node(new_edge_node_id, ID=new_edge_node_id, literal=literal, BipartitePart="U2", ExpressionRoot=endingNodeLabel, split_id=None, in_split_id=set(), out_split_id=None)
        
        # Connect the original node to the new EdgeNode
        graph.add_edge(output_node, new_edge_node_id)
        
        self.output_node_index += 1
        
        return graph, new_edge_node_id, literal, endingNodeLabel

    def split_graphs_with_height(self, graph):

        # Step 0: Design to store split nodes and outputLabel nodes
        OutputLine_Map = {}
        
        # Step 1: Top graph of U1 nodes in threshold height
        top_graph_nodes = [
            n for n in graph.nodes
            if graph.nodes[n].get("distance") <= self.HeightThereshold
        ]
        # print('top_graph_nodes',[graph.nodes[n]['literal'] for n in top_graph_nodes])
        top_graph = self.get_full_top_graph(top_graph_nodes)
        
        #update node attributes for split ids
        for node in top_graph.nodes:
            top_graph.nodes[node].update(graph.nodes[node])

        # print('len(top_graph.nodes)',len(top_graph.nodes))

        # Step 2: Find the root nodes (U1)
        root_nodes = [n for n in top_graph.nodes if top_graph.in_degree(n) == 0 and top_graph.nodes[n].get("BipartitePart") == "U1"]
        # print('root_nodes',root_nodes)
        # for root_node in root_nodes:
        #     print('root_node',top_graph.nodes[root_node]['literal'])

        # Step 3: Find split_nodes(u1) in top_graph
        pred_split_nodes, succ_split_nodes = self.find_split_nodes(top_graph)

        # Step 4: Find output label nodes(u1) in top_graph
        output_label_nodes = self.find_output_label_nodes(top_graph)
        for output_label_node in output_label_nodes:
            node_id = output_label_node["node_id"]
            literal = output_label_node["literal"]
            ExpressionRoot = output_label_node["ExpressionRoot"]
            top_graph, added_output_node_id, literal, endingNodeLabel = self.add_expression_output_node(top_graph, node_id)

            OutputLine_Map[literal] = endingNodeLabel
        
        
        # print('output_label_nodes', output_label_nodes)
        # print('succ_split_nodes',succ_split_nodes)
        
        # Step 6.1: Intitalising pre-requisive node successor of U1 nodes
        u1_succ_split_nodes = {}
        u1_nodes_set = {node for node in succ_split_nodes if top_graph.nodes[node].get("BipartitePart") == "U1"}
        u2_nodes_set = {node for node in succ_split_nodes if top_graph.nodes[node].get("BipartitePart") == "U2"}

        # print('u1_nodes_set', u1_nodes_set)
        # print('u2_nodes_set', u2_nodes_set)
        
        # Step 6.2: Add U1 nodes add its successors
        for u1_node in u1_nodes_set:
            u1_succ_split_nodes[u1_node] = succ_split_nodes[u1_node]
        
        # Step 6.3: Finding set of nodes have same successor do for U1 nodes only
        inverse_map_graph_chunk_1 = defaultdict(list)
        for u1_node, u2_nodes in u1_succ_split_nodes.items():
            for u2_node in u2_nodes:
                inverse_map_graph_chunk_1[u2_node].append(u1_node)
                
        # Step 6.4: Add Successors of U2 nodes (adding U1 nodes) to inverse_map_graph_chunk_2 dictionary //good
        inverse_map_graph_chunk_2 = {}
        for u2_node in u2_nodes_set:
            # inverse_map_graph_chunk_2[u2_node] = [u1_node for u1_node in self.Graph.successors(u2_node)]
            inverse_map_graph_chunk_2[u2_node] = succ_split_nodes[u2_node]

        # print("Step 6:", inverse_map_graph_chunk_2, inverse_map_graph_chunk_1)

        # Step 5: Add predicessor of U2 nodes (adding U1 nodes) to top_graph
        # u2_nodes_set = {node for node in pred_split_nodes if top_graph.nodes[node].get("BipartitePart") == "U2"}
        # top_graph_nodes_set = set(top_graph.nodes)
        
        # for u2_node in u2_nodes_set:
        #     for u1_pred in self.Graph.predecessors(u2_node):
        #         if self.Graph.nodes[u1_pred].get("BipartitePart") == "U1":
        #             # Add the U1 node if not already in the top graph
        #             if u1_pred not in top_graph_nodes_set:
        #                 top_graph.add_node(u1_pred, **self.Graph.nodes[u1_pred])
        #                 top_graph_nodes_set.add(u1_pred)

        #             # Add the edge (even if node exists already)
        #             if not top_graph.has_edge(u1_pred, u2_node):
        #                 top_graph.add_edge(u1_pred, u2_node, **self.Graph.edges[u1_pred, u2_node])

        # Both inverse_map_graph_chunk_1 and inverse_map_graph_chunk_2 have U2 nodes(children) as keys U1 nodes(root or parent in split graph) as values
        # Used to calculate split nodes
        
        # Step 7: Update the included nodes of all the
        self.Included_nodes.update([node for node in top_graph.nodes])
        split_graph_included_nodes = self.Included_nodes.copy()
        
        # Step 8.1: Add output node for U1 node in the top graph
        output_node_to_literal_map = {}
        for u1_node in u1_nodes_set:  #list of values in the inverse_map_graph_chunk_1
            top_graph, added_output_node_id, literal, endingNodeLabel = self.add_output_node(top_graph, u1_node)

            #Check for split_id in the main graph
            if(self.Graph.nodes[u1_node].get("split_id") == None and self.Graph.in_degree(u1_node) != 0):
                self.Graph.nodes[u1_node]["split_id"] = str(uuid.uuid4()) # Generate a unique split ID
            split_id = self.Graph.nodes[u1_node]["split_id"]
            
            if u1_node not in output_node_to_literal_map:
                output_node_to_literal_map[u1_node] = []
            output_node_to_literal_map[u1_node].append({'split_id':split_id, 'literal':literal, 'added_output_node_id':added_output_node_id, 'endingNodeLabel':endingNodeLabel})

        
        # Step 8.2: Add output nodes for U2 node in the top graph
        for u2_node in u2_nodes_set:
            top_graph, added_output_node_id, literal, endingNodeLabel = self.add_output_nodes(top_graph, u2_node)

            #Check for split_id in the main graph
            if(self.Graph.nodes[u2_node].get("split_id") == None and self.Graph.in_degree(u2_node) != 0):
                self.Graph.nodes[u2_node]["split_id"] = str(uuid.uuid4()) # Generate a unique split ID
            split_id = self.Graph.nodes[u2_node]["split_id"]
            
            u1_nodes = succ_split_nodes[u2_node]
            # print('u2_node',u2_node,'u1_nodes',u1_nodes)
            for u1_node in u1_nodes:
                if u1_node not in output_node_to_literal_map:
                    output_node_to_literal_map[u1_node] = []
                output_node_to_literal_map[u1_node].append({'split_id':split_id, 'literal':literal, 'added_output_node_id':added_output_node_id, 'endingNodeLabel':endingNodeLabel})
                
                
            
        # Step 9: Create split graphs from inverse_map_graph_chunk_1 and inverse_map_graph_chunk_2
        split_graphs = []

        # print('inverse_map_graph_chunk_1',inverse_map_graph_chunk_1)
        # print('inverse_map_graph_chunk_2',inverse_map_graph_chunk_2)

        # Step 9.1: Create a fresh subgraph rooted at u1_nodes
        subgraph_nodes = set()
        split_graph_present = False  # Create a flag to track graph creation

        # Step 9.2: split_nodes that have successor connections from U1 node (replication U1 node in split_graph)
        for u2_child, u1_parents in inverse_map_graph_chunk_1.items():
            
            # Step 9.2.1: include the U1 root and its U2 child
            for u1_node in u1_parents:
                subgraph_nodes.add(u1_node)
            split_graph_present = True  # Flag for nodes in subgraph_nodes
                
            # Step 9.2.2: Find descendants of the U2 node in the full graph (excluding already included nodes)
            pending = deque([u2_child])
            while pending:
                current = pending.popleft()
                subgraph_nodes.add(current)
                for child in self.Graph.successors(current):
                    # print('child1', child)
                    # print(child not in split_graph_included_nodes)
                    # print(child not in subgraph_nodes)
                    if child not in split_graph_included_nodes:
                        # print('child2', child)
                        pending.append(child)

            # Step 9.2.3: Record included nodes of split graph
            split_graph_included_nodes.update(subgraph_nodes)

        # print(123456)

        # Step 9.3: split_nodes that have successor connections from U2 node (only include successor U1 node of U2_parent)
        for u2_parent, u1_children in inverse_map_graph_chunk_2.items():

            # Step 9.3.1: include the U1 children
            pending = deque([])
            for u1_child in u1_children:
                subgraph_nodes.add(u1_child)
                pending.append(u1_child)
            split_graph_present = True # Flag for nodes in subgraph_nodes
                
            # Step 9.3.2: Find descendants of the U1 node in the full graph (excluding already included nodes)
            while pending:
                current = pending.popleft()
                subgraph_nodes.add(current)
                for child in self.Graph.successors(current):
                    # print('child1', child)
                    # print(child not in split_graph_included_nodes)
                    # print(child not in subgraph_nodes)
                    if child not in split_graph_included_nodes and child not in subgraph_nodes:
                        # print('child2', child)
                        pending.append(child)

            # Step 9.3.3: Record included nodes of split graph
            split_graph_included_nodes.update(subgraph_nodes)
            
        # Step 9.4: Create the split_graph from the subgraph nodes
        split_graph = self.Graph.subgraph(subgraph_nodes).copy()
        
        # Deep copy node attributes to avoid shared references
        for node in split_graph.nodes:
            # print('11',split_graph.nodes[node]['in_split_id'])
            split_graph.nodes[node]['in_split_id'] = copy.deepcopy(graph.nodes[node]['in_split_id'])
            # print('22',split_graph.nodes[node]['in_split_id'])
            # print('-----------------------------------------------------------------------')

        # Step 10: Update the split_ids based on the bus connections between top graph and splt_graph
        for u2_parent, u1_children in inverse_map_graph_chunk_2.items():
            # Step 10.1: Assign split_id's based on the output_node_to_literal_map (Step 8)
            # print('u2_parent',u2_parent, 'u1_children',u1_children)
            for u1_child in u1_children:
                for u1_map_item in output_node_to_literal_map[u1_child]:
                    split_id, literal, added_output_node_id, endingNodeLabel = u1_map_item['split_id'], u1_map_item['literal'], u1_map_item['added_output_node_id'], u1_map_item['endingNodeLabel']

                    OutputLine_Map[literal] = split_id # Update output line mapping

                    # Step 10.2: Tag nodes with split ID
                    top_graph.nodes[added_output_node_id]["out_split_id"] = split_id  #one
                    split_graph.nodes[u1_child]["in_split_id"].add(split_id) #many

        for u2_child, u1_parents in inverse_map_graph_chunk_1.items():
            # Step 10.3: Assign split_id's based on the output_node_to_literal_map (Step 8)
            for u1_node in u1_parents:
                for u1_map_item in output_node_to_literal_map[u1_node]:
                    split_id, literal, added_output_node_id, endingNodeLabel = u1_map_item['split_id'], u1_map_item['literal'], u1_map_item['added_output_node_id'], u1_map_item['endingNodeLabel']

                    OutputLine_Map[literal] = split_id # Update output line mapping

                    # Step 10.4: Tag nodes with split ID
                    top_graph.nodes[added_output_node_id]["out_split_id"] = split_id  #one
                    split_graph.nodes[u1_node]["in_split_id"].add(split_id) #many

        #  Step 11: Save the split_graph if the graph is present
        if(split_graph_present):
            split_graphs.append(split_graph)
            
        # Debug print
        # if(split_graph_present):
        #     print('---------------------------------------------------------------')
        #     print('split_graph2', [split_graph.nodes[node]['literal'] for node in split_graph.nodes])

        return top_graph, split_graphs, OutputLine_Map

    
    def GraphSplittingWithHeightConstraint(self, height=None):

        if(height is not None):
            self.HeightThereshold = height
            
        #Assign split_id to all the nodes
        for node in self.Graph.nodes:
            self.Graph.nodes[node]['split_id'] = None
            self.Graph.nodes[node]['out_split_id'] = None
            self.Graph.nodes[node]['in_split_id'] = set()
            
        unprocessed_graphs = [self.Graph]
        while(unprocessed_graphs):
            unprocessed_graph = unprocessed_graphs.pop(0)
            measured_graph = self.getDistanceFromRoot(unprocessed_graph) #attribute to each node has ditance from root to each node

            # print('measured_graph as ds',{measured_graph.nodes[n]['literal']:measured_graph.nodes[n]['distance'] for n in measured_graph if measured_graph.nodes[n]['BipartitePart']=="U1"})
            #split_graphs has wordLineID in root node or start
            #processed_graph has wprdLineID in leaf or end
            processed_graph, split_graphs, OutputLine_Map = self.split_graphs_with_height(measured_graph)

            # print('root node ids of processed_graph1', [n for n in processed_graph.nodes if processed_graph.in_degree(n) == 0 and processed_graph.nodes[n].get("BipartitePart") == "U1"])
            
            # split-id nodes
            in_split_ids = set()
            for node in processed_graph:
                for split_id in processed_graph.nodes[node]['in_split_id']:
                    in_split_ids.add(split_id)
            
            # print('in_split_ids',in_split_ids)

            # # Leaf nodes: out-degree == 0
            # leaf_nodes = [n for n, d in processed_graph.out_degree() if d == 0]
            # print("Leaf nodes:", leaf_nodes)
            # for leaf in leaf_nodes:
            #     print(f"{leaf} → {processed_graph.nodes[leaf]}")

            self.Processed_height_constraint_graphs_Map[frozenset(in_split_ids)] = ({'processed_graph':processed_graph,'OutputLine_Map':OutputLine_Map})
            
            unprocessed_graphs.extend(split_graphs)
            # print()
            # break
            
        #traverse through the graph and split the graph where the height constraing fails

        self.GraphProcessPhase = "4. Graph Splitting with Height constraint"

    def get_output_node_dependencies(self, graph):
        output_node_dependencies_map = {}
        
        # Step 1: Find output nodes (nodes with out-degree 0)
        output_nodes = [node for node in graph.nodes if graph.out_degree(node) == 0]
        
        for output_node in output_nodes:
            # Step 2: Find all ancestors (predecessors) of the output node
            ancestors = nx.ancestors(graph, output_node)
            # Optionally include the output node itself
            ancestors.add(output_node)
            # Get the literal of the output node as key
            output_literal = graph.nodes[output_node].get("literal", str(output_node))
    
            u1_nodes = {ancestor for ancestor in ancestors if graph.nodes[ancestor]['BipartitePart']=='U1'}
            u2_nodes = {ancestor for ancestor in ancestors if graph.nodes[ancestor]['BipartitePart']=='U2'}
            
            output_node_dependencies_map[output_literal] = (u1_nodes, u2_nodes)
        
        return output_node_dependencies_map
    
    def greedy_merge_dependency_sets(self, dependency_map, H):
        # Step 1: Convert dependency map to list of (key, all_nodes_set)
        sets_list = []
        for key, (u1, u2) in dependency_map.items():
            sets_list.append((key, u1, u2))
    
        # Step 2: Sort by size descending (or overlap potential, if available)
        sets_list.sort(key=lambda x: -len(x[2]))
    
        merged_groups = []
        used_keys = set()
    
        i = 0
        while(i<len(sets_list)):
            (key_i, set_i_u1, set_i_u2) = sets_list[i]
            if key_i in used_keys:
                i += 1
                continue
            
            merged_u1 = set_i_u1.copy()
            merged_u2 = set_i_u2.copy()
            group_keys = [key_i]
            used_keys.add(key_i)
            
            j = i + 1
            while(j<len(sets_list)):
                (key_j, set_j_u1, set_j_u2) = sets_list[j]
                if key_j in used_keys:
                    j += 1
                    continue
                if len(merged_u1 | set_j_u1) <= H and len(merged_u2 | set_j_u2) <= H:
                    merged_u1 |= set_j_u1
                    merged_u2 |= set_j_u2
                    group_keys.append(key_j)
                    used_keys.add(key_j)
                j += 1
                
            merged_groups.append((group_keys, merged_u1, merged_u2))
            i += 1
    
        return merged_groups

    def GraphSplittinWithSizeConstraint(self, size=None):

        if(size is not None):
           self.SizeThereshold = size

        for design_ID in self.Processed_height_constraint_graphs_Map:
            processed_graph = self.Processed_height_constraint_graphs_Map[design_ID]['processed_graph']
            OutputLine_Map = self.Processed_height_constraint_graphs_Map[design_ID]['OutputLine_Map']

            # divide the processed_graph into processed_group_graph
            output_node_dependencies_map = self.get_output_node_dependencies(processed_graph)
            merged_groups = self.greedy_merge_dependency_sets(output_node_dependencies_map, self.SizeThereshold)

            for idx, (group_keys, merged_set_u1, merged_set_u2) in enumerate(merged_groups):
                OutputLine_group_Map = {group_key:OutputLine_Map[group_key] for group_key in group_keys}
                all_nodes_in_group = merged_set_u1 | merged_set_u2

                processed_group_graph = processed_graph.subgraph(all_nodes_in_group).copy()

                #create design_id_group with idx and design_id
                design_ID_group = (design_ID, idx)
                self.Topological_order.append(design_ID_group)
                self.Processed_group_graphs_Map[design_ID_group] = ({'processed_group_graph':processed_group_graph,'OutputLine_group_Map':OutputLine_group_Map})

        self.GraphProcessPhase = "5. Graph Splitting with Crossbar size constraint"

        

    def CrossbarDesignRelalization(self):

        for design_ID_group in self.Processed_group_graphs_Map:
            processed_group_graph = self.Processed_group_graphs_Map[design_ID_group]['processed_group_graph']
            OutputLine_group_Map = self.Processed_group_graphs_Map[design_ID_group]['OutputLine_group_Map']
            
            colMap, rowMap, bit_line_counter  = {}, {}, 0
            word_lines = []
            for node in processed_group_graph.nodes:
                if(processed_group_graph.nodes[node]['BipartitePart']=='U2'):
                    colMap[processed_group_graph.nodes[node]['ID']] = bit_line_counter
                    bit_line_counter += 1
                if(processed_group_graph.nodes[node]['BipartitePart']=='U1'):
                    rowMap[processed_group_graph.nodes[node]['ID']] = int(processed_group_graph.nodes[node]['literal'].split('(')[0])
                    word_lines.append(int(processed_group_graph.nodes[node]['literal'].split('(')[0]))

            word_lines_count = len(word_lines)
            word_lines.sort()
            word_lines_map = {word_line:i for i, word_line in enumerate(word_lines)}
            # print(word_lines)
            for row_key in rowMap:
                rowMap[row_key] = word_lines_map[rowMap[row_key]]
            
            # print()
            # print('======================')
            # print(colMap, rowMap)
            # print('======================')
            # print()

            OutputLine_group_selectorLines_Map = {}
            output_node_dependencies_map_temp = self.get_output_node_dependencies(processed_group_graph)
            for output_node_label in output_node_dependencies_map_temp:
                u2_node_ids = output_node_dependencies_map_temp[output_node_label][1]
                OutputLine_group_selectorLines_Map[output_node_label] = [colMap[u2_node_id] for u2_node_id in u2_node_ids]
            self.Processed_group_graphs_Map[design_ID_group]['OutputLine_group_selectorLines_Map'] = OutputLine_group_selectorLines_Map
            
            #Setting dimention of BDD
            self.Processed_group_graphs_Map[design_ID_group]['BDD_dimentions'] = f"{word_lines_count} x {bit_line_counter}"

            #Setting selectorLine labels
            self.Processed_group_graphs_Map[design_ID_group]['Selector_Lines_Map'] = [f"C{i+1}" for i in range(bit_line_counter)]
            for col in colMap:
                self.Processed_group_graphs_Map[design_ID_group]['Selector_Lines_Map'][colMap[col]] = processed_group_graph.nodes[col]['literal']

            #Initialising crossbar design
            self.Processed_group_graphs_Map[design_ID_group]['Crossbar_design'] = [[0 for _ in range(bit_line_counter)] for _ in range(word_lines_count)]

            self.Processed_group_graphs_Map[design_ID_group]['DesignIdItemToWordLineInputMap'] = {} #{design_id_item:wordlinenum}

            input_split_IDs, _ = design_ID_group

            for input_split_id in input_split_IDs:
                input_node_id_templst = [node for node in processed_group_graph.nodes if input_split_id in processed_group_graph.nodes[node]["in_split_id"]]
                # if(len(input_node_id_templst)!=1):
                #     print('not thought', input_node_id_templst)
                #     continue
                # input_node_id = input_node_id_templst[0]
                # if(len(input_node_id_templst)>1):

                if(input_split_id not in self.Processed_group_graphs_Map[design_ID_group]['DesignIdItemToWordLineInputMap']):
                    self.Processed_group_graphs_Map[design_ID_group]['DesignIdItemToWordLineInputMap'][input_split_id] = []
                    
                for input_node_id in input_node_id_templst:            
                    self.Processed_group_graphs_Map[design_ID_group]['DesignIdItemToWordLineInputMap'][input_split_id].append(rowMap[input_node_id])
                    # break
                
            
            for i,(u, v, data) in enumerate(processed_group_graph.edges(data=True)):
                if(processed_group_graph.nodes[u]['BipartitePart']=='U2'):
                    row_i = rowMap[processed_group_graph.nodes[v]['ID']]
                    col_j = colMap[processed_group_graph.nodes[u]['ID']]
                    
                else:
                    row_i = rowMap[processed_group_graph.nodes[u]['ID']]
                    col_j = colMap[processed_group_graph.nodes[v]['ID']]
                
                self.Processed_group_graphs_Map[design_ID_group]['Crossbar_design'][row_i][col_j] = 1

            self.Processed_group_graphs_Map[design_ID_group]['LongestPaths'] = []
    
            longPaths = self.LongestpathInTreeAndCrossbar(processed_group_graph)

            # for longPath in longPaths:
            #     print('longPath', [processed_group_graph.nodes[node]['literal'] for node in longPath])
            
            CrossbarLongPaths = []
            for longPath in longPaths:
                CrossbarLongPath = []
                for j in range(len(longPath)-1):
                    if(j%2==0):
                        node1, node2 = longPath[j],longPath[j+1]
                    else:
                        node2, node1 = longPath[j],longPath[j+1]
                    row_index, col_index = rowMap[node1], colMap[node2]
                    
                    CrossbarLongPath.append((row_index, col_index))
                CrossbarLongPaths.append(CrossbarLongPath)

            self.Processed_group_graphs_Map[design_ID_group]['LongestPaths']  = CrossbarLongPaths
            
            disabledOutputsToInputs = self.LargestDisjointPathInDAG(processed_group_graph)

            Crossbar_disabledOutputsToInputs = {}
            disjoint_cells_to_be_added = []
            for outputLabel in disabledOutputsToInputs:
                disable_input_lines = disabledOutputsToInputs[outputLabel]['disable_input_lines']
                disable_selector_lines = disabledOutputsToInputs[outputLabel]['disable_selector_lines']
                disjoint_nodes_to_be_added = disabledOutputsToInputs[outputLabel]['disjoint_nodes_to_be_added']
                input_connected_nodes = disabledOutputsToInputs[outputLabel]['input_connected_nodes']
                output_connected_nodes = disabledOutputsToInputs[outputLabel]['output_connected_nodes']
                
                crossbar_disable_input_lines = []
                for node in disable_input_lines:
                    row_index = rowMap[node]
                    crossbar_disable_input_lines.append(row_index)

                crossbar_disable_selector_lines = []
                for node in disable_selector_lines:
                    col_index = colMap[node]
                    crossbar_disable_selector_lines.append(col_index)

                disjoint_nodes_to_be_added.sort(key = lambda x:(x[0],x[1]))

                disabled_cols = crossbar_disable_selector_lines.copy()

                # print('disabled_cols',disabled_cols)
                # print('disjoint_nodes_to_be_added',disjoint_nodes_to_be_added)

                for u1_cell, u2_cell in disjoint_nodes_to_be_added:
                    disjoint_cells_to_be_added.append((rowMap[u1_cell], colMap[u2_cell]))

                # print('disjoint_cells_to_be_added',disjoint_cells_to_be_added)
                # print()
                    
                Crossbar_disabledOutputsToInputs[outputLabel] = {
                    'disable_input_lines':crossbar_disable_input_lines, 
                    'disable_selector_lines':crossbar_disable_selector_lines,
                    'disjoint_cells_to_be_added':disjoint_cells_to_be_added
                }
                
            self.Processed_group_graphs_Map[design_ID_group]['Crossbar_disabledOutputsToInputs']  = Crossbar_disabledOutputsToInputs
            
        self.GraphProcessPhase = "5. Crossbar Realization"

    def LargestDisjointPathInDAG(self, graph):
        # print('LargestDisjointPathInDAG')
    
        input_nodes = {n for n, deg in graph.in_degree() if deg == 0}
        for node in graph.nodes:
            if len(graph.nodes[node].get('in_split_id', [])) > 0:
                input_nodes.add(node)
    
        output_nodes = {node for node in graph.nodes if 'O' in graph.nodes[node].get('literal', '')}
        # print("Output Nodes:", output_nodes)

        disabledOutputsToInputs = {}
    
        for output in output_nodes:
            # print(f"\nProcessing Output Node: {output}")
    
            reached_from_input = set(input_nodes)
            reached_from_output = set([output])
    
            queue_input = deque(input_nodes)
            queue_output = deque([output])
    
            # Track newly added in each layer
            next_input_layer = set(input_nodes)
            next_output_layer = set([output])
    
            intersecting_nodes = set()
            visited_input = set(input_nodes)
            visited_output = set([output])

            disjoint_nodes_to_be_added = []
    
            while queue_input or queue_output:
                # Process one input layer
                # if(len(next_input_layer)<len(next_output_layer)):
                if(len(next_input_layer)<len(next_output_layer) or len(queue_output)==0):
                    current_input_layer = list(queue_input)
                    queue_input.clear()
                    for node in current_input_layer:
                        for neighbor in graph.successors(node):
                            if neighbor not in visited_input:
                                visited_input.add(neighbor)
                                queue_input.append(neighbor)
                                next_input_layer.add(neighbor)
    
                # Process one output layer
                current_output_layer = list(queue_output)
                queue_output.clear()
                for node in current_output_layer:
                    for pred in graph.predecessors(node):
                        if pred not in visited_output:
                            visited_output.add(pred)
                            queue_output.append(pred)
                            next_output_layer.add(pred)

                intersection_nodes = next_input_layer & next_output_layer

                # print('len(next_input_layer)',len(next_input_layer))
                # print('len(next_output_layer)',len(next_output_layer))
                # print('len(queue_input)',len(queue_input))
                # print('len(queue_output)',len(queue_output))
                # print('len(intersection_nodes)',len(intersection_nodes))
                # print()
    
                # Check for intersection between current layers
                # intersection_nodes = next_input_layer & next_output_layer
                intersection = {node for node in intersection_nodes if graph.nodes[node]['BipartitePart']=='U2'}
                if intersection:
                    intersecting_nodes.update(intersection)

                    #remove the connection of predessesor to inyteraction node and send that node cell so that it can be set to high resistive state
                    for node in intersection:
                        for parent in graph.predecessors(node): # find parents of that node - add a disjoint child node to parents
                            disjoint_nodes_to_be_added.append((parent, node))

                    # Remove intersecting nodes from next queues to avoid further traversal
                    queue_input = deque([node for node in queue_input if node not in intersection])
                    queue_output = deque([node for node in queue_output if node not in intersection])
                    
                next_input_layer -= intersection
                # next_output_layer -= intersection

            outputNodeLabel = graph.nodes[output].get('literal', '')

            # print('--------------------------------------------------------')

            #inputs that need to be disabled
            graph_copy = copy.deepcopy(graph)
            disable_input_lst = self.find_input_nodes_of_paths(graph_copy, input_nodes, output, disjoint_nodes_to_be_added)

            # print('disable_input_lst',disable_input_lst)

            disabledOutputsToInputs[outputNodeLabel] = {"disable_selector_lines":intersecting_nodes, 
                                                        "disable_input_lines":disable_input_lst, 
                                                        "disjoint_nodes_to_be_added":disjoint_nodes_to_be_added,
                                                        "input_connected_nodes":next_input_layer,
                                                        "output_connected_nodes":next_output_layer,
                                                       }

            current_input_nodes = list(set(input_nodes) - set(disable_input_lst))

            # Verify disconnection
            graph_copy = copy.deepcopy(graph)
            is_disconnected, problemPath = self.verify_disconnection(graph_copy, current_input_nodes, output, disjoint_nodes_to_be_added)

            if not is_disconnected:
                print(problemPath)
                raise RuntimeError(
                    f"[ERROR] Output node {graph.nodes[output]['literal']} is still reachable from at least one input after removing intersecting nodes: {intersecting_nodes}"
                )

        return disabledOutputsToInputs

    def find_input_nodes_of_paths(self, graph, input_nodes, output_node, disabled_edges):
        # Remove the intersecting (disabled) nodes from the graph copy
        # graph.remove_nodes_from(disabled_nodes)

        graph.remove_edges_from(disabled_edges)

        disable_inputs_lst = []
        # Check if any input node still has a path to the output node
        for input_node in input_nodes:
            if nx.has_path(graph, input_node, output_node):
                path = nx.shortest_path(graph, input_node, output_node)
                disable_inputs_lst.append(path[0])
        return disable_inputs_lst
                
        
    def verify_disconnection(self, graph, input_nodes, output_node, disabled_edges):
        # Remove the intersecting (disabled) nodes from the graph copy
        # graph.remove_nodes_from(disabled_nodes)
        graph.remove_edges_from(disabled_edges)
    
        # Check if any input node still has a path to the output node
        for input_node in input_nodes:
            if nx.has_path(graph, input_node, output_node):
                path = nx.shortest_path(graph, input_node, output_node)
                return False, path  # A path still exists
        return True, []  # Fully disconnected

        
    def LongestpathInTreeAndCrossbar(self, graph):

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

        # Get all the start nodes
        start_nodes = {n for n, deg in graph.in_degree() if deg == 0}

        for node in graph.nodes:
            if(len(graph.nodes[node]['in_split_id'])>0):
                start_nodes.add(node)

        longest_paths = []
        for start_node in start_nodes:
            # 2) Compute the distance from start_node to all other nodes
            # Compute distances and retrieve ONE longest path
            distance_dict, single_longest_path, longest_distance = dag_longest_path_lengths(graph, start_node)
            longest_paths.append(single_longest_path)

        return longest_paths