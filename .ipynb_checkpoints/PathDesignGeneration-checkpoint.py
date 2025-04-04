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

class PATH:
    def __init__(self, CrossbarGridSize = 16):

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

        self.output_node_index = 0
        
        self.Processed_graphs_Map = {}

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

    def get_longest_distances_from_root(self, graph, root):
        longest_distances = {node: float('-inf') for node in graph.nodes}
        longest_distances[root] = 0
    
        for node in nx.topological_sort(graph):
            for neighbor in graph.successors(node):
                longest_distances[neighbor] = max(
                    longest_distances[neighbor],
                    longest_distances[node] + 1
                )
    
        return longest_distances

    def get_shortest_distances_from_root(self, graph, root):
        longest_distances = {node: float('-inf') for node in graph.nodes}
        longest_distances[root] = 0
    
        for node in nx.topological_sort(graph):
            for neighbor in graph.successors(node):
                longest_distances[neighbor] = min(
                    longest_distances[neighbor],
                    longest_distances[node] + 1
                )
    
        return longest_distances

    def getDistanceFromRoot(self, graph, root=None):
        # Create a deepcopy so original graph is not modified
        graph_copy = copy.deepcopy(graph)
    
        # If root is not given, find a node with in-degree 0 (assuming a DAG)
        if root is None:
            root_candidates = [n for n, d in graph_copy.in_degree() if d == 0]
            if not root_candidates:
                raise ValueError("No root node found (in-degree 0)")
            root = root_candidates[0]  # Use the first one found
    
        # Calculate shortest path lengths from the root
        # distances = nx.single_source_shortest_path_length(graph_copy, root)
        # distances = self.get_longest_distances_from_root(graph_copy, root)
        distances = self.get_shortest_distances_from_root(graph_copy, root)
    
        # Set distance as an attribute for each node
        for node, dist in distances.items():
            graph_copy.nodes[node]['distance'] = dist
    
        return graph_copy

    def split_graphs_with_height(self, graph, max_height=10):
            
        # Nodes within max height
        top_nodes = [n for n, data in graph.nodes(data=True) if data.get('distance') <= max_height]
    
        top_graph = graph.subgraph(top_nodes).copy()

        top_graph_root_node = [n for n, d in top_graph.in_degree() if d == 0][0]

        if(top_graph.nodes[top_graph_root_node]['split_id'] is None):
            design_id_key = top_graph.nodes[top_graph_root_node]['split_id']
        elif(top_graph.nodes[top_graph_root_node]['split_id'].split()[0]=="root"):
            design_id_key = top_graph.nodes[top_graph_root_node]['split_id'].split()[1]
            
        # Nodes that exceed the height
        overflow_nodes = sorted([n for n, data in graph.nodes(data=True) if data.get('distance') >= max_height and data.get('ExpressionRoot') is None], 
                               key=lambda n: graph.nodes[n].get('distance'))
    
        split_graphs = []
        visited = set()

        # print('overflow_nodes',overflow_nodes)
    
        for node in overflow_nodes:
            if node in visited:
                continue
    
            # Find nearest ancestor at max_height
            current = node
            split_root = None
            while True:
                preds = list(graph.predecessors(current))
                if not preds:
                    break
                for parent in preds:
                    parent_distance = graph.nodes[parent].get('distance')
                    if parent_distance == max_height:
                        split_root = parent
                        break
                if split_root:
                    break
                current = preds[0]
    
            if split_root is None:
                continue  # Skip if no proper split point
    
            # Generate a unique ID to mark continuity
            split_id = str(uuid.uuid4())
    
            # Add split_id to split_root (leaf in top_graph)
            if split_root in top_graph.nodes:
                top_graph.nodes[split_root]['split_id'] = f"leaf {split_id}"
    
            # Get descendants of this node
            descendants = nx.descendants(graph, split_root)
            descendants.add(split_root)
            subgraph_nodes = set(descendants)
    
            subgraph = graph.subgraph(subgraph_nodes).copy()
    
            # Add split_id to root of subgraph
            subgraph.nodes[split_root]['split_id'] = f"root {split_id}"
    
            visited.update(subgraph_nodes)
            split_graphs.append(subgraph)
    
        return split_graphs, top_graph, design_id_key
    
    def add_output_nodes(self, graph):
        #Add the output nodes to the BDD
        
        # Track the next available EdgeNode index
        edge_node_index = max([int(node.split("_")[-1]) for node in graph.nodes if "EdgeNode_" in str(node)]) + 1

        # print('edge_node_index',edge_node_index)
        
        OutputLine_Map = {}
        outputNodes = []
        
        for node in list(graph.nodes):
            node_data = graph.nodes[node]
            # print('node_data',node_data)

            FinalLeafNode = False
            SplitLeafNode = False
            
            if node_data.get("ExpressionRoot") is not None:
                FinalLeafNode = True
            elif node_data.get("split_id") is not None and node_data.get("split_id").split()[0]=="leaf":
                SplitLeafNode = True
            
            if FinalLeafNode or SplitLeafNode:  # If ExpressionRoot is not None
                new_edge_node_id = f"EdgeNode_{edge_node_index}"  # Create a new EdgeNode
        
                # Add the new node with '1' as the literal and 'U2' as BipartitePart
                graph.add_node(new_edge_node_id, ID=new_edge_node_id, literal="O"+str(self.output_node_index), BipartitePart="U2")
                
                # Connect the original node to the new EdgeNode
                graph.add_edge(node, new_edge_node_id)

                # print('node', node, node_data)
                # print('FinalLeafNode:',FinalLeafNode)
                # print('SplitLeafNode:',SplitLeafNode)
                # print('-----------------')
                
                if(FinalLeafNode):
                    OutputLine_Map["O"+str(self.output_node_index)] = graph.nodes[node].get('ExpressionRoot')
                elif(SplitLeafNode):
                    OutputLine_Map["O"+str(self.output_node_index)] = graph.nodes[node].get('split_id')
                
                outputNodes.append(new_edge_node_id)

                self.output_node_index += 1
                edge_node_index += 1  # Increment for the next node

        return graph, OutputLine_Map

    
    def GraphSplitting(self):
        #get distance of each node from root node

        for node in self.Graph.nodes:
            self.Graph.nodes[node]['split_id'] = None

        self.Design_id_to_tree_map = {}
            
        unprocessed_graphs = [self.Graph]
        
        processed_graphs_and_outputLine_Map = {}
        while(unprocessed_graphs):
            unprocessed_graph = unprocessed_graphs.pop(0)
            measured_graph = self.getDistanceFromRoot(unprocessed_graph) #attribute to each node has ditance from root to each node
            
            #split_graphs has wordLineID in root node or start
            #processed_graph has wprdLineID in leaf or end
            split_graphs, processed_graph, design_id_key = self.split_graphs_with_height(measured_graph)

            # print('len(split_graphs)',len(split_graphs))
            
            # # Root node: in-degree == 0
            # root_node = [n for n, d in processed_graph.in_degree() if d == 0][0]
            # print('root_node',root_node)
            # print(processed_graph.nodes[root_node])
            

            # # Leaf nodes: out-degree == 0
            # leaf_nodes = [n for n, d in processed_graph.out_degree() if d == 0]
            # print("Leaf nodes:", leaf_nodes)
            # for leaf in leaf_nodes:
            #     print(f"{leaf} → {processed_graph.nodes[leaf]}")
            
            processed_graph, OutputLine_Map = self.add_output_nodes(processed_graph)

            # print('OutputLine_Map',OutputLine_Map)

            # print('design_id_key',design_id_key)

            processed_graphs_and_outputLine_Map[design_id_key] = ({'processed_graph':processed_graph,'OutputLine_Map':OutputLine_Map})
            unprocessed_graphs.extend(split_graphs)

        self.Processed_graphs_Map = processed_graphs_and_outputLine_Map
            
        #traverse through the graph and split the graph where the height constraing fails

        self.GraphProcessPhase = "4. Graph Splitting"

    def CrossbarDesignRelalization(self):

        for Processed_graphs_Map_key in self.Processed_graphs_Map:
            processed_graph = self.Processed_graphs_Map[Processed_graphs_Map_key]['processed_graph']
            OutputLine_Map = self.Processed_graphs_Map[Processed_graphs_Map_key]['OutputLine_Map']
            
            colMap, rowMap, bit_line_counter  = {}, {}, 0
            word_lines = []
            for node in processed_graph.nodes:
                if(processed_graph.nodes[node]['BipartitePart']=='U2'):
                    colMap[processed_graph.nodes[node]['ID']] = bit_line_counter
                    bit_line_counter += 1
                if(processed_graph.nodes[node]['BipartitePart']=='U1'):
                    rowMap[processed_graph.nodes[node]['ID']] = processed_graph.nodes[node]['literal']
                    word_lines.append(processed_graph.nodes[node]['literal'])

            word_lines_count = len(word_lines)
            word_lines.sort()
            word_lines_map = {word_line:i for i, word_line in enumerate(word_lines)}
            print(word_lines_map)
            for row_key in rowMap:
                rowMap[row_key] = word_lines_map[rowMap[row_key]]
            
            print()
            print('======================')
            print(colMap, rowMap)
            print('======================')
            print()
            
            #Setting dimention of BDD
            self.Processed_graphs_Map[Processed_graphs_Map_key]['BDD_dimentions'] = f"{word_lines_count} x {bit_line_counter}"

            #Setting selectorLine labels
            self.Processed_graphs_Map[Processed_graphs_Map_key]['Selector_Lines_Map'] = [f"C{i+1}" for i in range(bit_line_counter)]
            for col in colMap:
                self.Processed_graphs_Map[Processed_graphs_Map_key]['Selector_Lines_Map'][colMap[col]] = processed_graph.nodes[col]['literal']

            #Initialising crossbar design
            self.Processed_graphs_Map[Processed_graphs_Map_key]['Crossbar_design'] = [[0 for _ in range(bit_line_counter)] for _ in range(word_lines_count)]

            for i,(u, v, data) in enumerate(processed_graph.edges(data=True)):
                print(u, processed_graph.nodes[u])
                print(v, processed_graph.nodes[v])
                print('-------------------')
            
            for i,(u, v, data) in enumerate(processed_graph.edges(data=True)):
                if(processed_graph.nodes[u]['BipartitePart']=='U2'):
                    row_i = rowMap[processed_graph.nodes[v]['ID']]
                    col_j = colMap[processed_graph.nodes[u]['ID']]
                    
                else:
                    row_i = rowMap[processed_graph.nodes[u]['ID']]
                    col_j = colMap[processed_graph.nodes[v]['ID']]
                
                self.Processed_graphs_Map[Processed_graphs_Map_key]['Crossbar_design'][row_i][col_j] = 1

            self.Processed_graphs_Map[Processed_graphs_Map_key]['LongestPath'] = []
    
            longPath = self.LongestpathInTreeAndCrossbar(processed_graph)

            CrossbarLongPath = []
            for j in range(len(longPath)-1):
                if(j%2==0):
                    node1, node2 = longPath[j],longPath[j+1]
                else:
                    node2, node1 = longPath[j],longPath[j+1]
                row_index, col_index = rowMap[node1], colMap[node2]
                
                CrossbarLongPath.append((row_index, col_index))
                    
            self.Processed_graphs_Map[Processed_graphs_Map_key]['LongestPath']  = CrossbarLongPath
            
        self.GraphProcessPhase = "5. Crossbar Realization"

    def LongestpathInTreeAndCrossbar(self, graph):
        
        start_node = [n for n, deg in graph.in_degree() if deg == 0][0]
        # print("Start nodes (in-degree = 0):", graph.nodes[start_node])

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
        distance_dict, single_longest_path, longest_distance = dag_longest_path_lengths(graph, start_node)

        return single_longest_path