# Data Journey DAG
# Version: 1.6.1
# Last Update: 2024/02/02
# Author: Tomio Kobayashi

# - generateProcesses  genProcesses() DONE
# - DAG check  checkDAG(from, to) DONE
# - Process coupling  (process1, process2) DONE
# - Link Node  linkFields(from, to) DONE
  
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random
from datetime import date
import textwrap
import re
from scipy.sparse import csr_matrix

class DataJourneyDAG:

    def __init__(self):
        self.vertex_names = []
        self.csr_matrix = []
        self.csr_matrix_T = []
        self.size_matrix = 0
        
        self.dic_vertex_names = {}
        self.dic_vertex_id = {}
        
        G = 0
        G_T = 0
        
        self.str_vertex_names = []
        self.str_csr_matrix = []
        self.str_csr_matrix_T = []
        self.str_size_matrix = 0
        
        self.str_dic_vertex_names = {}
        self.str_dic_vertex_id = {}
        
        str_G = 0
        str_G_T = 0
        
        self.dic_new2old = {}
        self.dic_old2new = {}

        self.set_complete = set()
        
        self.dic_conds = {}
        self.dic_opts = {}
        
    def csr_matrix_to_edge_list(self, csr_matrix):
        rows, cols = csr_matrix.nonzero()
        weights = csr_matrix.data
        edge_list = list(zip(rows, cols, weights))
        return edge_list
        
        
    def adjacency_matrix_to_edge_list(self, adc_matrix):
        edge_list = []
        num_nodes = len(adc_matrix)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if adc_matrix[i][j] >= 1:
                    edge_list.append((i, j, adc_matrix[i][j]))  # Add edges only for non-zero entries

        return edge_list
    
    def parameter_to_brightness(self, parameter_value):
        cmap = plt.cm.get_cmap('viridis')  # Choose a colormap
        brightness = cmap(parameter_value)[0]  # Extract first value (red channel)
        return brightness

    def draw_selected_vertices_reverse_proc(self, G, selected_vertices1, selected_vertices2, selected_vertices3, title, node_labels, 
                                            pos, wait_edges=None, reverse=False, figsize=(12, 8), showWeight=False, forStretch=False, excludeComp=False):

        comp_vertices = None
        subgraph_comp = None
        noncomp_vertices = None
        subgraph_noncomp = None
        if excludeComp:
            if reverse:
                excluded_tup = [(f[0], f[1]) for s in self.set_complete for f in list(nx.edge_dfs(G, source=self.dic_vertex_id[s]))]
            else:
                excluded_tup = [(f[0], f[1]) for s in self.set_complete for f in list(nx.edge_dfs(G, source=self.dic_vertex_id[s], orientation="reverse"))]
            excluded = set([f[0] for f in excluded_tup] + [f[1] for f in excluded_tup])
            comp_vertices = [s for s in selected_vertices1 if s in excluded]
            noncomp_vertices = [s for s in selected_vertices1 if s not in excluded]
            subgraph_comp = G.subgraph(comp_vertices)
            subgraph_noncomp = G.subgraph(noncomp_vertices)
            
        # Create a subgraph with only the selected vertices
        subgraph1 = G.subgraph(selected_vertices1)
        subgraph2 = G.subgraph(selected_vertices2)
        subgraph3 = G.subgraph(selected_vertices3)
        
        if reverse:
            subgraph1 = subgraph1.reverse()
            subgraph2 = subgraph2.reverse()
            subgraph3 = subgraph3.reverse()
            if excludeComp:
                subgraph_comp = subgraph_comp.reverse()
                subgraph_noncomp = subgraph_noncomp.reverse()

        # Set figure size to be larger
        plt.figure(figsize=figsize)

        # Set figure size to be larger
        node_labels = {node: "\n".join(["\n".join(textwrap.wrap(s, width=5)) for s in label.replace("_", "_\n").replace(" ", "\n").split("\n")]) for node, label in node_labels.items()}

        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0
        
        longest_path_length = None
        
        if showWeight:

            longest_path_length = None
            if not excludeComp:
                longest_path_length = nx.dag_longest_path_length(subgraph1)
            else:
                longest_path_length = nx.dag_longest_path_length(subgraph_noncomp)
                
            node_criticality = None
            if longest_path_length == 0:
                node_criticality = []
            else:
                if has_proc:
                    node_criticality = [((nx.dag_longest_path_length(subgraph1.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(subgraph1, target_node))])) + 
                                    nx.dag_longest_path_length(subgraph1.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(subgraph1, target_node, orientation='reverse'))]))) / 
                                    longest_path_length, target_node) for target_node in subgraph1.nodes() if self.dic_vertex_names[target_node][0:5] == "proc_"]
                else:
                    node_criticality = [((nx.dag_longest_path_length(subgraph1.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(subgraph1, target_node))])) + 
                                    nx.dag_longest_path_length(subgraph1.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(subgraph1, target_node, orientation='reverse'))]))) / 
                                    longest_path_length, target_node) for target_node in subgraph1.nodes()]
                
            node_parameter = {n[1]: round(n[0]*0.7+0.3, 3) for n in node_criticality} # make it between 0.2 and 1.0 to prevent too dark color
            if len(node_parameter) > 0:
#                 min_param = min(node_parameter.values())
#                 max_param = max(node_parameter.values())
#                 normalized_param = {node: (param - min_param) / (max_param - min_param) for node, param in node_parameter.items()}
                
                color_map = [plt.cm.viridis(node_parameter[node]) for node in subgraph2]

        if forStretch and len(selected_vertices1) > 10:
            max_height = max([v[1] for k, v in pos.items()])
            for k, v in pos.items():
                if v[0] % 3 == 1:
                    pos[k] = (v[0], v[1] + max_height * 0.025)
                if v[0] % 3 == 2:
                    pos[k] = (v[0], v[1] + max_height * 0.05)
                
        # Draw the graph
        if has_proc and forStretch:
            defFontSize=8
            defNodeSize=700
            defFontColor='grey'
        else:
            defFontSize=10
            defNodeSize=1000
            defFontColor='black'

        avg_duration = {}
        topological_order = None
        the_graph = None
        if showWeight:
#             self.showStats(subgraph1)
            # Find the topological order
            if not excludeComp:
                topological_order = list(nx.topological_sort(subgraph1))
                the_graph = subgraph1
            else:
                topological_order = list(nx.topological_sort(subgraph_noncomp))
                the_graph = subgraph_noncomp
                
    
            if reverse==False:
                weights = {}
                for i, j in the_graph.edges():
                    if j not in weights:
                        weights[j] = {}
                    weights[j][self.dic_vertex_names[i]] = the_graph[i][j]['weight']
                
                for t in topological_order:
                    if t not in weights:
                        avg_duration[t] = 0
                    else:
                        tot = 0
                        weight_params = {k: avg_duration[self.dic_vertex_id[k]] + v for k, v in weights[t].items()}
                        if self.dic_vertex_names[t] in self.dic_conds:
                            tot = logical_weight.calc_avg_result_weight(self.dic_conds[self.dic_vertex_names[t]], weight_params, opt_steps=self.dic_opts)
                        else:
                            tot = logical_weight.calc_avg_result_weight(" & ".join([k for k, v in weights[t].items()]), weight_params, opt_steps=self.dic_opts)
                        avg_duration[t] = tot
        if showWeight and reverse==False:
            node_labels1 = {k: v + "\n(" + str(round(avg_duration[k], 1)) + ")" for k, v in node_labels.items() if k in subgraph1 and k not in subgraph2 and k not in subgraph3}
            node_labels2 = {k: v + "\n(" + str(round(avg_duration[k], 1)) + ")" for k, v in node_labels.items() if k in subgraph2}
            node_labels3 = {k: v + "\n(" + str(round(avg_duration[k], 1)) + ")" for k, v in node_labels.items() if k in subgraph3}
        else:
            node_labels1 = {k: v for k, v in node_labels.items() if k in subgraph1 and k not in subgraph2 and k not in subgraph3}
            node_labels2 = {k: v for k, v in node_labels.items() if k in subgraph2}
            node_labels3 = {k: v for k, v in node_labels.items() if k in subgraph3}
        
        nx.draw(subgraph1, pos, linewidths=0, with_labels=True, labels=node_labels1, node_size=defNodeSize, node_color='skyblue', font_size=defFontSize, font_color=defFontColor, arrowsize=10, edgecolors='black')
        if has_proc and showWeight and len(node_parameter) > 0:
            nx.draw(subgraph2, pos, linewidths=0, with_labels=True, labels=node_labels2, node_size=defNodeSize, node_color=color_map, font_size=10, font_color='black', arrowsize=10, edgecolors='black')
        else:
            nx.draw(subgraph2, pos, linewidths=0, with_labels=True, labels=node_labels2, node_size=defNodeSize, node_color='orange', font_size=10, font_color='black', arrowsize=10, edgecolors='black')

        nx.draw(subgraph3, pos, linewidths=0, with_labels=True, labels=node_labels3, node_size=defNodeSize, node_color='pink', font_size=10, font_color='black', arrowsize=10, edgecolors='black')
        
        if excludeComp:
            nx.draw(subgraph_comp, pos, linewidths=0, with_labels=True, labels=node_labels1, node_size=defNodeSize, node_color='#DDDDDD', font_size=defFontSize, font_color=defFontColor, arrowsize=10, edgecolors='black')
    
        if showWeight:
            if wait_edges is None:
                edge_labels = {(i, j): subgraph1[i][j]['weight'] for i, j in subgraph1.edges()}
            else:
                dicWait = {(w[0], w[1]): w[2] for w in wait_edges}
                edge_labels = {(i, j): subgraph1[i][j]['weight'] if (i, j) not in dicWait else str(subgraph1[i][j]['weight']) + " (& " + str(dicWait[(i, j)]) + ")"  for i, j in subgraph1.edges()}
            nx.draw_networkx_edge_labels(subgraph1, pos, edge_labels=edge_labels)
        
            # Draw critical path edges with a different color
            
            longest_path = None
            if not excludeComp:
                longest_path_length = nx.dag_longest_path_length(subgraph1)
                longest_path = nx.dag_longest_path(subgraph1)  # Use NetworkX's built-in function
                critical_edges = [(longest_path[i], longest_path[i + 1]) for i in range(len(longest_path) - 1)]
                nx.draw_networkx_edges(subgraph1, pos, edgelist=critical_edges, edge_color='brown', width=1.25)
            else:
                longest_path_length = nx.dag_longest_path_length(subgraph_noncomp)
                longest_path = nx.dag_longest_path(subgraph_noncomp)  # Use NetworkX's built-in function
                critical_edges = [(longest_path[i], longest_path[i + 1]) for i in range(len(longest_path) - 1)]
                nx.draw_networkx_edges(subgraph_noncomp, pos, edgelist=critical_edges, edge_color='brown', width=1.25)

        if wait_edges is not None:
            wait_edge_list = [(w[0], w[1]) for w in wait_edges if w[2] < 5]
            nx.draw_networkx_edges(subgraph1, pos, edgelist=wait_edge_list, edge_color='aqua', width=1.00)
            wait_edge_list = [(w[0], w[1]) for w in wait_edges if w[2] >= 5]
            nx.draw_networkx_edges(subgraph1, pos, edgelist=wait_edge_list, edge_color='aqua', width=1.00)
            

        if forStretch:
            # Draw a vertical line at x=0.5 using matplotlib
            max_horizontal_pos = max([v[0] for k, v in pos.items()])

            for i in range(max_horizontal_pos, 0, -5):
    #             width = 0.5 if (max_horizontal_pos - i) % 10 == 0 else 0.25
                width = 0.25
                linestyle = "dashed" if (max_horizontal_pos - i) % 10 == 0 else "dotted"
                plt.axvline(x=i, color="orange", linestyle=linestyle, linewidth=width)

        if showWeight:

            self.showStats(subgraph1)
            # Find the topological order
#             topological_order = None
#             the_graph = None
#             if not excludeComp:
#                 topological_order = list(nx.topological_sort(subgraph1))
#                 the_graph = subgraph1
#             else:
#                 topological_order = list(nx.topological_sort(subgraph_noncomp))
#                 the_graph = subgraph_noncomp
            
            # Print the topological order
            print("TOPOLOGICAL ORDER:")
            print(" > ".join([self.dic_vertex_names[t] for t in topological_order if (has_proc and self.dic_vertex_names[t][0:5] == "proc_") or not has_proc]))

            print("")

            # Print the longest path and its length
            print("LONGEST PATH (" + str(longest_path_length) + "):")
            print(" > ".join([self.dic_vertex_names[t] for t in longest_path if (has_proc and self.dic_vertex_names[t][0:5] == "proc_") or not has_proc]))
            print("")
#             print("CRITICALITY:")
#             for n in sorted(node_criticality, reverse=True):
#                 print(self.dic_vertex_names[n[1]], round(n[0], 3))
#             print("")
            # Print the longest path and its length

    
            if reverse==False:
                print("AVERAGE COMPLETION USING CONDITIONS AND OPTIONAL PCT")
                print(", ".join([self.dic_vertex_names[t] + ": " + str(round(avg_duration[t], 3)) for t in topological_order]))
                print("")
    
            self.suggest_coupling(subgraph1)
            self.suggest_opportunities(subgraph1)

            
        plt.title(title)
        plt.show()

    def edge_list_to_csr_matrix(self, edges):
        rows = [edge[0] for edge in edges]
        cols = [edge[1] for edge in edges]
        weights = [edge[2] for edge in edges]
        num_nodes = max(max(rows), max(cols))+1
        matrix = csr_matrix((weights, (rows, cols)), shape=(num_nodes, num_nodes))
        return matrix

    
    def edge_list_to_adjacency_matrix(self, edges):

        num_nodes = max(max(edge) for edge in edges) + 1  # Determine the number of nodes
#         adjacency_matrix = np.array([np.array([0] * num_nodes) for _ in range(num_nodes)])
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for edge in edges:
            if len(edge) >= 3:
                adjacency_matrix[edge[0]][edge[1]] = edge[2]
            else:
                adjacency_matrix[edge[0]][edge[1]] = 1

        return adjacency_matrix

    def data_import(self, file_path, is_edge_list=False, is_oup_list=False):
#         Define rows as TO and columns as FROM
        adjacency_matrix = None
#         adjacency_matrix_T = None
        if is_oup_list:
            edges = self.read_oup_list_from_file(file_path)
            adjacency_matrix = self.edge_list_to_adjacency_matrix(edges)
        elif is_edge_list:
            edges = self.read_edge_list_from_file(file_path)
            adjacency_matrix = self.edge_list_to_adjacency_matrix(edges)
        else:
            data = np.genfromtxt(file_path, delimiter='\t', dtype=str, encoding=None)

            # Extract the names from the first record
            self.vertex_names = list(data[0])

            # Extract the adjacency matrix
            adjacency_matrix = data[1:]
        
        # Convert the adjacency matrix to a NumPy array of integers
        adjacency_matrix = np.array(adjacency_matrix, dtype=int)
                    
        if not nx.is_directed_acyclic_graph(nx.DiGraph(adjacency_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(adjacency_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return

#         # Generate random weights between 1 and 5 for testing
#         for i in range(len(adjacency_matrix)):
#             random_integer = random.randint(1, 5)
#             for j in range(len(adjacency_matrix)):
#                 if adjacency_matrix[i][j] == 1:
#                     adjacency_matrix[i][j] = random_integer

        
        self.csr_matrix = csr_matrix(adjacency_matrix)
        self.csr_matrix_T = self.csr_matrix.transpose()

        # Matrix of row=FROM, col=TO
        self.size_matrix = len(adjacency_matrix)

        self.G = nx.DiGraph(self.csr_matrix)
        self.G_T = nx.DiGraph(self.csr_matrix_T)
        
        for i in range(len(self.vertex_names)):
            self.dic_vertex_names[i] = self.vertex_names[i]
            self.dic_vertex_id[self.vertex_names[i]] = i

        # Find the largest connected component
        connected_components = list(nx.weakly_connected_components(self.G))
        
        largest_connected_component = None
        print("Sizes of Connected Graphs")
        print("---")
        sorted_graphs = sorted([[len(c), c] for c in connected_components], reverse=True)
        for i in range(len(sorted_graphs)):
            print(sorted_graphs[i][0])
            if i == 0:
                largest_connected_component = sorted_graphs[i][1]
            
        largest_G = self.G.subgraph(largest_connected_component)
        
        print("")
        # Show stats of procs
        print("Centrality Stats of the Largest Connected Component")
        print("---")
        self.showStats(largest_G)
            
    def write_edge_list_to_file(self, filename):
        
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)

        with open(filename, "w") as file:
            # Write headers
            file.write("\t".join(self.vertex_names) + "\n")
            for edge in edges:
#                 file.write(f"{edge[0]}\t{edge[1]}\n")
                file.write(f"{edge[0]}\t{edge[1]}\t{edge[2]}\n")
    
    def read_oup_list_from_file(self, filename):
#         Format is tab-delimited taxt file:
#             0 - Output Field Name
#             1 - Weight
#             2-n - Input Field Names
        dictf = {}
        dicfields = {}
        with open(filename, "r") as file:
            ini = True
            for line in file:
    #             print("line", line)
                ff = line.strip().split("\t")
                for i in [0] + [j for j in range(2, len(ff), 1)]:
                    if ff[i] not in dicfields:
                        dicfields[ff[i]] = len(dicfields)

                oup_ind = dicfields[ff[0]]
                if dicfields[ff[0]] not in dictf:
                    inp = [dicfields[ff[i]] for i in range(2, len(ff), 1)]
                    dictf[oup_ind] = (int(ff[1]), inp)
                else:
                    dictf[oup_ind][1].extend([dicfields[ff[i]] for i in range(2, len(ff), 1)]) 

    #     print("dictf", dictf)

        headers = [k[1] for k in sorted([(v, k) for k, v in dicfields.items()])]
        edges = [(vv, k, v[0]) for k, v in dictf.items() for vv in v[1]]

        self.vertex_names = headers
        return edges
    
    def read_complete_list_from_file(self, filename):
        with open(filename, "r") as file:
            for line in file:
                s = line.strip()
                self.set_complete.add(s)
                if s[0:5] != "proc_":
                    self.set_complete.add("proc_" + s)
                    
    def read_cond_list_from_file(self, filename):
        with open(filename, "r") as file:
            for line in file:
                conds = line.strip().split("\t")
                self.dic_conds[conds[0]] = conds[1]

    def read_opt_list_from_file(self, filename):
        with open(filename, "r") as file:
            for line in file:
                conds = line.strip().split("\t")
                self.dic_opts[conds[0]] = float(conds[1])
#         print("self.set_complete", self.set_complete)

    def read_edge_list_from_file(self, filename):
        edges = []
        with open(filename, "r") as file:
            ini = True
            for line in file:
                if ini:
                    self.vertex_names = line.strip().split("\t")
                    ini = False
                else:
                    source, target, weight = map(int, line.strip().split())
                    edges.append((source, target, weight))
        return edges

    def write_adjacency_matrix_to_file(self, filename):
        with open(filename, "w") as file:
            file.write("\t".join(self.vertex_names) + "\n") 
            for row in self.csr_matrix.toarray():
                file.write("\t".join(map(str, row)) + "\n")  # Join elements with tabs and add newline


    def genProcesses(self):
        
        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k[0:5] == "proc_"]) > 0
        if has_proc:
            print("Already contains processes.")
            return

                
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)
        new_edges = {}
        dicNewID = {}
        setVertex = set(self.vertex_names)


        dicEdgeWeight = {}
        for k in edges:
            if k[0] not in dicEdgeWeight or (k[0] in dicEdgeWeight and dicEdgeWeight[k[0]] < k[2]):
                dicEdgeWeight[k[0]] = k[2]
        for i in range(len(edges)):
#             new_vertex_name = "proc_" + self.dic_vertex_names[edges[i][0]] + self.dic_vertex_names[edges[i][1]]
            new_vertex_name = "proc_" + self.dic_vertex_names[edges[i][1]]
            new_id = 0
            if new_vertex_name not in setVertex:
                new_id = len(self.vertex_names)
                np.append(self.vertex_names, new_vertex_name)
                self.vertex_names.append(new_vertex_name)
                self.dic_vertex_names[new_id] = new_vertex_name
                self.dic_vertex_id[new_vertex_name] = new_id
                dicNewID[edges[i][1]] = new_id
                setVertex.add(new_vertex_name)
            else:
                new_id = dicNewID[edges[i][1]]
            w = 1
            if edges[i][1] in dicEdgeWeight: 
                w = dicEdgeWeight[edges[i][1]]
            if (edges[i][0], new_id) not in new_edges:
                new_edges[(edges[i][0], new_id)] = 1
            if (new_id, edges[i][1]) not in new_edges or ((new_id, edges[i][1]) in new_edges and w * 2 - 1 > new_edges[(new_id, edges[i][1])]):
                new_edges[(new_id, edges[i][1])] = w * 2 - 1
            
        new_edges = [(k[0], k[1], v) for k, v in new_edges.items()]
    
        self.csr_matrix = self.edge_list_to_csr_matrix(new_edges)
        self.csr_matrix_T = self.csr_matrix.transpose()
        self.size_matrix = self.csr_matrix.shape[0]
        
        
        self.G = nx.DiGraph(self.csr_matrix)
        self.G_T = nx.DiGraph(self.csr_matrix_T)
        
        # Show stats of procs
                
        print("")

        # Find the largest connected component
        connected_components = list(nx.weakly_connected_components(self.G))
        largest_connected_component = None
        sorted_graphs = sorted([[len(c), c] for c in connected_components], reverse=True)
        for i in range(len(sorted_graphs)):
            if i == 0:
                largest_connected_component = sorted_graphs[i][1]
        largest_G = self.G.subgraph(largest_connected_component)
        
        print("")
        # Show stats of procs
        print("Centrality Stats of the Largest Connected Component")
        print("---")
        self.showStats(largest_G)
        
    def coupleProcesses(self, proc1, proc2):
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)
        if proc1[0:5] != "proc_" or proc2[0:5] != "proc_":
            return
        new_edges = []
        old = self.dic_vertex_id[proc1]
        new = self.dic_vertex_id[proc2]
        
        lowers = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, old))])
        uppers = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, orientation='reverse'))])
        if new in lowers and new in uppers:
            print("Processes that lie on the same path cannot be coupled.", proc1, proc2)
            return
        
        newWeight = max([edges[i][2] for i in range(len(edges)) if edges[i][0] == old]) + max([edges[i][2] for i in range(len(edges)) if edges[i][0] == new])

        for i in range(len(edges)):
            id1 = edges[i][0]
            id2 = edges[i][1]
            weight = edges[i][2]
            if edges[i][0] == old:
                id1 = new
                weight = newWeight
            if edges[i][0] == new:
                id1 = new
                weight = newWeight
            if edges[i][1] == old:
                id2 = new
            
            new_edges.append((id1, id2, weight))
    
        tmp_matrix = self.edge_list_to_csr_matrix(new_edges)
                   
        if not nx.is_directed_acyclic_graph(nx.DiGraph(tmp_matrix)):       
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(tmp_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        else:
            self.csr_matrix = tmp_matrix
            self.csr_matrix_T = self.csr_matrix.transpose()
            self.size_matrix = self.csr_matrix.shape[0]
        
            self.G = nx.DiGraph(self.csr_matrix)
            self.G_T = nx.DiGraph(self.csr_matrix_T)

    def create_random_string_from_date(self):

        today = date.today()
        year_str = str(today.year)
        month_str = str(today.month).zfill(2)  # Pad month with leading zero if needed
        day_str = str(today.day).zfill(2)  # Pad day with leading zero if needed

        date_parts = list(year_str + month_str + day_str)
        random.shuffle(date_parts)  # Shuffle the digits randomly

        random_string = "".join(date_parts[:5])  # Take the first 5 digits
        return random_string


    def linkElements(self, element1, element2, weight, procName=""):
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)
        isProcessIncluded = (self.dic_vertex_names[edges[1][0]][0:5] == "proc_" or self.dic_vertex_names[edges[1][1]][0:5] == "proc_")
        if element1[0:5] == "proc_" or element1[0:5] == "proc_":
            return
        new_edges = copy.deepcopy(edges)
        if element1 not in self.dic_vertex_id or element2 not in self.dic_vertex_id:
            return
        id1 = self.dic_vertex_id[element1]
        id2 = self.dic_vertex_id[element2]
        
        newID = 99999
        if isProcessIncluded:
            if procName == "":
                procName = "proc_" + element2 + "_" + self.create_random_string_from_date()
            newID = max([max(e[0], e[1]) for e in edges]) + 1
            new_edges.append((id1, newID, 1))
            new_edges.append((newID, id2, weight))
            self.vertex_names.append(procName)
            self.dic_vertex_names[newID] = procName
            self.dic_vertex_id[procName] = newID
        else:
            new_edges.append((id1, id2, weight))
    
        tmp_matrix = self.edge_list_to_csr_matrix(new_edges)
        
#         if self.has_cycle(tmp_matrix):
        if not nx.is_directed_acyclic_graph(nx.DiGraph(tmp_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(tmp_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        else:
            self.csr_matrix_T = self.csr_matrix.transpose()
            self.size_matrix = self.csr_matrix.shape[0]
        
            self.G = nx.DiGraph(self.csr_matrix)
            self.G_T = nx.DiGraph(self.csr_matrix_T)
            if isProcessIncluded:
                self.vertex_names.append(procName)
                self.dic_vertex_names[newID] = procName
                self.dic_vertex_id[procName] = newID
            
            
    
    def drawOriginsStretchDummy(self, target_vertex, title="", figsize=None, showWeight=False):
        
        if isinstance(target_vertex, str):
            if target_vertex not in self.str_dic_vertex_id:
                print(target_vertex + " is not an element")
                return
            target_vertex = self.str_dic_vertex_id[target_vertex]
        
#         Draw the path FROM the target
        position = {}
        colpos = {}
        posfill = set()
        selected_vertices1 = set()
        selected_vertices2 = set()

            
        succs = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=self.dic_new2old[target_vertex], orientation="reverse"))])
        succs = [self.dic_vertex_names[s] for s in succs]
        pattern = re.compile(r'^dumm_(\d+)_')

                    
        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.str_csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.str_csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        
        res_vector = np.zeros((1, self.str_size_matrix))
        res_vector[0][target_vertex] = 1
        for i in range(50000):
            if sum(res_vector[i]) == 0:
                break
            res_vector.resize((res_vector.shape[0] + 1, res_vector.shape[1]))
            new_row = np.zeros(self.str_size_matrix)
            res_vector[-1, :] = new_row
            res_vector[i+1] = self.str_csr_matrix.dot(res_vector[i])
    
            for k in range(len(res_vector[i+1])):
                chkstr = re.sub(pattern, '', self.str_dic_vertex_names[k])
                if res_vector[i+1][k] > 0 and (chkstr not in succs and "proc_" + chkstr not in succs):
                    res_vector[i+1][k] = 0
                    continue

        for i in range(len(res_vector)):
            colpos[i] = 0
                    
#         This part is to find the starting steps.  Currently not used.
        succs2 = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=self.dic_new2old[target_vertex]))])
        succs2 = set([self.dic_old2new[s] for s in succs2.nodes()])
        succLastReached = {}
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0:
                    if self.str_dic_vertex_names[j] in succs:
                        succLastReached[j] = i
        last_pos = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            last_pos += 1
        print("WAITING STEPS:")
        for v in sorted([[v, k] for k, v in succLastReached.items()], reverse=True):
            start_step = last_pos-v[0]-1

            for e in self.G.out_edges(self.dic_new2old[v[1]]):
                if self.dic_old2new[e[1]] not in succLastReached:
                    continue
                if (last_pos - succLastReached[self.dic_old2new[e[1]]] - 1) - (start_step + self.G[e[0]][e[1]]["weight"]) - 1 > 0:
                    print(self.str_dic_vertex_names[v[1]] + " (" + str(start_step) + ") -> " + self.dic_vertex_names[e[1]] + " (" + str(start_step + self.G[e[0]][e[1]]["weight"]) + 
                          ") with wait " + str((last_pos - succLastReached[self.dic_old2new[e[1]]] - 1) - (start_step + self.G[e[0]][e[1]]["weight"] - 1)))

                    
                    
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0 and j != self.str_size_matrix: 
                    if self.str_dic_vertex_names[j][0:5] == "proc_":
                        selected_vertices2.add(j)
                        selected_vertices1.add(j)
                    else:
                        selected_vertices1.add(j)

#         initialize the positions
        last_pos = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            last_pos += 1
        done = False
        largest_j = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            nonzero = 0
            for j in range(len(res_vector[i])):
                if j not in selected_vertices1 and j not in selected_vertices2:
                    continue
                if res_vector[i][j]:
                    posfill.add(j)
                    position[j] = ((last_pos-i), colpos[(last_pos-i)]) 
                    colpos[(last_pos-i)] += 1
                    if largest_j < j:
                        largest_j = j
                        
                        
        dicPos = {i: 0 for i in range(len(colpos))}
        for i in range(len(res_vector)):
            colpos[i] = 0
        for k, v in position.items():
            colpos[v[0]] += 1
        for k, v in sorted(position.items(), reverse=True):
            position[k] = (v[0], dicPos[v[0]])
            dicPos[v[0]] += 1
            
#         re-align the vertical position
        maxheight = max([v for k, v in colpos.items() if v != 0])
        newpos = {}
        for k, v in position.items():
            gap = (maxheight) / colpos[v[0]]
            newheight = (gap/2) + v[1]*gap
            newpos[k] = (v[0], newheight)
    
    
#         change orders to minimize line crossings
        for posx in sorted(list(set([v[0] for k, v in newpos.items()]))):
#             for iii in range(1):
            for iii in range(int(colpos[posx]/2+1)):
                for node in [k for k, v in newpos.items() if v[0] == posx]:
                    incoming_edges = self.str_G.in_edges(node)
                    predecessors = [edge[0] for edge in incoming_edges]
                    pred_in_pos = list(set([p for p in predecessors if p in newpos]))
                    if len(pred_in_pos) > 0:
                        pred_heights = [newpos[p][1] for p in pred_in_pos]
                        avg_pred_heights = average = sum(pred_heights) / len(pred_heights)
                        closest_node = 0
                        closest_height = 9999
                        closest_dist = 9999
                        [my_pos, my_height] = newpos[node]
                        my_dist = np.abs(my_height - avg_pred_heights)
                        for k, v in newpos.items():
                            if k == node:
                                continue
                            if v[0] == my_pos:
                                dist = np.abs(v[1] - avg_pred_heights)
                                if dist < closest_dist:
                                    closest_node = k
                                    closest_dist = dist
                                    closest_height = v[1]
                        if closest_dist < my_dist:
                            newpos[closest_node] = (my_pos, my_height)
                            newpos[node] = (my_pos, closest_height)

        position = newpos

        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]

        node_labels = {i: name for i, name in enumerate(self.str_vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.str_dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.str_dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Origins with Weighted Pipelining Including Dummy Nodes"
        has_proc = len([k for k in self.str_dic_vertex_id if k[0:5]  == "proc_"]) > 0
        title += " (" + str(last_pos-1) + " steps)"
        
        if figsize is None:
            figsize = (12, 8)

        self.draw_dummy(self.str_G, selected_vertices1,selected_vertices2, selected_vertices3, 
                        title=title, node_labels=node_labels, pos=position, reverse=False, figsize=figsize, showWeight=showWeight, forStretch=True)
    
    def drawOriginsStretch(self, target_vertex, title="", figsize=None, showWeight=False, excludeComp=False):
        
        if isinstance(target_vertex, str):
            if target_vertex not in self.str_dic_vertex_id:
                print(target_vertex + " is not an element")
                return
            target_vertex = self.str_dic_vertex_id[target_vertex]
        
#         Draw the path FROM the target
        position = {}
        colpos = {}
        posfill = set()
        selected_vertices1 = set()
        selected_vertices2 = set()

        succs = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=self.dic_new2old[target_vertex], orientation="reverse"))])
        succs = [self.dic_vertex_names[s] for s in succs]
        pattern = re.compile(r'^dumm_(\d+)_')
        
        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.str_csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.str_csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        
        res_vector = np.zeros((1, self.str_size_matrix))
        res_vector[0][target_vertex] = 1
        for i in range(50000):
            if sum(res_vector[i]) == 0:
                break
            res_vector.resize((res_vector.shape[0] + 1, res_vector.shape[1]))
            new_row = np.zeros(self.str_size_matrix)
            res_vector[-1, :] = new_row
            res_vector[i+1] = self.str_csr_matrix.dot(res_vector[i])
            
            for k in range(len(res_vector[i+1])):
                chkstr = re.sub(pattern, '', self.str_dic_vertex_names[k])
                if res_vector[i+1][k] > 0 and (chkstr not in succs and "proc_" + chkstr not in succs):
                    res_vector[i+1][k] = 0
                    continue

        for i in range(len(res_vector)):
            colpos[i] = 0

#         This part is to find the starting steps.  Currently not used.
        succs2 = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=self.dic_new2old[target_vertex]))])
        succs2 = set([self.dic_old2new[s] for s in succs2.nodes()])
        succLastReached = {}
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0:
                    if self.str_dic_vertex_names[j] in succs:
                        succLastReached[j] = i
        last_pos = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            last_pos += 1
            
            
        wait_edges = []
        for v in sorted([[v, k] for k, v in succLastReached.items()], reverse=True):
            start_step = last_pos-v[0]-1

            for e in self.G.out_edges(self.dic_new2old[v[1]]):
                if self.dic_old2new[e[1]] not in succLastReached:
                    continue
                if (last_pos - succLastReached[self.dic_old2new[e[1]]] - 1) - (start_step + self.G[e[0]][e[1]]["weight"]) - 1 > 0:
                    wait_edges.append((e[0], e[1], (last_pos - succLastReached[self.dic_old2new[e[1]]] - 1) - (start_step + self.G[e[0]][e[1]]["weight"] - 1)))
            
            
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0 and j != self.str_size_matrix: 
                    if self.str_dic_vertex_names[j][0:5] == "proc_":
                        selected_vertices2.add(j)
                        selected_vertices1.add(j)
                    else:
                        selected_vertices1.add(j)

#         initialize the positions
        last_pos = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            last_pos += 1
        done = False
        largest_j = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            nonzero = 0
            for j in range(len(res_vector[i])):
                if j not in selected_vertices1 and j not in selected_vertices2:
                    continue
                if res_vector[i][j]:
                    posfill.add(j)
                    position[j] = ((last_pos-i), colpos[(last_pos-i)]) 
                    colpos[(last_pos-i)] += 1
                    if largest_j < j:
                        largest_j = j
                        
                        
        dicPos = {i: 0 for i in range(len(colpos))}
        for i in range(len(res_vector)):
            colpos[i] = 0
        for k, v in position.items():
            colpos[v[0]] += 1
        for k, v in sorted(position.items(), reverse=True):
            position[k] = (v[0], dicPos[v[0]])
            dicPos[v[0]] += 1
            
#         re-align the vertical position
        maxheight = max([v for k, v in colpos.items() if v != 0])
        newpos = {}
        for k, v in position.items():
            gap = (maxheight) / colpos[v[0]]
            newheight = (gap/2) + v[1]*gap
            newpos[k] = (v[0], newheight)
    
    
#         change orders to minimize line crossings
        for posx in sorted(list(set([v[0] for k, v in newpos.items()]))):
#             for iii in range(1):
            for iii in range(int(colpos[posx]/2+1)):
                for node in [k for k, v in newpos.items() if v[0] == posx]:
                    incoming_edges = self.str_G.in_edges(node)
                    predecessors = [edge[0] for edge in incoming_edges]
                    pred_in_pos = list(set([p for p in predecessors if p in newpos]))
                    if len(pred_in_pos) > 0:
                        pred_heights = [newpos[p][1] for p in pred_in_pos]
                        avg_pred_heights = average = sum(pred_heights) / len(pred_heights)
                        closest_node = 0
                        closest_height = 9999
                        closest_dist = 9999
                        [my_pos, my_height] = newpos[node]
                        my_dist = np.abs(my_height - avg_pred_heights)
                        for k, v in newpos.items():
                            if k == node:
                                continue
                            if v[0] == my_pos:
                                dist = np.abs(v[1] - avg_pred_heights)
                                if dist < closest_dist:
                                    closest_node = k
                                    closest_dist = dist
                                    closest_height = v[1]
                        if closest_dist < my_dist:
                            newpos[closest_node] = (my_pos, my_height)
                            newpos[node] = (my_pos, closest_height)

        newpos = {self.dic_new2old[k]: v for k, v in newpos.items() if k in self.dic_new2old}
    
        position = newpos

        
        selected_vertices1 = [self.dic_new2old[k] for k in selected_vertices1 if k in self.dic_new2old]
        selected_vertices2 = [self.dic_new2old[k] for k in selected_vertices2 if k in self.dic_new2old]
        selected_vertices3 = [self.dic_new2old[target_vertex]]

        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Origins with Weighted Based Pipelining"
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0
        title += " (" + str(last_pos-1) + " steps)"
        
        if figsize is None:
            figsize = (12, 8)
        
        self.draw_selected_vertices_reverse_proc(self.G, selected_vertices1,selected_vertices2, selected_vertices3, 
                                title=title, node_labels=node_labels, pos=position, figsize=figsize, showWeight=showWeight, forStretch=True, wait_edges=wait_edges, excludeComp=excludeComp)

        
    def drawOrigins(self, target_vertex, title="", figsize=None, showWeight=False, excludeComp=False):

        if isinstance(target_vertex, str):
            if target_vertex not in self.dic_vertex_id:
                print(target_vertex + " is not an element")
                return
            target_vertex = self.dic_vertex_id[target_vertex]
    
            
        # Draw the path TO the target
        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        
        res_vector = np.zeros((1, self.size_matrix))
        res_vector[0][target_vertex] = 1
        for i in range(50000):
            if sum(res_vector[i]) == 0:
                break
            res_vector.resize((res_vector.shape[0] + 1, res_vector.shape[1]))
            new_row = np.zeros(self.size_matrix)
            res_vector[-1, :] = new_row
            
            res_vector[i+1] = self.csr_matrix.dot(res_vector[i])
            
        selected_vertices1 = set()
        selected_vertices2 = set()
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0: 
                    if self.dic_vertex_names[j][0:5] == "proc_":
                        selected_vertices2.add(j)
                        selected_vertices1.add(j)
                    else:
                        selected_vertices1.add(j)

        position = {}
        colpos = {}
        posfill = set()

        for i in range(len(res_vector)):
            colpos[i] = 0

        last_pos = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            last_pos += 1

        done = False
        largest_j = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            nonzero = 0
            for j in range(len(res_vector[i])):
                if j not in selected_vertices1 and j not in selected_vertices2:
                    continue
                if res_vector[i][j]:
                    posfill.add(j)
                    position[j] = ((last_pos-i), colpos[(last_pos-i)]) 
                    colpos[(last_pos-i)] += 1
                    if largest_j < j:
                        largest_j = j

        dicPos = {i: 0 for i in range(len(colpos))}
        for i in range(len(res_vector)):
            colpos[i] = 0
        for k, v in position.items():
            colpos[v[0]] += 1
        for k, v in sorted(position.items(), reverse=True):
            position[k] = (v[0], dicPos[v[0]])
            dicPos[v[0]] += 1
        
        maxheight = max([v for k, v in colpos.items() if v != 0])
        newpos = {}
        for k, v in position.items():
            
            gap = (maxheight) / colpos[v[0]]
            newheight = (gap/2) + v[1]*gap
            
            newpos[k] = (v[0], newheight)

#         change orders to minimize line crossings
        for posx in sorted(list(set([v[0] for k, v in newpos.items()]))):
#             for iii in range(1):
            for iii in range(int(colpos[posx]/2+1)):
                for node in [k for k, v in newpos.items() if v[0] == posx]:
                    incoming_edges = self.G.in_edges(node)
                    predecessors = [edge[0] for edge in incoming_edges]
                    pred_in_pos = list(set([p for p in predecessors if p in newpos]))
                    if len(pred_in_pos) > 0:
                        pred_heights = [newpos[p][1] for p in pred_in_pos]
                        avg_pred_heights = average = sum(pred_heights) / len(pred_heights)
                        closest_node = 0
                        closest_height = 9999
                        closest_dist = 9999
                        [my_pos, my_height] = newpos[node]
                        my_dist = np.abs(my_height - avg_pred_heights)
                        for k, v in newpos.items():
                            if k == node:
                                continue
                            if v[0] == my_pos:
                                dist = np.abs(v[1] - avg_pred_heights)
                                if dist < closest_dist:
                                    closest_node = k
                                    closest_dist = dist
                                    closest_height = v[1]
                        if closest_dist < my_dist:
                            newpos[closest_node] = (my_pos, my_height)
                            newpos[node] = (my_pos, closest_height)

        position = newpos

        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Origins"
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0
        title += " (" + str(len(set([v[0] for k, v in position.items() if (has_proc and self.dic_vertex_names[k][0:5] == "proc_") or (not has_proc and self.dic_vertex_names[k][0:5] != "proc_")]))-1) + " steps)"
    
        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]
        
        if figsize is None:
            figsize = (12, 8)
        self.draw_selected_vertices_reverse_proc(self.G, selected_vertices1,selected_vertices2, selected_vertices3, 
                                title=title, node_labels=node_labels, pos=position, figsize=figsize, showWeight=showWeight, excludeComp=excludeComp)

        


    def drawOffspringsStretch(self, target_vertex, title="", figsize=None, showWeight=False, excludeComp=False):

        
        if isinstance(target_vertex, str):
            if target_vertex not in self.str_dic_vertex_id:
                print(target_vertex + " is not an element")
                return
            target_vertex = self.str_dic_vertex_id[target_vertex]

        
#         Draw the path FROM the target
        position = {}
        colpos = {}
        posfill = set()
        selected_vertices1 = set()
        selected_vertices2 = set()

    
        succs = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=self.dic_new2old[target_vertex]))])
        succs = [self.dic_vertex_names[s] for s in succs]
        pattern = re.compile(r'^dumm_(\d+)_')

        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.str_csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.str_csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        
        res_vector = np.zeros((1, self.str_size_matrix))
        res_vector[0][target_vertex] = 1
        for i in range(50000):
            if sum(res_vector[i]) == 0:
                break
            res_vector.resize((res_vector.shape[0] + 1, res_vector.shape[1]))
            new_row = np.zeros(self.str_size_matrix)
            res_vector[-1, :] = new_row
            res_vector[i+1] = self.str_csr_matrix_T.dot(res_vector[i])
    
            for k in range(len(res_vector[i+1])):
                chkstr = re.sub(pattern, '', self.str_dic_vertex_names[k])
                if res_vector[i+1][k] > 0 and (chkstr not in succs and "proc_" + chkstr not in succs):
                    res_vector[i+1][k] = 0
                    continue
        
        for i in range(len(res_vector)):
            colpos[i] = 0
            
#         This part is to find the starting steps.  Currently not used.
        succs2 = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=self.dic_new2old[target_vertex]))])
        succs2 = set([self.dic_old2new[s] for s in succs2.nodes()])
        succLastReached = {}
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0:
                    if self.str_dic_vertex_names[j] in succs:
                        succLastReached[j] = i
#         print("WAITING STEPS:")             
        wait_edges = []
        for v in sorted([[v, k] for k, v in succLastReached.items()]):
            for e in self.G.out_edges(self.dic_new2old[v[1]]):
                if succLastReached[self.dic_old2new[e[1]]] - (v[0] + self.G[e[0]][e[1]]["weight"]) > 0: 
                    wait_edges.append((e[0], e[1], succLastReached[self.dic_old2new[e[1]]] - (v[0] + self.G[e[0]][e[1]]["weight"])))        
                    
                    
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0 and j != self.str_size_matrix: 
                    if self.str_dic_vertex_names[j][0:5] == "proc_":
                        selected_vertices2.add(j)
                        selected_vertices1.add(j)
                    else:
                        selected_vertices1.add(j)

#         initialize the positions
        last_pos = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            last_pos += 1
        done = False
        largest_j = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            nonzero = 0
            for j in range(len(res_vector[i])):
                if j not in selected_vertices1 and j not in selected_vertices2:
                    continue
                if res_vector[i][j]:
                    posfill.add(j)
                    position[j] = (i, colpos[i]) 
                    colpos[i] += 1
                    if largest_j < j:
                        largest_j = j
        
        dicPos = {i: 0 for i in range(len(colpos))}
        for i in range(len(res_vector)):
            colpos[i] = 0
        for k, v in position.items():
            colpos[v[0]] += 1
        for k, v in sorted(position.items(), reverse=True):
            position[k] = (v[0], dicPos[v[0]])
            dicPos[v[0]] += 1

            
#         re-align the vertical position
        maxheight = max([v for k, v in colpos.items() if v != 0])
        newpos = {}
        for k, v in position.items():
            gap = (maxheight) / colpos[v[0]]
            newheight = (gap/2) + v[1]*gap
            newpos[k] = (v[0], newheight)

#         change orders to minimize line crossings
        for posx in sorted(list(set([v[0] for k, v in newpos.items()]))):
#             for iii in range(1):
            for iii in range(int(colpos[posx]/2+1)):
                for node in [k for k, v in newpos.items() if v[0] == posx]:
                    incoming_edges = self.str_G.in_edges(node)
                    predecessors = [edge[0] for edge in incoming_edges]
                    pred_in_pos = list(set([p for p in predecessors if p in newpos]))
                    if len(pred_in_pos) > 0:
                        pred_heights = [newpos[p][1] for p in pred_in_pos]
                        avg_pred_heights = average = sum(pred_heights) / len(pred_heights)
                        closest_node = 0
                        closest_height = 9999
                        closest_dist = 9999
                        [my_pos, my_height] = newpos[node]
                        my_dist = np.abs(my_height - avg_pred_heights)
                        for k, v in newpos.items():
                            if k == node:
                                continue
                            if v[0] == my_pos:
                                dist = np.abs(v[1] - avg_pred_heights)
                                if dist < closest_dist:
                                    closest_node = k
                                    closest_dist = dist
                                    closest_height = v[1]
                        if closest_dist < my_dist:
                            newpos[closest_node] = (my_pos, my_height)
                            newpos[node] = (my_pos, closest_height)

        
        newpos = {self.dic_new2old[k]: v for k, v in newpos.items() if k in self.dic_new2old}
        position = newpos

        selected_vertices1 = [self.dic_new2old[k] for k in selected_vertices1 if k in self.dic_new2old]
        selected_vertices2 = [self.dic_new2old[k] for k in selected_vertices2 if k in self.dic_new2old]
        selected_vertices3 = [self.dic_new2old[target_vertex]]

        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Offsprings with Weighted Pipelining"
        title += " (" + str(last_pos-1) + " steps)"
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0

        if figsize is None:
            figsize = (12, 8)
        
        self.draw_selected_vertices_reverse_proc(self.G_T, selected_vertices1,selected_vertices2, selected_vertices3, 
                        title=title, node_labels=node_labels, pos=position, reverse=True, figsize=figsize, showWeight=showWeight, forStretch=True, wait_edges=wait_edges, excludeComp=excludeComp)


    def drawOffspringsStretchDummy(self, target_vertex, title="", figsize=None, showWeight=False):

        
        if isinstance(target_vertex, str):
            if target_vertex not in self.str_dic_vertex_id:
                print(target_vertex + " is not an element")
                return
            target_vertex = self.str_dic_vertex_id[target_vertex]
                    
#         Draw the path FROM the target
        position = {}
        colpos = {}
        posfill = set()
        selected_vertices1 = set()
        selected_vertices2 = set()

        succs = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=self.dic_new2old[target_vertex]))])
        succs = [self.dic_vertex_names[s] for s in succs]
        pattern = re.compile(r'^dumm_(\d+)_')

        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.str_csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.str_csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        
        res_vector = np.zeros((1, self.str_size_matrix))
        res_vector[0][target_vertex] = 1
        for i in range(50000):
            if sum(res_vector[i]) == 0:
                break
            res_vector.resize((res_vector.shape[0] + 1, res_vector.shape[1]))
            new_row = np.zeros(self.str_size_matrix)
            res_vector[-1, :] = new_row
            res_vector[i+1] = self.str_csr_matrix_T.dot(res_vector[i])
            
            for k in range(len(res_vector[i+1])):
                chkstr = re.sub(pattern, '', self.str_dic_vertex_names[k])
                if res_vector[i+1][k] > 0 and (chkstr not in succs and "proc_" + chkstr not in succs):
                    res_vector[i+1][k] = 0
                    continue

        for i in range(len(res_vector)):
            colpos[i] = 0
                    
#         This part is to find the starting steps.  Currently not used.
        succs2 = self.G.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(self.G, source=self.dic_new2old[target_vertex]))])
        succs2 = set([self.dic_old2new[s] for s in succs2.nodes()])
        succLastReached = {}
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0:
                    if self.str_dic_vertex_names[j] in succs:
                        succLastReached[j] = i
        print("WAITING STEPS:")
        for v in sorted([[v, k] for k, v in succLastReached.items()]):
            for e in self.G.out_edges(self.dic_new2old[v[1]]):
                if succLastReached[self.dic_old2new[e[1]]] - (v[0] + self.G[e[0]][e[1]]["weight"]) > 0:
                    print(self.str_dic_vertex_names[v[1]] + " (" + str(v[0]) + ") -> " + self.dic_vertex_names[e[1]] + " (" + str(v[0] + self.G[e[0]][e[1]]["weight"]) + 
                          ") with wait " + str(succLastReached[self.dic_old2new[e[1]]] - (v[0] + self.G[e[0]][e[1]]["weight"])))
                                        
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0 and j != self.str_size_matrix: 
                    if self.str_dic_vertex_names[j][0:5] == "proc_":
                        selected_vertices2.add(j)
                        selected_vertices1.add(j)
                    else:
                        selected_vertices1.add(j)

#         initialize the positions
        last_pos = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            last_pos += 1
        done = False
        largest_j = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            nonzero = 0
            for j in range(len(res_vector[i])):
                if j not in selected_vertices1 and j not in selected_vertices2:
                    continue
                if res_vector[i][j]:
                    posfill.add(j)
                    position[j] = (i, colpos[i]) 
                    colpos[i] += 1
                    if largest_j < j:
                        largest_j = j
        
        dicPos = {i: 0 for i in range(len(colpos))}
        for i in range(len(res_vector)):
            colpos[i] = 0
        for k, v in position.items():
            colpos[v[0]] += 1
        for k, v in sorted(position.items(), reverse=True):
            position[k] = (v[0], dicPos[v[0]])
            dicPos[v[0]] += 1

            
#         re-align the vertical position
        maxheight = max([v for k, v in colpos.items() if v != 0])
        newpos = {}
        for k, v in position.items():
            gap = (maxheight) / colpos[v[0]]
            newheight = (gap/2) + v[1]*gap
            newpos[k] = (v[0], newheight)

#         change orders to minimize line crossings
        for posx in sorted(list(set([v[0] for k, v in newpos.items()]))):
#             for iii in range(1):
            for iii in range(int(colpos[posx]/2+1)):
                for node in [k for k, v in newpos.items() if v[0] == posx]:
                    incoming_edges = self.str_G.in_edges(node)
                    predecessors = [edge[0] for edge in incoming_edges]
                    pred_in_pos = list(set([p for p in predecessors if p in newpos]))
                    if len(pred_in_pos) > 0:
                        pred_heights = [newpos[p][1] for p in pred_in_pos]
                        avg_pred_heights = average = sum(pred_heights) / len(pred_heights)
                        closest_node = 0
                        closest_height = 9999
                        closest_dist = 9999
                        [my_pos, my_height] = newpos[node]
                        my_dist = np.abs(my_height - avg_pred_heights)
                        for k, v in newpos.items():
                            if k == node:
                                continue
                            if v[0] == my_pos:
                                dist = np.abs(v[1] - avg_pred_heights)
                                if dist < closest_dist:
                                    closest_node = k
                                    closest_dist = dist
                                    closest_height = v[1]
                        if closest_dist < my_dist:
                            newpos[closest_node] = (my_pos, my_height)
                            newpos[node] = (my_pos, closest_height)

        position = newpos

        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]

        node_labels = {i: name for i, name in enumerate(self.str_vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.str_dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.str_dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Offsprings with Weighted Pipelining Including Dummy Nodes"
        has_proc = len([k for k in self.str_dic_vertex_id if k[0:5]  == "proc_"]) > 0
#         title += " (" + str(len(set([v[0] for k, v in position.items() if (has_proc and self.str_dic_vertex_names[k][0:5] == "proc_") or (not has_proc and self.str_dic_vertex_names[k][0:5] != "proc_")]))-1) + " steps)"
        title += " (" + str(last_pos-1) + " steps)"
        
        if figsize is None:
            figsize = (12, 8)
        
        self.draw_dummy(self.str_G_T, selected_vertices1,selected_vertices2, selected_vertices3, 
                        title=title, node_labels=node_labels, pos=position, reverse=True, figsize=figsize, showWeight=showWeight, forStretch=True)


        
    def drawOffsprings(self, target_vertex, title="", figsize=None, showWeight=False, excludeComp=False):
        
        if isinstance(target_vertex, str):
            if target_vertex not in self.dic_vertex_id:
                print(target_vertex + " is not an element")
                return
            target_vertex = self.dic_vertex_id[target_vertex]

#         Draw the path FROM the target
        position = {}
        colpos = {}
        posfill = set()
        selected_vertices1 = set()
        selected_vertices2 = set()
            
#         # Draw the path TO the target
        if not nx.is_directed_acyclic_graph(nx.DiGraph(self.csr_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(self.csr_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
            return
        
        res_vector = np.zeros((1, self.size_matrix))
        res_vector[0][target_vertex] = 1
        for i in range(50000):
            if sum(res_vector[i]) == 0:
                break
            res_vector.resize((res_vector.shape[0] + 1, res_vector.shape[1]))
            new_row = np.zeros(self.size_matrix)
            res_vector[-1, :] = new_row
            res_vector[i+1] = self.csr_matrix_T.dot(res_vector[i])

        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
                if res_vector[i][j] != 0 and j != self.size_matrix: 
                    if self.dic_vertex_names[j][0:5] == "proc_":
                        selected_vertices2.add(j)
                        selected_vertices1.add(j)
                    else:
                        selected_vertices1.add(j)

        for i in range(len(res_vector)):
            colpos[i] = 0
            
            
#         initialize the positions
        last_pos = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            last_pos += 1
        done = False
        largest_j = 0
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            nonzero = 0
            for j in range(len(res_vector[i])):
                if j not in selected_vertices1 and j not in selected_vertices2:
                    continue
                if res_vector[i][j]:
                    posfill.add(j)
                    position[j] = (i, colpos[i]) 
                    colpos[i] += 1
                    if largest_j < j:
                        largest_j = j
        
        dicPos = {i: 0 for i in range(len(colpos))}
        for i in range(len(res_vector)):
            colpos[i] = 0
        for k, v in position.items():
            colpos[v[0]] += 1
        for k, v in sorted(position.items(), reverse=True):
            position[k] = (v[0], dicPos[v[0]])
            dicPos[v[0]] += 1
           
        maxheight = max([v for k, v in colpos.items() if v != 0])
        newpos = {}
        for k, v in position.items():
            gap = (maxheight) / colpos[v[0]]
            newheight = (gap/2) + v[1]*gap
            newpos[k] = (v[0], newheight)

#         change orders to minimize line crossings
        for posx in sorted(list(set([v[0] for k, v in newpos.items()]))):
#             for iii in range(1):
            for iii in range(int(colpos[posx]/2+1)):
                for node in [k for k, v in newpos.items() if v[0] == posx]:
                    incoming_edges = self.G.in_edges(node)
                    predecessors = [edge[0] for edge in incoming_edges]
                    pred_in_pos = list(set([p for p in predecessors if p in newpos]))
                    if len(pred_in_pos) > 0:
                        pred_heights = [newpos[p][1] for p in pred_in_pos]
                        avg_pred_heights = average = sum(pred_heights) / len(pred_heights)
                        closest_node = 0
                        closest_height = 9999
                        closest_dist = 9999
                        [my_pos, my_height] = newpos[node]
                        my_dist = np.abs(my_height - avg_pred_heights)
                        for k, v in newpos.items():
                            if k == node:
                                continue
                            if v[0] == my_pos:
                                dist = np.abs(v[1] - avg_pred_heights)
                                if dist < closest_dist:
                                    closest_node = k
                                    closest_dist = dist
                                    closest_height = v[1]
                        if closest_dist < my_dist:
                            newpos[closest_node] = (my_pos, my_height)
                            newpos[node] = (my_pos, closest_height)

        position = newpos
        
        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        
        print("Number of Elements: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] != "proc_"])))
        print("Number of Processes: " + str(len([1 for k in selected_vertices1 if self.dic_vertex_names[k][0:5] == "proc_"])))
        if title == "":
            title = "Data Offsprings"
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0
        title += " (" + str(len(set([v[0] for k, v in position.items() if (has_proc and self.dic_vertex_names[k][0:5] == "proc_") or (not has_proc and self.dic_vertex_names[k][0:5] != "proc_")]))-1) + " steps)"
        if figsize is None:
            figsize = (12, 8)
            
        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]
        
        self.draw_selected_vertices_reverse_proc(self.G_T, selected_vertices1,selected_vertices2, selected_vertices3, 
                        title=title, node_labels=node_labels, pos=position, reverse=True, figsize=figsize, showWeight=showWeight, excludeComp=excludeComp)
        
        
    def showSourceNodes(self):
        sources = [node for node in self.G.nodes() if len(list(self.G.predecessors(node))) == 0]

        print("Source Nodes:", [self.dic_vertex_names[s] for s in sources])
        
        
    def showSinkNodes(self):
        sinks = [node for node in self.G.nodes() if len(list(self.G.successors(node))) == 0]

        print("Sink Nodes:", [self.dic_vertex_names[s] for s in sinks])
        
    def showStats(self, g):
        
        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0
        if has_proc and list(nx.weakly_connected_components(g)) == 1:
            is_bipartite = nx.bipartite.sets(g)
            if is_bipartite:
                self.showBipartiteStats(g)
                return
            
        in_degree_centrality = nx.in_degree_centrality(g)
        out_degree_centrality = nx.out_degree_centrality(g)
#         closeness_centrality = nx.closeness_centrality(g)
#         betweenness_centrality = nx.betweenness_centrality(g)
        
        cnt_max = 5
        print("In Degree Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in in_degree_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 3))
            cnt += 1
        print("")
        
        print("Out Degree Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in out_degree_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 3))
            cnt += 1
        print("")
        
#         print("Closeness Centrality:")
#         cnt = 0
#         for z in sorted([(v, k) for k, v in closeness_centrality.items()], reverse=True):
#             if cnt == cnt_max:
#                 break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
#             cnt += 1
#         print("")
#         print("Betweenness Centrality:")
#         cnt = 0
#         for z in sorted([(v, k) for k, v in betweenness_centrality.items()], reverse=True):
#             if cnt == cnt_max:
#                 break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
#             cnt += 1
#         print("")
        
    def showBipartiteStats(self, g):
        is_bipartite, node_sets = nx.bipartite.sets(g)
        projection = nx.bipartite.projected_graph(g, node_sets)
        degree_centrality = nx.degree_centrality(projection)
#         closeness_centrality = nx.closeness_centrality(projection)
#         betweenness_centrality = nx.betweenness_centrality(projection)
        cnt_max = 5
        print("Degree Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in degree_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 3))
            cnt += 1
        print("")
#         print("Closeness Centrality:")
#         cnt = 0
#         for z in sorted([(v, k) for k, v in closeness_centrality.items()], reverse=True):
#             if cnt == cnt_max:
#                 break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
#             cnt += 1
#         print("")
#         print("Betweenness Centrality:")
#         cnt = 0
#         for z in sorted([(v, k) for k, v in betweenness_centrality.items()], reverse=True):
#             if cnt == cnt_max:
#                 break
#             print(self.dic_vertex_names[z[1]], round(z[0], 3))
#             cnt += 1
#         print("")
        
    def drawFromLargestComponent(self, figsize=(30, 30), showWeight=False, excludeComp=False):
        
        connected_components = list(nx.weakly_connected_components(self.G))
        largest_connected_component = None
        sorted_graphs = sorted([[len(c), c] for c in connected_components], reverse=True)
        for i in range(len(sorted_graphs)):
            if i == 0:
                largest_connected_component = sorted_graphs[i][1]
        largest_G = self.G.subgraph(largest_connected_component)
        
        # Find the topological order
        topological_order = list(nx.topological_sort(largest_G))
        self.drawOffsprings(topological_order[0], figsize=figsize, showWeight=showWeight, excludeComp=excludeComp)
        self.drawOrigins(topological_order[-1], figsize=figsize, showWeight=showWeight, excludeComp=excludeComp)
        
    
    def suggest_coupling(self, g):
                
        longest_path_length = nx.dag_longest_path_length(g)
        
        if longest_path_length == 0:
            return
        
        longest_path = nx.dag_longest_path(g)
        node_criticality = None
        node_criticality = [((nx.dag_longest_path_length(g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, target_node))])) + 
                        nx.dag_longest_path_length(g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, target_node, orientation='reverse'))]))) / 
                        longest_path_length, target_node) for target_node in g.nodes() if self.dic_vertex_names[target_node][0:5] == "proc_"]

        
        print("SUGGESTED COUPLINGS")
        cnt = 0
        for i in range(len(node_criticality)):
            if cnt > 20:
                break
            if node_criticality[i][1] in longest_path:
                continue
            for j in range(i+1, len(node_criticality), 1):
                if node_criticality[j][1] in longest_path:
                    continue
                if node_criticality[i][0] + node_criticality[j][0] > 1.3:
                    continue;
                lowers = g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, node_criticality[i][1]))])
                uppers = g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, node_criticality[i][1], orientation='reverse'))])
                if node_criticality[j][1] not in lowers and node_criticality[j][1] not in uppers:

                    lowers_j = g.edge_subgraph([(f[0], f[1]) for f in list(nx.edge_dfs(g, node_criticality[j][1]))])
                    diff_longest = np.abs(nx.dag_longest_path_length(lowers) - nx.dag_longest_path_length(lowers_j))
                    if diff_longest < 6:
                        print(self.dic_vertex_names[node_criticality[i][1]], self.dic_vertex_names[node_criticality[j][1]])
                        
                        cnt += 1
                        if cnt > 20:
                            break
                            
        
        print("")
        
    def suggest_opportunities(self, g):
                
        longest_path_length = nx.dag_longest_path_length(g)

        opportunity_list = []
        for n in g.nodes():
            if self.dic_vertex_names[n][0:5] != "proc_":
                continue

            h = g.copy()
            outgoing_edges = list(h.out_edges(n))
            for e in outgoing_edges:
                h[e[0]][e[1]]["weight"] = 1      
            improvement_without = longest_path_length - nx.dag_longest_path_length(h)
            if improvement_without > 1:
                opportunity_list.append((improvement_without, n))

        print("SUGGESTED THROUGHPUT IMPROVEMENT OPPORTUNITIES")
        for n in sorted(opportunity_list, reverse=True):
            print(self.dic_vertex_names[n[1]], n[0])
            
        print("")

    def getMatrix(self):
        return self.csr_matrix.toarray()
    
    def getVertexNames(self):
        return self.vertex_names
    
    def getDicVertexNames(self):
        return self.dic_vertex_names
  
    def draw_dummy(self, G, selected_vertices1, selected_vertices2, selected_vertices3, title, node_labels, 
                                            pos, reverse=False, figsize=(12, 8), showWeight=False, forStretch=False):
            
        # Create a subgraph with only the selected vertices
        subgraph1 = G.subgraph(selected_vertices1)
        subgraph2 = G.subgraph(selected_vertices2)
        subgraph3 = G.subgraph(selected_vertices3)
        
        if reverse:
            subgraph1 = subgraph1.reverse()
            subgraph2 = subgraph2.reverse()
            subgraph3 = subgraph3.reverse()

        # Set figure size to be larger
        plt.figure(figsize=figsize)

        # Set figure size to be larger
        node_labels = {node: "\n".join(["\n".join(textwrap.wrap(s, width=5)) for s in label.replace("_", "_\n").replace(" ", "\n").split("\n")]) for node, label in node_labels.items()}

        defFontSize=6
        defNodeSize=300
        defFontColor='black'
            
        node_labels1 = {k: v for k, v in node_labels.items() if k in subgraph1 and k not in subgraph2 and k not in subgraph3}
        node_labels2 = {k: v for k, v in node_labels.items() if k in subgraph2}
        node_labels3 = {k: v for k, v in node_labels.items() if k in subgraph3}
        
        nx.draw(subgraph1, pos, linewidths=0, with_labels=True, labels=node_labels1, node_size=defNodeSize, node_color='skyblue', font_size=defFontSize, font_color=defFontColor, arrowsize=10, edgecolors='black')
        nx.draw(subgraph2, pos, linewidths=0, with_labels=True, labels=node_labels2, node_size=defNodeSize, node_color='orange', font_size=defFontSize, font_color='black', arrowsize=10, edgecolors='black')
        nx.draw(subgraph3, pos, linewidths=0, with_labels=True, labels=node_labels3, node_size=defNodeSize, node_color='pink', font_size=defFontSize, font_color='black', arrowsize=10, edgecolors='black')
        
        if showWeight:
            edge_labels = {(i, j): subgraph1[i][j]['weight'] for i, j in subgraph1.edges()}
            nx.draw_networkx_edge_labels(subgraph1, pos, edge_labels=edge_labels)

            longest_path = nx.dag_longest_path(subgraph1)  # Use NetworkX's built-in function
            critical_edges = [(longest_path[i], longest_path[i + 1]) for i in range(len(longest_path) - 1)]
            nx.draw_networkx_edges(subgraph1, pos, edgelist=critical_edges, edge_color='brown', width=1.25)
            
        
        if forStretch:
            # Draw a vertical line at x=0.5 using matplotlib
            max_horizontal_pos = max([v[0] for k, v in pos.items()])

            for i in range(max_horizontal_pos, 0, -5):
    #             width = 0.5 if (max_horizontal_pos - i) % 10 == 0 else 0.25
                width = 0.25
                linestyle = "dashed" if (max_horizontal_pos - i) % 10 == 0 else "dotted"
                plt.axvline(x=i, color="orange", linestyle=linestyle, linewidth=width)
                
        plt.title(title)
        plt.show()
            
            
            
    def populateStretch(self):


        matrix = self.csr_matrix.toarray()
        vv = copy.deepcopy(self.vertex_names)

        list_weights = []
        for i in range(len(matrix)):
            lis = []
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    if len(lis) > 0:
                        lis[2].append(j)
                    else:
                        lis = [i, matrix[i][j], [j]]
            if len(lis) > 0:
                    list_weights.append(lis)

        dic_old2new = {}
        weights_so_far = 0
        dic_weights = {t[0]: t[1] for t in list_weights}

        for i in range(len(matrix)):
            dic_old2new[i] = i + weights_so_far
            if i in dic_weights:
                weights_so_far += dic_weights[i]-1
        
    
        # insert new blank records
        for f in sorted(list_weights, reverse=True):
            row = f[0] # Insert at the second row (index 1)
            row_name = vv[row]
            weight = f[1]  # Insert at the second row (index 1)
            for i in range(weight-1):
                vv.insert(row+1, "dumm_" + str(i+1) + "_" + row_name)

        # insert new blank records
        matrix = np.zeros((len(vv), len(vv)))
        for i in range(len(vv)):
            if i < len(vv)-1:
                if vv[i][0:5] == "dumm_" and vv[i+1][0:5] == "dumm_":
                    matrix[i][i+1] = 1
        
        for f in list_weights:
            old_fr = f[0]
            new_fr = dic_old2new[f[0]]
            tt = f[2]
            weight = f[1]
            # new jump 
            # for weight 1 only
            if weight == 1:
                for t in tt:
                    old_to = t
                    new_to = dic_old2new[t]
                    matrix[new_fr][new_to] = 1
            else:

            # new jump 
            # for 2<weight last new line
                for t in tt:
                    old_to = t
                    new_to = dic_old2new[t]
                    matrix[new_fr+weight-1][new_to] = 1

            # new jump 
            # for 2<weight 
                matrix[new_fr][new_fr+1] = 1

        self.str_vertex_names = vv
        self.dic_old2new = dic_old2new
        self.str_csr_matrix = csr_matrix(matrix)
        self.str_csr_matrix_T = self.str_csr_matrix.transpose()
        self.str_size_matrix = self.str_csr_matrix.shape[0]
        self.str_G = nx.DiGraph(self.str_csr_matrix)
        self.str_G_T = nx.DiGraph(self.str_csr_matrix_T)
    
        for i in range(len(self.str_vertex_names)):
            self.str_dic_vertex_names[i] = self.str_vertex_names[i]
            self.str_dic_vertex_id[self.str_vertex_names[i]] = i

        self.dic_new2old = {v: k for k,v in self.dic_old2new.items()}
    

    def keepOnlyProcesses(self):
        
        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k[0:5]  == "proc_"]) > 0
        if not has_proc:
            print("There are no processe nodes in the graph.")
            return
        connected_components = list(nx.weakly_connected_components(self.G))
        largest_connected_component = None
        if len(connected_components) > 0:
            print("Keeping only the largest connected graph.")
            sorted_graphs = sorted([[len(c), c] for c in connected_components], reverse=True)
            for i in range(len(sorted_graphs)):
                if i == 0:
                    largest_connected_component = sorted_graphs[i][1]
            largest_G = self.G.subgraph(largest_connected_component)
        
            is_bipartite = nx.bipartite.sets(largest_G)
            if not is_bipartite:
                print("There is not a bipartite graph.")
                return
        
            self.G =  largest_G
        else:
            is_bipartite = nx.bipartite.sets(self.G)
            if not is_bipartite:
                print("There is not a bipartite graph.")
                return
        
        
        edges = self.csr_matrix_to_edge_list(self.csr_matrix)
        new_edges = []
        for i in range(len(edges)):
            id1 = edges[i][0]
            id2 = edges[i][1]
            if self.dic_vertex_names[id1][0:5] != "proc_":
                continue
            
            w = self.G[id1][id2]["weight"]
                
            for s in list(self.G.successors(id2)):
                w2 = self.G[id2][s]["weight"]
                weight = int((w + w2)/2)
                if weight == 0: weight = 1
                new_edges.append((id1, s, weight))
        tmp_matrix = self.edge_list_to_csr_matrix(new_edges)

        if not nx.is_directed_acyclic_graph(nx.DiGraph(tmp_matrix)):
            print("The graph is not a Directed Acyclic Graph (DAG).")

            # Find cycles in the graph
            cycles = list(nx.simple_cycles(nx.DiGraph(tmp_matrix)))

            print("Cycles in the graph:")
            for cycle in cycles:
                print(cycle)
        else:
            self.csr_matrix = tmp_matrix
            self.csr_matrix_T = self.csr_matrix.transpose()
            self.size_matrix = self.csr_matrix.shape[0]
        
            self.G = nx.DiGraph(self.csr_matrix)
            self.G_T = nx.DiGraph(self.csr_matrix_T)
            
            for i in range(len(copy.deepcopy(self.vertex_names))):
                if i not in self.G.nodes:
                    self.vertex_names.pop(i)
                    
                    



