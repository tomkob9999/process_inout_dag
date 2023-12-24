# Version: 1.1.0
# Last Update: 2023/12/24
# Author: Tomio Kobayashi

# - generateProcesses  genProcesses() DONE
# - DAG check  checkDAG(from, to) DONE
# - Process coupling  (process1, process2) DONE
# - Link Node  linkFields(from, to) DONE
# - Add Node  addField(name) NOT YET
  
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random
from datetime import date

# from networkx.algorithms import bipartite

class DataJourneyDAG:

    def __init__(self):
        self.vertex_names = []
        self.adjacency_matrix = []
        self.adjacency_matrix_T = []
        self.size_matrix = 0
        
        self.dic_vertex_names = {}
        self.dic_vertex_id = {}
        
        G = 0
        G_T = 0
        
    def adjacency_matrix_to_edge_list(self, adc_matrix):
        edge_list = []
        num_nodes = len(adc_matrix)

        for i in range(num_nodes):
            for j in range(num_nodes):
#                 if adc_matrix[i][j] == 1:
#                     edge_list.append((i, j))  # Add edges only for non-zero entries
                if adc_matrix[i][j] >= 1:
                    edge_list.append((i, j, adc_matrix[i][j]))  # Add edges only for non-zero entries

        return edge_list

    def draw_selected_vertices_reverse_proc(self, G, selected_vertices1, selected_vertices2, selected_vertices3, title, node_labels, 
                                            pos, reverse=False, figsize=(12, 8), showWeight=False):

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


        # Draw the graph
        nx.draw(subgraph1, pos, with_labels=True, labels=node_labels, node_size=1000, node_color='skyblue', font_size=10, font_color='black', arrowsize=10, edgecolors='black')
        nx.draw(subgraph2, pos, with_labels=True, labels=node_labels, node_size=1000, node_color='orange', font_size=10, font_color='black', arrowsize=10, edgecolors='black')
        nx.draw(subgraph3, pos, with_labels=True, labels=node_labels, node_size=1500, node_color='pink', font_size=10, font_color='black', arrowsize=10, edgecolors='black')

        if showWeight:
            edge_labels = {(i, j): subgraph1[i][j]['weight'] for i, j in subgraph1.edges()}
            nx.draw_networkx_edge_labels(subgraph1, pos, edge_labels=edge_labels)
        
        plt.title(title)
        plt.show()
            
        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k  == "proc_"]) > 0
        if has_proc:
            self.showBipartiteStats(subgraph1)
        else:
            self.showStats(subgraph1)
            
        # Find the topological order
        topological_order = list(nx.topological_sort(subgraph1))
        # Print the topological order
        print("TOPOLOGICAL ORDER:")
        print(" > ".join([self.dic_vertex_names[t] for t in topological_order if (has_proc and self.dic_vertex_names[t][0:5] == "proc_") or not has_proc]))
        
        print("")
        longest_path = nx.dag_longest_path(subgraph1)  # Use NetworkX's built-in function
        longest_path_length = nx.dag_longest_path_length(subgraph1)

        # Print the longest path and its length
        print("LONGEST PATH (" + str(longest_path_length) + "):")
        print(" > ".join([self.dic_vertex_names[t] for t in longest_path if (has_proc and self.dic_vertex_names[t][0:5] == "proc_") or not has_proc]))
        print("")

        
    def edge_list_to_adjacency_matrix(self, edges):
        """
        Converts an edge list to an adjacency matrix.

        Args:
            edges: A list of tuples, where each tuple represents an edge (source node, target node).

        Returns:
            A 2D list representing the adjacency matrix.
        """

        num_nodes = max(max(edge) for edge in edges) + 1  # Determine the number of nodes
        adjacency_matrix = np.array([np.array([0] * num_nodes) for _ in range(num_nodes)])

        for edge in edges:
            if len(edge) >= 3:
                adjacency_matrix[edge[0]][edge[1]] = edge[2]
            else:
                adjacency_matrix[edge[0]][edge[1]] = 1

        return adjacency_matrix

    def data_import(self, file_path, is_edge_list=False):
#         Define rows as TO and columns as FROM
        if is_edge_list:
            edges = self.read_edge_list_from_file(file_path)
            self.adjacency_matrix = self.edge_list_to_adjacency_matrix(edges)
        else:
            data = np.genfromtxt(file_path, delimiter='\t', dtype=str, encoding=None)

            # Extract the names from the first record
            self.vertex_names = list(data[0])

            # Extract the adjacency matrix
            self.adjacency_matrix = data[1:]
        
        # Convert the adjacency matrix to a NumPy array of integers
        self.adjacency_matrix = np.array(self.adjacency_matrix, dtype=int)

#         # Generate random weights between 1 and 5 for testing
#         for i in range(len(self.adjacency_matrix)):
#             random_integer = random.randint(1, 5)
#             for j in range(len(self.adjacency_matrix)):
#                 if self.adjacency_matrix[j][i] == 1:
#                     self.adjacency_matrix[j][i] = random_integer

#         print("self.adjacency_matrix")
#         for i in range(len(self.adjacency_matrix)):
#             print(self.adjacency_matrix[i])
        
        if self.has_cycle(self.adjacency_matrix):
            print("The result graph is not a DAG")
            return

        self.adjacency_matrix_T = self.adjacency_matrix.T

        # Matrix of row=FROM, col=TO
        self.size_matrix = len(self.adjacency_matrix)

        self.G = nx.DiGraph(self.adjacency_matrix)
        self.G_T = nx.DiGraph(self.adjacency_matrix_T)
        
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
        """
        Writes an edge list to a text file.

        Args:
            edges: A list of tuples, where each tuple represents an edge (source node, target node).
            filename: The name of the file to write to.
        """
        edges = self.adjacency_matrix_to_edge_list(self.adjacency_matrix)

        with open(filename, "w") as file:
            # Write headers
            file.write("\t".join(self.vertex_names) + "\n")
            for edge in edges:
                file.write(f"{edge[0]}\t{edge[1]}\n")

    def read_edge_list_from_file(self, filename):
        """
        Reads an edge list from a text file.

        Args:
            filename: The name of the file to read.

        Returns:
            A list of tuples, where each tuple represents an edge (source node, target node).
        """

        edges = []
        with open(filename, "r") as file:
            ini = True
            for line in file:
                if ini:
                    self.vertex_names = line.strip().split("\t")
                    ini = False
                else:
                    source, target = map(int, line.strip().split())
                    edges.append((source, target))
        return edges

    def write_adjacency_matrix_to_file(self, filename):
        with open(filename, "w") as file:
            file.write("\t".join(self.vertex_names) + "\n") 
            for row in self.adjacency_matrix:
                file.write("\t".join(map(str, row)) + "\n")  # Join elements with tabs and add newline


    def genProcesses(self):
        
        # Show stats of procs
        has_proc = len([k for k in self.dic_vertex_id if k  == "proc_"]) > 0
        if has_proc:
            print("Already contains processes.")
            return
        
        edges = self.adjacency_matrix_to_edge_list(self.adjacency_matrix)
        new_edges = []
        dicNewID = {}
        setVertex = set(self.vertex_names)
        for i in range(len(edges)):
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
            new_edges.append((edges[i][0], new_id, 1))
            new_edges.append((new_id, edges[i][1], edges[i][2]))
    
        self.adjacency_matrix = self.edge_list_to_adjacency_matrix(new_edges)
        self.adjacency_matrix_T = self.adjacency_matrix.T
        self.size_matrix = len(self.adjacency_matrix)

        self.G = nx.DiGraph(self.adjacency_matrix)
        self.G_T = nx.DiGraph(self.adjacency_matrix_T)
        
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
        self.showBipartiteStats(largest_G)
        
    def coupleProcesses(self, proc1, proc2):
        edges = self.adjacency_matrix_to_edge_list(self.adjacency_matrix)
        if proc1[0:5] != "proc_" or proc2[0:5] != "proc_":
            return
        new_edges = []
        old = self.dic_vertex_id[proc1]
        new = self.dic_vertex_id[proc2]
        for i in range(len(edges)):
            id1 = edges[i][0]
            id2 = edges[i][1]
            if edges[i][0] == old:
                id1 = new
            if edges[i][1] == old:
                id2 = new
            
            new_edges.append((id1, id2))
    
        tmp_matrix = self.edge_list_to_adjacency_matrix(new_edges)
        
        if self.has_cycle(tmp_matrix):
            print("The result graph is not a DAG")
        else:
            self.adjacency_matrix = tmp_matrix
            self.adjacency_matrix = self.edge_list_to_adjacency_matrix(new_edges)
            self.adjacency_matrix_T = self.adjacency_matrix.T
            self.size_matrix = len(self.adjacency_matrix)

            self.G = nx.DiGraph(self.adjacency_matrix)
            self.G_T = nx.DiGraph(self.adjacency_matrix_T)

    def create_random_string_from_date(self):

        today = date.today()
        year_str = str(today.year)
        month_str = str(today.month).zfill(2)  # Pad month with leading zero if needed
        day_str = str(today.day).zfill(2)  # Pad day with leading zero if needed

        date_parts = list(year_str + month_str + day_str)
        random.shuffle(date_parts)  # Shuffle the digits randomly

        random_string = "".join(date_parts[:5])  # Take the first 5 digits
        return random_string


    def linkElements(self, element1, element2, procName=""):
        edges = self.adjacency_matrix_to_edge_list(self.adjacency_matrix)
        isProcessIncluded = (self.dic_vertex_names[edges[1][0]][0:5] == "proc_" or self.dic_vertex_names[edges[1][1]][0:5] == "proc_")
        if element1[0:5] == "proc_" or element1[0:5] == "proc_":
            return
        new_edges = copy.deepcopy(edges)
        if element1 not in self.dic_vertex_id or element2 not in self.dic_vertex_id:
            return
        id1 = self.dic_vertex_id[element1]
        id2 = self.dic_vertex_id[element2]
        if isProcessIncluded:
            if procName == "":
                procName = "proc_new" + self.create_random_string_from_date()
            newID = max([max(e[0], e[1]) for e in edges]) + 1
            new_edges.append((id1, newID))
            new_edges.append((newID, id2))
            self.vertex_names.append(procName)
            self.dic_vertex_names[newID] = procName
            self.dic_vertex_id[procName] = newID
        else:
            new_edges.append((id1, id2))
    
        tmp_matrix = self.edge_list_to_adjacency_matrix(new_edges)
        
        if self.has_cycle(tmp_matrix):
            print("The resulting graph is not a DAG")
        else:
            self.adjacency_matrix = tmp_matrix
            self.adjacency_matrix = self.edge_list_to_adjacency_matrix(new_edges)
            self.adjacency_matrix_T = self.adjacency_matrix.T
            self.size_matrix = len(self.adjacency_matrix)

            self.G = nx.DiGraph(self.adjacency_matrix)
            self.G_T = nx.DiGraph(self.adjacency_matrix_T)
            
    def has_cycle(self, adjacency_matrix):
        visited = [False] * len(adjacency_matrix)
        recursively_visited = [False] * len(adjacency_matrix)

        def dfs(node):
            visited[node] = True
            recursively_visited[node] = True

            for neighbor in range(len(adjacency_matrix)):
                if adjacency_matrix[node][neighbor] == 1:
                    if visited[neighbor] and recursively_visited[neighbor]:
                        return True  # Cycle detected
                    elif not visited[neighbor]:
                        if dfs(neighbor):
                            return True

            recursively_visited[node] = False
            return False

        for i in range(len(adjacency_matrix)):
            if not visited[i]:
                if dfs(i):
                    return True

        return False


    def drawOrigins(self, target_vertex, title="", figsize=None, showWeight=False):

        if isinstance(target_vertex, str):
            if target_vertex not in self.dic_vertex_id:
                print(target_vertex + " is not an element")
                return
            target_vertex = self.dic_vertex_id[target_vertex]

        # Draw the path TO the target
        res_vector = np.array([np.zeros(self.size_matrix) for i in range(self.size_matrix+1)])
        res_vector[0][target_vertex] = 1
        for i in range(self.size_matrix):
            if sum(res_vector[i]) == 0:
                break
            res_vector[i+1] = self.adjacency_matrix @ res_vector[i]

        res_vector_T = res_vector.T

        selected_vertices1 = set()
        selected_vertices2 = set()
        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
#                 if j == 0:
#                     continue
#                 if res_vector[i][j] != 0 and j != 0: 
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
#                 comment out to prevent back directing
#                 if j in posfill:
#                     continue
#                 if j == 0:
#                     continue
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
            for iii in range(1):
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
        title += " (" + str(len(set([v[0] for k, v in position.items() if self.dic_vertex_names[k][0:5] != "proc_"]))-1) + " stages)"
    
        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]
        
        if figsize is None:
            figsize = (12, 8)
        self.draw_selected_vertices_reverse_proc(self.G, selected_vertices1,selected_vertices2, selected_vertices3, 
                                title=title, node_labels=node_labels, pos=position, figsize=figsize, showWeight=showWeight)

        
    def drawOffsprings(self, target_vertex, title="", figsize=None, showWeight=False):
        
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
        res_vector = np.array([np.zeros(self.size_matrix) for i in range(self.size_matrix+1)])
        res_vector[0][target_vertex] = 1

        for i in range(len(res_vector)):
            colpos[i] = 0

        for i in range(self.size_matrix):
            if sum(res_vector[i]) == 0:
                break
            res_vector[i+1] = self.adjacency_matrix_T @ res_vector[i]

        for i in range(len(res_vector)):
            if sum(res_vector[i]) == 0:
                break
            for j in range(len(res_vector[i])):
#                 if res_vector[i][j] != 0 and j != self.size_matrix and j != 0: 
                if res_vector[i][j] != 0 and j != self.size_matrix: 
                    if self.dic_vertex_names[j][0:5] == "proc_":
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
#                 comment out to prevent back directing
#                 if j in posfill:
#                     continue
#                 if j == 0 or j == self.size_matrix:
#                 if j == self.size_matrix:
#                     continue
#                 if j == last_pos:
#                     continue
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
            for iii in range(1):
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
        title += " (" + str(len(set([v[0] for k, v in position.items() if self.dic_vertex_names[k][0:5] != "proc_"]))-1) + " stages)"
        if figsize is None:
            figsize = (12, 8)
            
        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]

        self.draw_selected_vertices_reverse_proc(self.G_T, selected_vertices1,selected_vertices2, selected_vertices3, 
                        title=title, node_labels=node_labels, pos=position, reverse=True, figsize=figsize, showWeight=showWeight)

        
    def showSourceNodes(self):
        sources = [node for node in self.G.nodes() if len(list(self.G.predecessors(node))) == 0]

        print("Source Nodes:", [self.dic_vertex_names[s] for s in sources])
        
        
    def showSinkNodes(self):
        sinks = [node for node in self.G.nodes() if len(list(self.G.successors(node))) == 0]

        print("Sink Nodes:", [self.dic_vertex_names[s] for s in sinks])
        
    def showStats(self, g):
        
        in_degree_centrality = nx.in_degree_centrality(g)
        out_degree_centrality = nx.out_degree_centrality(g)
        closeness_centrality = nx.closeness_centrality(g)
        betweenness_centrality = nx.betweenness_centrality(g)
        
        cnt_max = 5
        print("In Degree Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in in_degree_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 4))
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
        
        print("Closeness Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in closeness_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 3))
            cnt += 1
        print("")
        print("Betweenness Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in betweenness_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 3))
            cnt += 1
        print("")
        
    def showBipartiteStats(self, g):
        is_bipartite, node_sets = nx.bipartite.sets(g)
        projection = nx.bipartite.projected_graph(g, node_sets)
        degree_centrality = nx.degree_centrality(projection)
        closeness_centrality = nx.closeness_centrality(projection)
        betweenness_centrality = nx.betweenness_centrality(projection)
        print("Degree Centrality:")
        cnt_max = 5
        cnt = 0
        for z in sorted([(v, k) for k, v in degree_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 3))
            cnt += 1
        print("")
        print("Closeness Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in closeness_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 3))
            cnt += 1
        print("")
        print("Betweenness Centrality:")
        cnt = 0
        for z in sorted([(v, k) for k, v in betweenness_centrality.items()], reverse=True):
            if cnt == cnt_max:
                break
            print(self.dic_vertex_names[z[1]], round(z[0], 3))
            cnt += 1
        print("")
        
    def drawFromLargestComponent(self, figsize=(30, 30), showWeight=False):
        
        connected_components = list(nx.weakly_connected_components(self.G))
        largest_connected_component = None
        sorted_graphs = sorted([[len(c), c] for c in connected_components], reverse=True)
        for i in range(len(sorted_graphs)):
            if i == 0:
                largest_connected_component = sorted_graphs[i][1]
        largest_G = self.G.subgraph(largest_connected_component)
        
        # Find the topological order
        topological_order = list(nx.topological_sort(largest_G))
        self.drawOffsprings(topological_order[0], figsize=figsize, showWeight=showWeight)
        self.drawOrigins(topological_order[-1], figsize=figsize, showWeight=showWeight)
        



mydag = DataJourneyDAG()
# mydag.data_import('/kaggle/input/matrix2/adjacency_matrix2.txt')
# mydag.data_import('dag_data_with_headers.txt')
mydag.data_import('dag_data9.txt')
# mydag.data_import('edge_list.txt', is_edge_list=True)
mydag.write_edge_list_to_file('edge_list2.txt')
# mydag.write_adjacency_matrix_to_file('dag_data9.txt')

mydag.genProcesses()
mydag.coupleProcesses("proc_b", "proc_d")
mydag.drawOrigins(250)
mydag.drawOffsprings(250)


