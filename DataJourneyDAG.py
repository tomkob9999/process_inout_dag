# Verwsion: 1.0.9
# Last Update: 2023/12/21
# Author: Tomio Kobayashi

# - generateProcesses  genProcesses() DONE
# - DAG check  checkDAG(from, to) DONE
# - Process coupling  (process1, process2) DONE
# - Add Node  addField(name) NOT YET
# - Link Node  linkFields(from, to) NOT NEEDED
  
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

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
        """
        Converts an adjacency matrix to an edge list.

        Args:
            adjacency_matrix: A 2D list representing the adjacency matrix.

        Returns:
            A list of tuples, where each tuple represents an edge (source node, target node).
        """

        edge_list = []
        num_nodes = len(adc_matrix)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if adc_matrix[i][j] == 1:
                    edge_list.append((i, j))  # Add edges only for non-zero entries

        return edge_list

    def draw_selected_vertices_reverse_proc(self, G, selected_vertices1, selected_vertices2, selected_vertices3, title, node_labels, pos, reverse=False):

        # Create a subgraph with only the selected vertices
        subgraph1 = G.subgraph(selected_vertices1)
        subgraph2 = G.subgraph(selected_vertices2)
        subgraph3 = G.subgraph(selected_vertices3)
        
        if reverse:
            subgraph1 = subgraph1.reverse()
            subgraph2 = subgraph2.reverse()
            subgraph3 = subgraph3.reverse()

        # Set figure size to be larger
        plt.figure(figsize=(12, 8))

        # Draw the graph
        nx.draw(subgraph1, pos, with_labels=True, labels=node_labels, node_size=1000, node_color='skyblue', font_size=10, font_color='black', arrowsize=10, edgecolors='black')
        nx.draw(subgraph2, pos, with_labels=True, labels=node_labels, node_size=1000, node_color='orange', font_size=10, font_color='black', arrowsize=10, edgecolors='black')
        nx.draw(subgraph3, pos, with_labels=True, labels=node_labels, node_size=1500, node_color='pink', font_size=10, font_color='black', arrowsize=10, edgecolors='black')

        plt.title(title)
        plt.show()
        
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

        is_cyclic = self.has_cycle(self.adjacency_matrix)
#         print("is_cyclic", is_cyclic)
        
        self.adjacency_matrix_T = self.adjacency_matrix.T

        # Matrix of row=FROM, col=TO
        self.size_matrix = len(self.adjacency_matrix)

        self.G = nx.DiGraph(self.adjacency_matrix)
        self.G_T = nx.DiGraph(self.adjacency_matrix_T)
        
        for i in range(len(self.vertex_names)):
            self.dic_vertex_names[i] = self.vertex_names[i]
            self.dic_vertex_id[self.vertex_names[i]] = i
        
        
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
        """
        Writes a matrix to a tab-delimited text file.

        Args:
            matrix: The matrix to write.
            filename: The name of the file to write to.
        """

        with open(filename, "w") as file:
            file.write("\t".join(self.vertex_names) + "\n") 
            for row in self.adjacency_matrix:
                file.write("\t".join(map(str, row)) + "\n")  # Join elements with tabs and add newline


    def genProcesses(self):
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
            new_edges.append((edges[i][0], new_id))
            new_edges.append((new_id, edges[i][1]))
    
        self.adjacency_matrix = self.edge_list_to_adjacency_matrix(new_edges)
        self.adjacency_matrix_T = self.adjacency_matrix.T
        self.size_matrix = len(self.adjacency_matrix)

        self.G = nx.DiGraph(self.adjacency_matrix)
        self.G_T = nx.DiGraph(self.adjacency_matrix_T)
# Example usage

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
    
    def has_cycle(self, adjacency_matrix):
#        """
#        Checks if the given adjacency matrix represents a DAG.

#        Args:
#            adjacency_matrix: The adjacency matrix to check.

#        Returns:
#            True if the matrix contains a cycle, False otherwise.
#        """
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


    def drawOrigins(self, target_vertex):


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
                if j == 0:
                    continue
                if res_vector[i][j] != 0 and j != 0: 
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
                if j in posfill:
                    continue
                if j == 0:
                    continue
                if res_vector[i][j]:
                    posfill.add(j)
                    position[j] = ((last_pos-i), colpos[(last_pos-i)]) 
                    colpos[(last_pos-i)] += 1
                    if largest_j < j:
                        largest_j = j
                        
        maxheight = max([v for k, v in colpos.items() if v != 0])
        newpos = {}
        for k, v in position.items():
            newheight = v[1]
            if colpos[v[0]] != maxheight:
                if maxheight == 2 and colpos[v[0]] == 1:
                    newheight = 0.5
                else:
                    newheight = maxheight/(colpos[v[0]]+1)*(newheight+1)
            newpos[k] = (v[0], newheight)
        position = newpos

        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]
        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}
        title = "Data Origins (" + str(len(set([v[0] for k, v in position.items() if self.dic_vertex_names[k][0:5] != "proc_"]))-1) + " stages)"
        
        self.draw_selected_vertices_reverse_proc(self.G, selected_vertices1,selected_vertices2, selected_vertices3, title=title, node_labels=node_labels, pos=position)

        
    def drawOffsprings(self, target_vertex):

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
                if res_vector[i][j] != 0 and j != self.size_matrix and j != 0: 
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
                if j in posfill:
                    continue
                if j == 0 or j == self.size_matrix:
                    continue
                if res_vector[i][j]:
                    posfill.add(j)
                    position[j] = (i, colpos[i]) 
                    colpos[i] += 1
                    if largest_j < j:
                        largest_j = j

#         re-align the vertical position
        maxheight = max([v for k, v in colpos.items() if v != 0])
        newpos = {}
        for k, v in position.items():
            newheight = v[1]
            if colpos[v[0]] != maxheight:
                if maxheight == 2 and colpos[v[0]] == 1:
                    newheight = 0.5
                else:
                    newheight = maxheight/(colpos[v[0]]+1)*(newheight+1)
            newpos[k] = (v[0], newheight)
                    
        position = newpos

        selected_vertices1 = list(selected_vertices1)
        selected_vertices2 = list(selected_vertices2)
        selected_vertices3 = [target_vertex]

        node_labels = {i: name for i, name in enumerate(self.vertex_names) if i in selected_vertices1 or i in selected_vertices2}

        title = "Data Offsprings (" + str(len(set([v[0] for k, v in position.items() if self.dic_vertex_names[k][0:5] != "proc_"]))-1) + " stages)"
        self.draw_selected_vertices_reverse_proc(self.G_T, selected_vertices1,selected_vertices2, selected_vertices3, title=title, node_labels=node_labels, pos=position, reverse=True)



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


