# Data Journey DAG

### Input: tab-deliited file containing adjacency matrix (rows=FROM, columns=TO) with headers (adjacency_matrix2.txt as a sample)
### Output: Graph of origin data and offspring data for the specified data element

After processes are inserted between data fields, the graph becomes a bipartite graph and it shows as follows.

![aa9](https://github.com/tomkob9999/data_journey_dag/assets/96751911/9c60b222-4859-4d18-a87d-3381d9d0e75c)


Implemented Features
- Edge List file read/write DONE
- generateProcesses  genProcesses() DONE
- DAG check  checkDAG(from, to) DONE
- Process coupling  (process1, process2) DONE

Planned Features
- Add Node  addField(name) NOT YET
  
