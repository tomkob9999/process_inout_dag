# Data Journey DAG

### Input: tab-deliited file containing adjacency matrix (rows=FROM, columns=TO) with headers (adjacency_matrix2.txt as a sample)
### Output: Graph of origin data and offspring data for the specified data element

After processes are created between data fields by genProcesses() command, the graph becomes a bipartite graph and it shows as follows.  Some processes are merged by coupleProcesses() as well.

![aa10](https://github.com/tomkob9999/data_journey_dag/assets/96751911/67df56d7-1b83-47f7-be04-286f91bf7894)


Implemented Features
- Edge List file read/write DONE
- generateProcesses  genProcesses() DONE
- DAG check  checkDAG(from, to) DONE
- Process coupling  (process1, process2) DONE

Planned Features
- Add Node  addField(name) NOT YET
  
