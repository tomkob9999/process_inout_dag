# Data Journey DAG

### Input: tab-deliited file containing adjacency matrix (rows=FROM, columns=TO) with headers (adjacency_matrix2.txt as a sample)
### Output: Graph of origin data and offspring data for the specified data element

After processes are created between data fields by genProcesses() command, the graph becomes a bipartite graph and it shows as follows.  Some processes are merged by coupleProcesses() as well.

![aa11](https://github.com/tomkob9999/data_journey_dag/assets/96751911/3b517922-35ae-4ed4-bfce-11a1ae93b7eb)



Implemented Features
- Edge List file read/write DONE
- generateProcesses  genProcesses() DONE
- DAG check  checkDAG(from, to) DONE
- Process coupling  (process1, process2) DONE

Planned Features
- Add Node  addField(name) NOT YET
  
