# Data Journey DAG

### Input: tab-delimited file containing adjacency matrix (rows=FROM, columns=TO) with headers (adjacency_matrix2.txt as a sample)
### Output: Graph of origin data and offspring data for the specified data element

After processes are created between data fields by genProcesses() command, the graph becomes a bipartite graph and it shows as follows.  Processes can be merged by coupleProcesses() as well.

![a12](https://github.com/tomkob9999/data_journey_dag/assets/96751911/589fe347-05d5-4db6-99c0-3dd70cff0145)

### After the processes are generated by genProcesses(), the number of stages and the number of processes are guaranteed to be minimum while maximum parallelism and minimum dependencies between elements/processes.  Further coupling of procsses will cause elements/processes dependencies and thus bottleneck risk.  Further sequencing of processes will cause extra stages and longer throughputs.
  The data dependencies are business invariants, and cannot be altered while the resources and the process arrangements(sequential,parallel,small,big,etc) and efficiency themselves can be improved or degraded.  Therefore, the DAG represents the ideal state of how the processes are positions that bears the maximum parallelism and the least dependencies.  In reality, the resource allocation and the process arrangements/efficiency are constraints and the actual implementation of process ellude away from the ideal state, so the graph should present as the valuable comparison between the reality and the ideal. 
  
Hilights:

- The graph is kept to be a DAG.
- After the processes are added, the graph becomes a bipartite graph of elements and processes as well.
- compatible with adjacency matrix and edge list formats.
- The generated flow graphs show as topologically ordered.
- any node with the prefix "proc_" is regarded as a process, and everything else is regarded as an element by the tool.

Implemented Features
- Edge List file read/write DONE
- generateProcesses  genProcesses() DONE
- DAG check  checkDAG(from, to) DONE
- Process coupling  (process1, process2) DONE
- Link Elements  linkElements(element1, element1) DONE

Planned Features
- Add Node  addField(name) NOT YET
  
