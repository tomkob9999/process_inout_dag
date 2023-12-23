# Data Journey DAG

### Input: tab-delimited file containing adjacency matrix (rows=FROM, columns=TO) with headers (adjacency_matrix2.txt as a sample)
### Output: Graph of origin data and offspring data for the specified data element

After processes are created between data fields by genProcesses() command, the graph becomes a bipartite graph and it shows as follows.  Processes can be merged by coupleProcesses() as well.

![aa14](https://github.com/tomkob9999/data_journey_dag/assets/96751911/9def45aa-b027-41e7-99d9-720ddacbe3f0)

![aa15](https://github.com/tomkob9999/data_journey_dag/assets/96751911/e1d609b4-3bf4-4dc1-83c3-eea387a051d2)

Data Process Jouney Chart

If there is no universally accepted name for this chart, so I would call it Data Processing Journey Chart.  It has the following properties.
- The underlying graph data is a DAG
- The underlying graph data is a bipartite graph.
- The chart starts and ends with a data eleent.  They data element and the process should show side by side and each column should only show either of them.

And furthermore, a Data Process Jouney Chart is said to be the Ideal Form, which guarantees maximum parallelism and minimum inter-process dependencies, has the following properties.
- Each data element can have only one incoiming process
- All the incoming data element will go to the same process if they share the same output data field.
- Maximum parallelism
- Minimum inter-process dependencies
- The number of stages is minimal.

  The data element inter-dependencies are business invariants, and cannot be altered while the resources and the process arrangements and efficiency themselves can be varied.  The DAG represents the ideal state of process arrangements that bears the least stages, the maximum parallelism and the least dependencies.  In reality, the resource allocation and the process arrangements/efficiency are constraints, and the actual implementation of process always elude away from the ideal state, so the generated graph can be utilized compare between the reality and the ideal.
  
Hilights:

- The graph is kept as a DAG.
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
  
