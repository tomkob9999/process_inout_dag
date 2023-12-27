# Data Journey DAG

### Input: tab-delimited file containing adjacency matrix (rows=FROM, columns=TO) with headers (adjacency_matrix2.txt as a sample)
### Output: Graph of origin data and offspring data for the specified data element

After processes are created between data fields by genProcesses() command, the graph becomes a bipartite graph and it shows as follows.  Processes can be merged by coupleProcesses() as well.

![aa14](https://github.com/tomkob9999/data_journey_dag/assets/96751911/9def45aa-b027-41e7-99d9-720ddacbe3f0)

![aa15](https://github.com/tomkob9999/data_journey_dag/assets/96751911/e1d609b4-3bf4-4dc1-83c3-eea387a051d2)

Well-Alligned Bipartite-DAG Layout

Graphs's are displayed in the well aligned layout.

- It has the following properties.
- The underlying graph data is a DAG
- The underlying graph data is a bipartite graph.
- The chart starts and ends with a data eleent.  They data element and the process should show side by side and each column should only show either of them.

It has two forms: compact and stretched.  Both locate the horizontal positions at the expected completion time.  The compact form uses the assumption that each process takes the same unit length of 1.  The stretched form uses the weights.  The compact form has better overall visibility and better view of parallelism.  The stretched form shows more relistic view of how each process should flow.

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


With edge weights added, this tool can be used as  the schedule planning tool where the inputs and outputs are specified as data elements.  The existing longest path identifies the critical path.

With showWeight option on, the critical path is shown as brown line and the brightness indicates higher the criticality of the process.

![aa23](https://github.com/tomkob9999/data_journey_dag/assets/96751911/66a208d6-7640-474d-9275-b14732b793b8)

It also shows some statistics.

![aa24](https://github.com/tomkob9999/data_journey_dag/assets/96751911/dbca9a1c-e72f-4b74-8fd5-2e74b4aa2eb9)

Still, the initial and the main analysis should be done without the weights to identify the ideal flow and the stages needed assuming all the processes take the same unit duration because the analysis is purely based on the input/output invariants.  Weights are much less reliable information as they merely represent the current suboptimal state.



