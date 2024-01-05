# Data Journey DAG

### Input: tab-delimited file containing adjacency matrix (rows=FROM, columns=TO) with headers (adjacency_matrix2.txt as a sample)
### Output: Graph of origin data and offspring data for the specified data element

After processes are created between data fields by genProcesses() command, the graph becomes a bipartite graph and it shows as follows.  Processes can be merged by coupleProcesses() as well.

![aa5](https://github.com/tomkob9999/data_journey_dag/assets/96751911/01355890-0545-49b8-b654-571cd87ef539)


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


With edge weights added, this tool can be used as the schedule planning tool where the inputs and outputs are specified as data elements.  The existing longest path identifies the critical path.

With showWeight option on, the critical path is shown as brown line and the brightness indicates higher the criticality of the process.

Compatct form

![aa2](https://github.com/tomkob9999/data_journey_dag/assets/96751911/2eb1cadd-b467-4fad-9608-bd0281ea50ee)


Stretched form

![aa4](https://github.com/tomkob9999/data_journey_dag/assets/96751911/f84cc930-4ab6-4183-936e-04207dd4efe1)


The stretch form has a feature of showing edges with wait time in light blue. It shows its wait time in the bracket next to the weight.
For example, in the diagram below, proc_COL88 flows into proc_COL90 and the wedge shows 2 (& 2).  proc_COL90 starts 2 clocks, or steps, after proc_COL88 arrives because it also has proc_COL86 as the pre-condition and it needs to wait for its arrival.


![aa3](https://github.com/tomkob9999/data_journey_dag/assets/96751911/62c1cb9c-7c0c-47bf-9e70-b452f28e8f97)




Still, the initial and the main analysis should be done without the weights to identify the ideal flow and the stages needed assuming all the processes take the same unit duration because the analysis is purely based on the input/output invariants.  Weights are much less reliable information as they merely represent the current suboptimal state.



