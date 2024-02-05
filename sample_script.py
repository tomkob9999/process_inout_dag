import time

def measure(start_time):
    end_time = time.time()

    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    milliseconds = int(elapsed_time * 1000 % 1000)
    return f"{int(minutes)}:{str(int(seconds)).zfill(2)}:{str(int(milliseconds)).zfill(3)}"
           
start_time = time.time()

showWeight = True

mydag = ProcessInOutDAG()
# mydag.data_import('/kaggle/input/matrix3/dag_data.txt')
# mydag.drawOrigins(250)
# mydag.data_import('/kaggle/input/matrix2/adjacency_matrix2.txt')
# mydag.data_import('/kaggle/input/matrix11/adjacency_matrix11.txt')
# mydag.data_import('/kaggle/input/matrix6/dag_data6.txt')  # 2000, scarce
mydag.data_import('/kaggle/input/matrix12/weighted_matrix12.txt')
# mydag.data_import('/kaggle/input/inp-weight4/inp_weight_list4.txt', is_oup_list=True)
# mydag.data_import('/kaggle/input/inp-weight5/inp_weight_list5.txt', is_oup_list=True) # Japanese, but garbled
# mydag.data_import('/kaggle/input/inp-weight6/inp_weight_list6.txt', is_oup_list=True)

print("data_import() finished", measure(start_time))
start_time = time.time()

mydag.read_cond_list_from_file('/kaggle/input/cond-list3/cond_list1.txt')
print("read_cond_list_from_file() finished", measure(start_time))
start_time = time.time()

mydag.read_opt_list_from_file('/kaggle/input/cond-list3/opt_list1.txt')
print("read_opt_list_from_file() finished", measure(start_time))
start_time = time.time()

# mydag.drawFromLargestComponent(figsize=(12, 8), showWeight=False)

# print("drawFromLargestComponent() finished", measure(start_time))
# start_time = time.time()



# mydag.genProcesses()

# print("genProcesses() finished", measure(start_time))
# start_time = time.time()

# # mydag.drawFromLargestComponent(figsize=(50, 50), showWeight=True)
# mydag.drawFromLargestComponent(figsize=(12, 8), showWeight=False)

# print("drawFromLargestComponent() finished", measure(start_time))
# start_time = time.time()


mydag.populateStretch()
print("populateStretch() finished", measure(start_time))
start_time = time.time()


# mydag.drawOffsprings("COL34", showWeight=showWeight)
# print("drawOffsprings() finished", measure(start_time))
# start_time = time.time()


# mydag.drawOriginsStretch("COL34", figsize=(13,8), showWeight=showWeight)
# print("drawOriginsStretch() finished", measure(start_time))
# start_time = time.time()

# mydag.drawOffspringsStretch("COL34", figsize=(13,8), showWeight=showWeight)
# print("drawOffspringsStretch() finished", measure(start_time))
# start_time = time.time()

# mydag.drawOrigins("EXIT", showWeight=showWeight)
# mydag.drawOrigins("COL99", showWeight=showWeight)
mydag.drawOrigins("COL89", showWeight=showWeight)
print("drawOrigins() finished", measure(start_time))
start_time = time.time()

mydag.drawOriginsStretch("COL89", figsize=(13,8), showWeight=showWeight)
print("drawOriginsStretch() finished", measure(start_time))
start_time = time.time()

mydag.drawOffsprings("COL30", showWeight=showWeight)
print("drawOffsprings() finished", measure(start_time))
start_time = time.time()

mydag.drawOffspringsStretch("COL30", figsize=(13,8), showWeight=showWeight)
print("drawOffspringsStretch() finished", measure(start_time))
start_time = time.time()


# mydag.drawOriginsStretch("COL89", figsize=(13,8), showWeight=showWeight)
# print("drawOriginsStretch() finished", measure(start_time))
# start_time = time.time()

# mydag.drawOffspringsStretch("COL70", figsize=(13,8), showWeight=showWeight)
# print("drawOffspringsStretch() finished", measure(start_time))
# start_time = time.time()

# mydag.drawOffspringsStretch("COL50", figsize=(13,8), showWeight=showWeight)
# print("drawOffspringsStretch() finished", measure(start_time))
# start_time = time.time()


# mydag.drawOffsprings(0, figsize=(50, 50))

# print("drawOffsprings() finished", measure(start_time))
# start_time = time.time()

# mydag.coupleProcesses("proc_COL33", "proc_COL32")
# print("coupleProcesses() finished", measure(start_time))
# start_time = time.time()
# mydag.coupleProcesses("proc_COL29", "proc_COL22")
# print("coupleProcesses() finished", measure(start_time))
# start_time = time.time()
# mydag.coupleProcesses("proc_COL20", "proc_COL22")
# print("coupleProcesses() finished", measure(start_time))
# start_time = time.time()
# mydag.coupleProcesses("proc_COL27", "proc_COL22")
# print("coupleProcesses() finished", measure(start_time))
# start_time = time.time()
# mydag.coupleProcesses("proc_COL24", "proc_COL23")
# print("coupleProcesses() finished", measure(start_time))
# start_time = time.time()
# mydag.coupleProcesses("proc_COL25", "proc_COL23")
# print("coupleProcesses() finished", measure(start_time))
# start_time = time.time()


# for i in range(20, 24, 1):
#     mydag.drawOrigins(i, showWeight=True)
#     print("drawOrigins() finished", measure(start_time))
#     start_time = time.time()
#     mydag.drawOffsprings(i, showWeight=True)
#     print("drawOffsprings() finished", measure(start_time))
#     start_time = time.time()
# for i in range(1000, 1005, 1):
#     mydag.drawOrigins(i, showWeight=True)
#     print("drawOrigins() finished", measure(start_time))
#     start_time = time.time()
#     mydag.drawOffsprings(i, showWeight=True)
#     print("drawOffsprings() finished", measure(start_time))
#     start_time = time.time()


# print("drawOffsprings() finished", measure(start_time))