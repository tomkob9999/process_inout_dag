, labels=np.unique(res))
                f1 = f1_score(answer, res)
#                 f1 = f1_score(answer, res, average='weighted', labels=np.unique(res))
#                 f1 = f1_score(answer, res, average='weighted', labels=np.unique(res))
#                 print(str(sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])) + "/" + str(len(res)), " records matched")
                print(str(sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])) + "/" + str(len(res)), " records matched " + f" ({sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])/len(res)*100:.2f}%)")
                print(f"Precision: {precision * 100:.2f}%")
                print(f"Recall: {recall * 100:.2f}%")
                print(f"F1 Score: {f1 * 100:.2f}%")

                self.opt_f1 = f1
                
                print(self.replaceSegName(final_expr))
                
                print("")
                
                self.expression_opt = final_expr
                
                return final_expr
            
            
    def random_split_matrix(matrix, divide_by=2):
        matrix = copy.deepcopy(matrix)
        rows = list(matrix)  # Convert to list for easier shuffling
        random.shuffle(rows)  # Shuffle rows in-place
        split_index = len(rows) // divide_by  # Integer division for equal or near-equal halves
        return rows[:split_index], rows[split_index:]

    def train_and_optimize(self, data_list=None, max_dnf_len=4, error_tolerance=0.02, 
                       min_match=0.03, use_approx_dnf=False, redundant_thresh=1.00, elements_count_penalty=1.0, 
                           use_compact_opt=False, cnt_out=20, useUnion=False, useExpanded=False):
        
        print("Training started...")
        
        headers = data_list[0]
        data_list2 = data_list[1:]
        
        train_data, valid_data = Deterministic_Regressor.random_split_matrix(data_list2)

        train_inp = [headers] + train_data
        
        self.train(data_list=train_inp, max_dnf_len=max_dnf_len, 
                        error_tolerance=error_tolerance, min_match=min_match, use_approx_dnf=use_approx_dnf, redundant_thresh=redundant_thresh, useExpanded=useExpanded)

        print("Optimization started...")
        inp = [headers] + valid_data
           
        inp = [[Deterministic_Regressor.try_convert_to_numeric(inp[i][j]) for j in range(len(inp[i]))] for i in range(len(inp))]
                
        answer = [int(inp[i][-1]) for i in range(1, len(inp), 1)]
        inp = [row[:-1] for row in inp]
             
        if use_compact_opt:
            return self.optimize_compact(inp, answer, cnt_out=cnt_out, useUnion=useUnion)
        else:
            return self.optimize_params(inp, answer, elements_count_penalty=1.0, useUnion=useUnion)
    
    def train_and_optimize_bulk(self, data_list, expected_answers, max_dnf_len=4, error_tolerance=0.02,  
                   min_match=0.03, use_approx_dnf=False, redundant_thresh=1.00, elements_count_penalty=1.0, use_compact_opt=False, cnt_out=20, useUnion=False, useExpanded=False):

        self.children = [Deterministic_Regressor() for _ in range(len(expected_answers))]

        for i in range(len(self.children)):
            print("Child", i)
            
            self.children[i].dic_segments = copy.deepcopy(self.dic_segments)
            d_list = copy.deepcopy(data_list)
            d_list[0].append("res")
            
            for k in range(len(d_list)-1):
                d_list[k+1].append(expected_answers[i][k])
            self.children[i].train_and_optimize(data_list=d_list, max_dnf_len=max_dnf_len, error_tolerance=error_tolerance, 
                    min_match=min_match, use_approx_dnf=use_approx_dnf, redundant_thresh=redundant_thresh, 
                                                elements_count_penalty=elements_count_penalty, use_compact_opt=use_compact_opt, cnt_out=cnt_out, useUnion=useUnion, useExpanded=useExpanded)

            
            
    def solve_with_opt_bulk(self, inp_p):
        res = []
        for c in self.children:
            r = c.solve_with_opt(inp_p)
            if r == None or len(r) == 0:
                r == [0] * (len(inp_p)-1)
            res.append(r)
            
        return res
    
    def solve_with_opt_class(self, inp_p):
        
        res = self.solve_with_opt_bulk(inp_p)
        
        dic_f1 = {i: self.children[i].opt_f1 for i in range(len(self.children))}
        
        len_rows = len(res[0])
        len_res = len(res)
        new_res = [0] * len_rows
        for i in range(len_rows):
            numbers = [s[1] for s in sorted([(random.random()*dic_f1[i], i) for i in range(len_res)], reverse=True)]
            for k in range(len(numbers)):
                if res[numbers[k]][i] == 1:
                    new_res[i] = numbers[k]
                    break
                if k == len(numbers) - 1:
                    new_res[i] = numbers[k]
                    

        return new_res
    
    def train_and_optimize_class(self, data_list, expected_answers, max_dnf_len=4, error_tolerance=0.02, 
               min_match=0.03, use_approx_dnf=False, redundant_thresh=1.00, elements_count_penalty=1.0, use_compact_opt=False, cnt_out=10, useUnion=False, useExpanded=False):
        
        answers = [[0 for _ in range(len(expected_answers))] for _ in range(max(expected_answers)+1)]
        for i in range(len(answers[0])):
            answers[expected_answers[i]][i] = 1
        self.train_and_optimize_bulk(data_list=data_list, expected_answers=answers, max_dnf_len=max_dnf_len, error_tolerance=error_tolerance, 
                    min_match=min_match, use_approx_dnf=use_approx_dnf, redundant_thresh=redundant_thresh, 
                                     elements_count_penalty=elements_count_penalty, use_compact_opt=use_compact_opt, cnt_out=cnt_out, useUnion=useUnion, useExpanded=useExpanded)
    
    def prepropcess(self, whole_rows, by_two, splitter=3):
        self.whole_rows = self.clean_and_discretize(whole_rows, by_two)
        headers = self.whole_rows
        data = self.whole_rows[1:]
        random.shuffle(data)  # Shuffle rows in-place
        split_index = len(data) // splitter  # Integer division for equal or near-equal halves
        self.test_rows = data[:split_index]
#         print("len(self.test_rows)", len(self.test_rows))
        self.train_rows = data[split_index:]
#         print("len(self.train_rows)", len(self.train_rows))
    
    def get_train_dat_wo_head(self):
        return [row[:-1] for row in self.train_rows]
    def get_train_res_wo_head(self):
        return [row[-1] for row in self.train_rows]
    def get_train_dat_with_head(self):
        return [self.whole_rows[0][:-1]] + [row[:-1] for row in self.train_rows]
    def get_train_datres_wo_head(self):
        return self.train_rows
    def get_train_datres_with_head(self):
        return [self.whole_rows[0]] + self.train_rows
    
    def get_test_dat_wo_head(self):
        return [row[:-1] for row in self.test_rows]
    def get_test_res_wo_head(self):
        return [row[-1] for row in self.test_rows]
    def get_test_dat_with_head(self):
        return [self.whole_rows[0][:-1]] + [row[:-1] for row in self.test_rows]
    def get_test_datres_wo_head(self):
        return self.test_rows[1:]
    def get_test_datres_with_head(self):
        return [self.whole_rows[0]] + self.test_rows
    

    def show_stats(predicted, actual, average="weighted", elements_count_penalty=1.0):
        
        if len(predicted) != len(actual):
            print("The row number does not match")
            return
        answer = actual
        res = predicted
        
#         precision = precision_score(answer, res, average=average)
#         recall = recall_score(answer, res, average=average)
#         f1 = f1_score(answer, res, average=average, labels=np.unique(res))
        precision = precision_score(answer, res, average=average, labels=np.unique(res))
        recall = recall_score(answer, res, average=average, labels=np.unique(res))
        f1 = f1_score(answer, res, average=average, labels=np.unique(res))
        print("")
        print("####### PREDICTION STATS #######")
        print("")
        print(str(sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])) + "/" + str(len(res)), " records matched" + f" ({sum([1 if answer[i] == res[i] else 0 for i in range(len(answer))])/len(res)*100:.2f}%)")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")
        print("")
        print("##############")
        print("")
