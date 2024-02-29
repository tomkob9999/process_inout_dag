# logical_weight
#
# Version: 1.1.6
# Last Update: 2024/02/29
# Author: Tomio Kobayashi
#
# Average of max and min of multiple random variables.  In case of error, securely. max and min are returned

import scipy.stats as stats
import numpy as np
import re

class logical_weight:
    def __init__(self):
        self.dic_exp = {}
        
    def lognormal(mu, sigma):
#         return np.log(np.random.lognormal(mu, min(mu, sigma))+1)
        if mu > 500:
            f = np.log(np.random.lognormal(500, sigma)+1)
            ret = mu + f - 500
        else:
            ret = np.log(np.random.lognormal(mu, sigma)+1)
            
        if ret == float('inf'):
            return mu
        else:
            return ret
    
    def get_avg_max(rands, sigma=1., nloop=3000):
        return np.mean([max([np.random.normal(r, sigma) for r in rands]) for i in range(nloop)])

    # Good for non-zero values.  Long tailed
    def get_avg_max_nonzero(rands, sigma=1., nloop=3000):
        ret = np.mean([max([logical_weight.lognormal(r, sigma) for r in rands]) for i in range(nloop)])
        if ret == float('inf'):
            return max(rands)
        else:
            return ret

    def get_avg_min(rands, sigma=1, nloop=3000):
        return np.mean([min([np.random.normal(r, sigma) for r in rands]) for i in range(nloop)])

    # Good for non-zero values.  Long tailed
    def get_avg_min_nonzero(rands, sigma=1., nloop=3000):
        ret = np.mean([min([logical_weight.lognormal(r, sigma) for r in rands]) for i in range(nloop)])
        if ret == float('inf'):
            return min(rands)
        else:
            return ret


    def calc_avg_result_weight(inp_exp, weights, use_lognormal=False, loop_limit=2000, sigma=1.):

        exp = inp_exp

        for k, v in weights.items():
            exp = exp.replace(k, str(v*1.0))
        pattern_and = "\((\s*[^\|\(\)]*\s*&\s[^\|\(\)]*\s*)+\)"
        pattern_or = "\((\s*[^\|\(\)]*\s*\|\s[^\|\(\)]*\s*)+\)"

        cnt = 1
        while True:
            cnt += 1
            if cnt > loop_limit:
                print("too many loops")
                break
            if re.search("^\d*(\.\d*)$", exp.strip()):
                break
            exp = "(" + exp.strip() + ")" if exp.strip()[0] != "(" else exp.strip()
            pattern = pattern_and
            match = re.search(pattern, exp)
            if match:
                val_list = [float(s) for s in match.group().replace("(", "").replace(")", "").replace(" ", "").replace("&", ",").replace("|", ",").split(",")]
                out_len = logical_weight.get_avg_max_nonzero(val_list, sigma=sigma) if use_lognormal else logical_weight.get_avg_max(val_list, sigma=sigma)
                exp = re.sub(pattern, str(out_len), exp, count=1)
            else:
                pattern = pattern_or
                match = re.search(pattern, exp)
                if match:
                    val_list = [float(s) for s in match.group().replace("(", "").replace(")", "").replace(" ", "").replace("&", ",").replace("|", ",").split(",")]
                    out_len = logical_weight.get_avg_min_nonzero(val_list, sigma=sigma) if use_lognormal else logical_weight.get_avg_min(val_list, sigma=sigma)
                    exp = re.sub(pattern, str(out_len), exp, count=1)
                else:
                    break
#         return float(exp) 
        return logical_weight.get_avg_max_nonzero([float(exp)], sigma=sigma) if use_lognormal else logical_weight.get_avg_max([float(exp)], sigma=sigma)

    def calc_avg_result_weight_memo(self, inp_exp, weights, use_lognormal=False, loop_limit=2000, nloop=3000, sigma=1.):

        exp = inp_exp

        for k, v in weights.items():
            exp = exp.replace(k, str(v*1.0))

        if exp in self.dic_exp:
            return self.dic_exp[exp]
        start_exp = exp
        
        pattern_and = "\((\s*[^\|\(\)]*\s*&\s[^\|\(\)]*\s*)+\)"
        pattern_or = "\((\s*[^\|\(\)]*\s*\|\s[^\|\(\)]*\s*)+\)"

        cnt = 1
        while True:
            cnt += 1
            if cnt > loop_limit:
                print("too many loops")
                break
            if re.search("^\d*(\.\d*)$", exp.strip()):
                break
            exp = "(" + exp.strip() + ")" if exp.strip()[0] != "(" else exp.strip()
            pattern = pattern_and
            match = re.search(pattern, exp)
            if match:
                val_list = [float(s) for s in match.group().replace("(", "").replace(")", "").replace(" ", "").replace("&", ",").replace("|", ",").split(",")]
                out_len = logical_weight.get_avg_max_nonzero(val_list, sigma=sigma, nloop=nloop) if use_lognormal else logical_weight.get_avg_max(val_list, sigma=sigma, nloop=nloop)
                exp = re.sub(pattern, str(out_len), exp, count=1)
            else:
                pattern = pattern_or
                match = re.search(pattern, exp)
                if match:
                    val_list = [float(s) for s in match.group().replace("(", "").replace(")", "").replace(" ", "").replace("&", ",").replace("|", ",").split(",")]
                    out_len = logical_weight.get_avg_min_nonzero(val_list, sigma=sigma, nloop=nloop) if use_lognormal else logical_weight.get_avg_min(val_list, sigma=sigma, nloop=nloop)
                    exp = re.sub(pattern, str(out_len), exp, count=1)
                else:
                    break
#         return float(exp) 
        
        res = logical_weight.get_avg_max_nonzero([float(exp)], sigma=sigma, nloop=nloop) if use_lognormal else logical_weight.get_avg_max([float(exp)], sigma=sigma, nloop=nloop)
        self.dic_exp[start_exp] = res
        return res