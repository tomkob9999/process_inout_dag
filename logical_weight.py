# logical_weight
#
# Version: 1.0.3
# Last Update: 2024/02/03
# Author: Tomio Kobayashi
#
# Average of max and min of multiple random variables.  In case of error, securely. max and min are returned
#

import scipy.stats as stats
import numpy as np
import re

class logical_weight:
    def get_avg_max(rands, std_ratio=0.25):
        try:
            sds = [np.abs(r * std_ratio) for r in rands]

            max_prb = [0.0]*len(rands)
            max_value = max(rands)
            max_indices = [index for index, value in enumerate(rands) if value == max_value]
            i = max_indices[0]
            for j in range(len(rands)):
                if j in max_indices:
                    continue
                max_prb[j] = stats.norm.cdf(0, loc=(rands[i] - rands[j]), scale=(sds[i]+sds[j])) 

            maxmax = (1 - sum(max_prb))/len(max_indices)
            for m in max_indices:
                max_prb[m] = maxmax
#             return max(sum([(stats.norm(loc=(rands[i]), scale=(sds[i])).ppf(1-max_prb[i]/2)) * max_prb[i] for i in range(len(rands))]), max(rands))
            val = max(sum([(stats.norm(loc=(rands[i]), scale=(sds[i])).ppf(1-max_prb[i]/2)) * max_prb[i] for i in range(len(rands))]), max(rands))
            return val if not np.isnan(val) else max(rands) 

        except Exception as e:
            return max(rands)

    # Good for non-zero values.  Long tailed
    def get_avg_max_nonzero(rands, std_ratio=0.25, sigma=0.5):
        try:
            sds = [np.abs(r * std_ratio) for r in rands]

            max_prb = [0.0]*len(rands)
            max_value = max(rands)
            max_indices = [index for index, value in enumerate(rands) if value == max_value]
            i = max_indices[0]
            for j in range(len(rands)):
                if j in max_indices:
                    continue
                max_prb[j] = stats.norm.cdf(0, loc=(rands[i] - rands[j]), scale=(sds[i]+sds[j])) 

            maxmax = (1 - sum(max_prb))/len(max_indices)
            for m in max_indices:
                max_prb[m] = maxmax

            val = max(sum([(stats.lognorm(sigma, scale=rands[i]).ppf(1-max_prb[i]/2)) * max_prb[i] for i in range(len(rands))]), max(rands))
            return val if not np.isnan(val) else max(rands) 

        except Exception as e:
            return max(rands)

    def get_avg_min(rands, std_ratio=0.25):
        try:
            sds = [np.abs(r * std_ratio) for r in rands]

            min_prb = [0.0]*len(rands)
            min_value = min(rands)
            min_indices = [index for index, value in enumerate(rands) if value == min_value]

            i = min_indices[0]
            for j in range(len(rands)):
                if j in min_indices:
                    continue
                min_prb[j] = stats.norm.cdf(0, loc=(rands[j] - rands[i]), scale=(sds[i]+sds[j])) 
            minmin = (1 - sum(min_prb))/len(min_indices)
            for m in min_indices:
                min_prb[m] = minmin
            
            val = min(sum([(stats.norm(loc=(rands[i]), scale=(sds[i])).ppf(min_prb[i]/2)) * min_prb[i] for i in range(len(rands))]), min(rands))
            return val if not np.isnan(val) else min(rands) 

        except Exception as e:
            return min(rands)

    # Good for non-zero values.  Long tailed
    def get_avg_min_nonzero(rands, std_ratio=0.25, sigma=0.5):
        try:
            sds = [np.abs(r * std_ratio) for r in rands]

            min_prb = [0.0]*len(rands)
            min_value = min(rands)
            min_indices = [index for index, value in enumerate(rands) if value == min_value]

            i = min_indices[0]
            for j in range(len(rands)):
                if j in min_indices:
                    continue
                min_prb[j] = stats.norm.cdf(0, loc=(rands[j] - rands[i]), scale=(sds[i]+sds[j])) 
            minmin = (1 - sum(min_prb))/len(min_indices)
            for m in min_indices:
                min_prb[m] = minmin
                
            val = min(sum([(stats.lognorm(sigma, scale=rands[i]).ppf(min_prb[i]/2)) * min_prb[i] for i in range(len(rands))]), min(rands))
            
            return val if not np.isnan(val) else min(rands) 
        except Exception as e:
            return min(rands)

    def calc_avg_result_weight(inp_exp, weights, use_lognormal=False, loop_limit=2000, opt_steps={}):

        exp = inp_exp

        for k, v in weights.items():
            multi_factor = opt_steps[k] if k in opt_steps else 1.0
            exp = exp.replace(k, str(v*multi_factor))

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
                out_len = logical_weight.get_avg_max_nonzero(val_list) if use_lognormal else logical_weight.get_avg_max(val_list)
                exp = re.sub(pattern, str(out_len), exp, count=1)
            else:
                pattern = pattern_or
                match = re.search(pattern, exp)
                if match:
                    val_list = [float(s) for s in match.group().replace("(", "").replace(")", "").replace(" ", "").replace("&", ",").replace("|", ",").split(",")]
                    out_len = logical_weight.get_avg_min_nonzero(val_list) if use_lognormal else logical_weight.get_avg_min(val_list)
                    exp = re.sub(pattern, str(out_len), exp, count=1)
                else:
                    break

        return float(exp) 