# logical_weight
#
# Version: 1.0.5
# Last Update: 2024/02/03
# Author: Tomio Kobayashi
#
# Average of max and min of multiple random variables.  In case of error, securely. max and min are returned
#

import scipy.stats as stats
import numpy as np
import re

class logical_weight:
    def get_avg_max(rands, normal_sigma=1):
        try:
            rands_prb = np.array(rands)/sum(rands)
            val = max(sum([(stats.norm(loc=(rands[i]), scale=(normal_sigma)).ppf(1-rands_prb[i]/2)) * rands_prb[i] for i in range(len(rands))]), max(rands))
            return val if not np.isnan(val) else max(rands) 

        except Exception as e:
            print(e)
            return max(rands)

    # Good for non-zero values.  Long tailed
    def get_avg_max_nonzero(rands, log_normal_sigma=0.5):
        try:

            rands_prb = np.array(rands)/sum(rands)

            val = max(sum([(stats.lognorm(log_normal_sigma, scale=rands[i]).ppf(1-rands_prb[i]/2)) * rands_prb[i] for i in range(len(rands))]), max(rands))
            return val if not np.isnan(val) else max(rands) 

        except Exception as e:
            print(e)
            return max(rands)

    def get_avg_min(rands, normal_sigma=1):
        try:
            rands_prb = np.array(rands)/sum(rands)
                
            val = min(sum([(stats.norm(loc=(rands[i]), scale=(normal_sigma)).ppf(rands_prb[i]/2)) * rands_prb[i] for i in range(len(rands))]), min(rands))
            return val if not np.isnan(val) else min(rands) 

        except Exception as e:
            print(e)
            return min(rands)

    # Good for non-zero values.  Long tailed
    def get_avg_min_nonzero(rands, log_normal_sigma=0.5):
        try:
            rands_prb = np.array(rands)/sum(rands)
            val = min(sum([(stats.lognorm(log_normal_sigma, scale=rands[i]).ppf(rands_prb[i]/2)) * rands_prb[i] for i in range(len(rands))]), min(rands))
            
            return val if not np.isnan(val) else min(rands) 
        except Exception as e:
            print(e)
            return min(rands)

    def calc_avg_result_weight(inp_exp, weights, use_lognormal=False, loop_limit=2000, opt_steps={}, normal_sigma=1, log_normal_sigma=0.5):

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
                out_len = logical_weight.get_avg_max_nonzero(val_list, log_normal_sigma=log_normal_sigma) if use_lognormal else logical_weight.get_avg_max(val_list, normal_sigma=normal_sigma)
                exp = re.sub(pattern, str(out_len), exp, count=1)
            else:
                pattern = pattern_or
                match = re.search(pattern, exp)
                if match:
                    val_list = [float(s) for s in match.group().replace("(", "").replace(")", "").replace(" ", "").replace("&", ",").replace("|", ",").split(",")]
                    out_len = logical_weight.get_avg_min_nonzero(val_list, log_normal_sigma=log_normal_sigma) if use_lognormal else logical_weight.get_avg_min(val_list, normal_sigma=normal_sigma)
                    exp = re.sub(pattern, str(out_len), exp, count=1)
                else:
                    break

        return float(exp) 