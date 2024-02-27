# logical_weight
#
# Version: 1.1.3
# Last Update: 2024/02/27
# Author: Tomio Kobayashi
#
# Average of max and min of multiple random variables.  In case of error, securely. max and min are returned

import scipy.stats as stats
import numpy as np
import re

class logical_weight:
    def lognormal(mu, sigma):
        return np.log(np.random.lognormal(mu, min(mu, sigma))+1)
    
    def get_avg_max(rands, sigma=1.):
        nloop = 3000
        return np.mean([max([np.random.normal(r, sigma) for r in rands]) for i in range(nloop)])
#         return max([np.mean([np.random.normal(r, sigma) for r in rands]) for i in range(nloop)])

    # Good for non-zero values.  Long tailed
    def get_avg_max_nonzero(rands, sigma=1.):
        nloop = 3000
#         ret = np.mean([max([np.log(np.random.lognormal(r, sigma)+1) for r in rands]) for i in range(nloop)])
        if max(rands) > 300000:
            rands2 = [r/300000 for r in rands]
            ret = np.mean([max([np.log(np.random.lognormal(r, sigma)+1) for r in rands2]) for i in range(nloop)])
            ret *= 300000
        elif max(rands) > 500:
            rands2 = [r/500 for r in rands]
            ret = np.mean([max([np.log(np.random.lognormal(r, sigma)+1) for r in rands2]) for i in range(nloop)])
            ret *= 500
        else:
            ret = np.mean([max([np.log(np.random.lognormal(r, sigma)+1) for r in rands]) for i in range(nloop)])
        if ret == float('inf'):
            return max(rands)
        else:
            return ret

    def get_avg_min(rands, sigma=1):
        nloop = 3000
        return np.mean([min([np.random.normal(r, sigma) for r in rands]) for i in range(nloop)])

    # Good for non-zero values.  Long tailed
    def get_avg_min_nonzero(rands, sigma=1.):
        nloop = 3000
#         ret = np.mean([min([np.log(np.random.lognormal(r, sigma)+1) for r in rands]) for i in range(nloop)])
        if min(rands) > 300000:
            rands2 = [r/300000 for r in rands]
            ret = np.mean([min([np.log(np.random.lognormal(r, sigma)+1) for r in rands2]) for i in range(nloop)])
            ret *= 300000
        elif min(rands) > 500:
            rands2 = [r/500 for r in rands]
            ret = np.mean([min([np.log(np.random.lognormal(r, sigma)+1) for r in rands2]) for i in range(nloop)])
            ret *= 500
        else:
            ret = np.mean([min([np.log(np.random.lognormal(r, sigma)+1) for r in rands]) for i in range(nloop)])
        if ret == float('inf'):
            return min(rands)
        else:
            return ret


#     def calc_avg_result_weight(inp_exp, weights, use_lognormal=False, loop_limit=2000, opt_steps={}, normal_sigma=1, log_normal_sigma=0.5):
#     def calc_avg_result_weight(inp_exp, weights, use_lognormal=False, loop_limit=2000, opt_steps={}, sigma=1.):
    def calc_avg_result_weight(inp_exp, weights, use_lognormal=False, loop_limit=2000, sigma=1.):

        exp = inp_exp

        for k, v in weights.items():
#             multi_factor = opt_steps[k] if k in opt_steps else 1.0
#             exp = exp.replace(k, str(v*multi_factor))
            exp = exp.replace(k, str(v*1.0))
#         print("exp", exp)
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