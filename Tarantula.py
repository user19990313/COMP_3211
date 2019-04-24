import numpy as np

def division(a,b):
    if a==0 or b==0:
        return 0.0
    return a/b

test_cov = np.loadtxt("qsort_qsort.py.csv",dtype=np.int32, delimiter=",")
test_result = np.loadtxt("qsort_result.txt",dtype=np.int32, delimiter=",")

test_cov_binary=np.int32(test_cov>0)

statement_count=len(test_cov[0])
testcase_count=len(test_result)
susp=[0.0]*statement_count
total_pass=0
total_fail=0
pass_count = [0]*statement_count

for i in range(0,testcase_count):
    if test_result[i]==0:
        total_pass+=1
        pass_count=np.add(test_cov_binary[i],pass_count)
    else:
        total_fail+=1

for i in range(0,statement_count):
    if pass_count[i]>0:
        susp[i]=division(division(pass_count[i],total_pass) , (division(pass_count[i],total_pass)+division(pass_count[i],total_fail)))
        susp[i]=1-susp[i]

susp_sorted=[[idx,susp_value] for idx,susp_value in enumerate(susp,1)]

susp_sorted=sorted(susp_sorted,key=lambda x:x[1],reverse=True)
for idx,susp_value in susp_sorted:
    print("line {0}, suspiciousness is {1:.3f}".format(idx,susp_value))

