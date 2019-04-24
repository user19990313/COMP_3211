###LzA for SE Project

import sys
import os
import trace
import linecache
import csv

libdir = os.path.dirname(linecache.__file__)
if libdir[-1] == '4':
    libdir = libdir[:-2]
ignore_dirs = [sys.prefix, sys.exec_prefix, libdir, libdir + "64"]

init_list = []


def init_env (init_cmd_list):
    for statement in init_cmd_list:
        init_list.append(statement)


def init_scope (scope):
    for statement in init_list:
        exec(statement, scope)
    return scope


def run (index, target, exec, csv_file_prefix=None):
    a = False
    test_result = []
    scope = {}
    init_scope(scope)
    with open(target + "_result.txt", "a") as txt:

        # print("Executing test %d: " % index, end='')
        tracer = trace.Trace(ignoredirs=ignore_dirs, trace=0, count=1)
        statement = "a=" + exec
        tracer.runctx(statement, scope, scope)
        a = scope["a"]
        result = tracer.results()
        result_dict = result_extract(result)
        test_result.append((a, result_dict))
        # print("%s\n" % str(bool(a)))
        # print(type(a))
        if (a):
            txt.write('0\n')
        else:
            txt.write('1\n')
        for filename, result_list in result_dict.items():
            if csv_file_prefix != None:
                export_to_csv(csv_file_prefix + '_' + os.path.basename(filename) + '.csv',
                              result_list)
    return test_result


def result_extract (res):
    per_file = {}
    result_dict = {}
    for filename, lineno in res.counts:
        lines_hit = per_file[filename] = per_file.get(filename, {})
        lines_hit[lineno] = res.counts[(filename, lineno)]
    for filename, count in per_file.items():
        if str(filename) == "<string>":
            continue
        source = linecache.getlines(filename)

        li = assemble_data(source, count)
        result_dict[filename] = li
    return result_dict


def assemble_data (lines, lines_hit):
    result_list = []
    for lineno, line in enumerate(lines, 1):
        if lineno in lines_hit:
            hit = lines_hit[lineno]
        else:
            hit = 0
        result_list.append((hit, line))
    return result_list


def export_to_csv (csvname, result_list):
    row = [x for index, (x, line) in enumerate(result_list)]
    # print(row)
    with open(csvname, 'a', newline='') as csvfile:
        (csv.writer(csvfile)).writerow(row)


def run_loader (test_py_file, test_func_name, target_name, test_time):
    init_env(["import " + test_py_file])
    for i in range(test_time):
        run(i, target_name, test_py_file + "." + test_func_name, target_name)


if __name__ == '__main__':
    run_loader("test_sort", "check()", "buggy_sort", 500)

    # init_env(["import test_sort"]);
    # run(['test_sort.check()',         'test_sort.check()',         'test_sort.check()'], "qsort")
