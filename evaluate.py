import BPNN
import Tarantula
import pytcov


def _run (mod, cov_filename, result_filename):
    return mod.run(cov_filename, result_filename)


def _get_score (result, total_line_num, buggy_lines):
    # EXAM score - Smaller is better
    # percentage of lines that NEED to examine
    for idx, x in enumerate(result, start=1):
        if x[0] in buggy_lines:
            print(result[:idx])
            return idx / total_line_num


def _get_total_lines (filename):
    non_blank_count = 0
    with open(filename) as infp:
        for line in infp:
            if line.strip():
                non_blank_count += 1
    return non_blank_count


def run (prog_filename, cov_filename, result_filename, buggy_lines):
    result = {}
    result["BPNN"] = _run(BPNN, cov_filename, result_filename)
    result["Tarantula"] = _run(Tarantula, cov_filename, result_filename)
    result["BPNN"] = _get_score(result["BPNN"], _get_total_lines(prog_filename), buggy_lines)
    result["Tarantula"] = _get_score(result["Tarantula"], _get_total_lines(prog_filename), buggy_lines)
    return result


if __name__ == '__main__':
    overall_plots = []
    pytcov.run_loader("test_sort", "check1()", "buggy1_sort", 500)
    result = run("buggy/buggy1.py", "buggy1_sort_buggy1.py.csv", "buggy1_sort_result.txt", [7])
    overall_plots.append(result)

    pytcov.run_loader("test_sort", "check2()", "buggy2_sort", 500)
    result = run("buggy2.py", "buggy2_sort_buggy2.py.csv", "buggy2_sort_result.txt", [7])
    overall_plots.append(result)

    pytcov.run_loader("test_sort", "check3()", "buggy3_sort", 500)
    result = run("buggy3.py", "buggy3_sort_buggy3.py.csv", "buggy3_sort_result.txt", [7])
    overall_plots.append(result)

    pytcov.run_loader("test_sort", "check4()", "buggy4_sort", 500)
    result = run("buggy4.py", "buggy4_sort_buggy4.py.csv", "buggy4_sort_result.txt", [7])
    overall_plots.append(result)

    pytcov.run_loader("test_sort", "check5()", "buggy5_sort", 500)
    result = run("buggy5.py", "buggy5_sort_buggy5.py.csv", "buggy5_sort_result.txt", [7])
    overall_plots.append(result)

    pytcov.run_loader("test_sort", "check6()", "buggy6_sort", 500)
    result = run("buggy6.py", "buggy6_sort_buggy6.py.csv", "buggy6_sort_result.txt", [7])
    overall_plots.append(result)

    print(overall_plots)
