import BPNN
import Tarantula
import pytcov


def _run(mod, cov_filename, result_filename):
    return mod.run(cov_filename, result_filename)


def _get_score(result, total_line_num, buggy_lines):
    # EXAM score - Higher is better
    # percentage of lines that DO NOT NEED to examine
    for idx, x in enumerate(result, start=1):
        if x[0] in buggy_lines:
            print(result[:idx])
            return 1 - idx / total_line_num


def _get_total_lines(filename):
    non_blank_count = 0
    with open(filename) as infp:
        for line in infp:
            if line.strip():
                non_blank_count += 1
    return non_blank_count


def run(prog_filename, cov_filename, result_filename, buggy_lines):
    result = {}
    result["BPNN"] = _run(BPNN, cov_filename, result_filename)
    result["Tarantula"] = _run(Tarantula, cov_filename, result_filename)
    result["BPNN"] = _get_score(result["BPNN"], _get_total_lines(prog_filename), buggy_lines)
    result["Tarantula"] = _get_score(result["Tarantula"], _get_total_lines(prog_filename), buggy_lines)
    print(result)


if __name__ == '__main__':
    pytcov.run_loader("test_sort", "check()", "buggy_sort", 500)
    run("buggy.py", "buggy_sort_buggy.py.csv", "buggy_sort_result.txt", [7])
