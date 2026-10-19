import re
from typing import Any

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    TEST_EXIT_CODE,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.infra_failure import TIER_ENVIRONMENT, classify_logs
from swebench.types import TestSpec
from swebench.harness.log_parsers import PARSER_REGISTRY


SUITE_RAN = re.compile(
    r"Executed [1-9]\d* of \d+"
    r"|TOTAL: [1-9]\d* (?:SUCCESS|FAILED)"
    r"|[1-9]\d* passing"
    r"|Tests:\s+[1-9]\d*"
    r"|Test Suites:\s+(?:\d+ \w+, )*[1-9]\d* total"
    r"|^# tests [1-9]\d*"
    r"|[1-9]\d* specs?, \d+ failures?"
    r"|': ok$",
    re.M,
)

TEST_EXIT_CODE_RE = re.compile(rf"{re.escape(TEST_EXIT_CODE)}:\s*(-?\d+)")


def parse_test_exit_code(content: str) -> int | None:
    """Return the recorded test command exit status, or None if absent."""
    match = TEST_EXIT_CODE_RE.search(content)
    return int(match.group(1)) if match else None


def _resolve_case(case: str, sm: dict[str, str]) -> str | None:
    if case in sm:
        return case
    if case.count("[") > case.count("]"):
        matches = [k for k in sm if k.startswith(case)]
        passing = {TestStatus.PASSED.value, TestStatus.XFAIL.value}
        if matches and len({sm[k] in passing for k in matches}) == 1:
            return matches[0]
    return None


def test_passed(case: str, sm: dict[str, str]) -> bool:
    key = _resolve_case(case, sm)
    return key is not None and sm[key] in [
        TestStatus.PASSED.value,
        TestStatus.XFAIL.value,
    ]


def test_maintained(case: str, sm: dict[str, str]) -> bool:
    """P2P semantics: a skipped test is not a regression, unlike for F2P."""
    key = _resolve_case(case, sm)
    return test_passed(case, sm) or (
        key is not None and sm[key] == TestStatus.SKIPPED.value
    )


def test_failed(case: str, sm: dict[str, str]) -> bool:
    key = _resolve_case(case, sm)
    return key is None or sm[key] in [
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
        TestStatus.SKIPPED.value,
    ]


def get_logs_eval(test_spec: TestSpec, log_fp: str) -> tuple[dict[str, str], bool]:
    log_parser = PARSER_REGISTRY[test_spec.log_parser]

    with open(log_fp) as f:
        content = f.read()
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            return {}, False
        elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
            return {}, False

        sliced = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        status_map = log_parser(sliced, test_spec)
        if not status_map:
            status_map = log_parser(content, test_spec)
        if not status_map and not SUITE_RAN.search(content):
            return {}, False

        exit_code = parse_test_exit_code(content)
        if (
            exit_code not in (None, 0)
            and status_map
            and not any(
                status in (TestStatus.FAILED.value, TestStatus.ERROR.value)
                for status in status_map.values()
            )
        ):
            return {}, False
        return status_map, True


def get_eval_tests_report(
    eval_status_map: dict[str, str],
    gold_results: dict[str, str],
    calculate_to_fail: bool = False,
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> dict[str, dict[str, list[str]]]:
    def check_pass_and_fail(test_case, eval_status_map, success, failed):
        if test_passed(test_case, eval_status_map):
            success.append(test_case)
        elif test_failed(test_case, eval_status_map):
            failed.append(test_case)

    def check_maintained(test_case, eval_status_map, success, failed):
        if test_maintained(test_case, eval_status_map):
            success.append(test_case)
        elif test_failed(test_case, eval_status_map):
            failed.append(test_case)

    def check_fail_only(test_case, eval_status_map, success, failed):
        if (
            test_case in eval_status_map
            and eval_status_map[test_case] == TestStatus.FAILED.value
        ):
            failed.append(test_case)
        else:
            success.append(test_case)

    check_test_case = (
        check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only
    )

    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        check_test_case(test_case, eval_status_map, f2p_success, f2p_failure)

    check_p2p = (
        check_maintained if eval_type == EvalType.PASS_AND_FAIL else check_fail_only
    )
    p2p_success = []
    p2p_failure = []
    for test_case in gold_results[PASS_TO_PASS]:
        check_p2p(test_case, eval_status_map, p2p_success, p2p_failure)

    results = {
        FAIL_TO_PASS: {
            "success": f2p_success,
            "failure": f2p_failure,
        },
        PASS_TO_PASS: {
            "success": p2p_success,
            "failure": p2p_failure,
        },
    }

    f2f_success = []
    f2f_failure = []
    p2f_success = []
    p2f_failure = []
    if calculate_to_fail:
        for test_case in gold_results[FAIL_TO_FAIL]:
            check_test_case(test_case, eval_status_map, f2f_success, f2f_failure)
        for test_case in gold_results[PASS_TO_FAIL]:
            check_test_case(test_case, eval_status_map, p2f_success, p2f_failure)

    results.update(
        {
            FAIL_TO_FAIL: {
                "success": f2f_success,
                "failure": f2f_failure,
            },
            PASS_TO_FAIL: {
                "success": p2f_success,
                "failure": p2f_failure,
            },
        }
    )
    return results


def compute_fail_to_pass(report: dict[str, dict[str, Any]]) -> float:
    total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]["success"]) / total


def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    total = len(report[PASS_TO_PASS]["success"]) + len(report[PASS_TO_PASS]["failure"])
    if total == 0:
        return 1
    return len(report[PASS_TO_PASS]["success"]) / total


def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)

    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    elif f2p < 1 and f2p > 0 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    else:
        return ResolvedStatus.NO.value


def get_eval_report(
    test_spec: TestSpec,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    report_map = {}

    instance_id = prediction["instance_id"]
    report_map[instance_id] = {
        "patch_is_None": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "resolved": False,
        "infra_failure": False,
    }

    if prediction["model_patch"] is None:
        report_map[instance_id]["patch_is_None"] = True
        return report_map
    report_map[instance_id]["patch_exists"] = True

    eval_status_map, found = get_logs_eval(test_spec, test_log_path)

    if not found:
        classification = classify_logs(test_log_path)
        if classification:
            reason, tier = classification
            report_map[instance_id]["infra_failure"] = tier == TIER_ENVIRONMENT
            report_map[instance_id]["infra_failure_reason"] = reason
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    eval_ref = {
        "instance_id": test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }

    eval_type = EvalType(test_spec.eval_type)

    report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]["resolved"] = True

    if include_tests_status:
        report_map[instance_id]["tests_status"] = report  # type: ignore

    return report_map
