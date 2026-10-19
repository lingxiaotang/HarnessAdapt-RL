import re
from swebench.harness.constants import TestStatus
from swebench.types import TestSpec


def parse_log_maven(log: str, test_spec: TestSpec) -> dict[str, str]:
    test_status_map = {}
    pending_tests: list[str] = []
    unmatched_results: list[str] = []

    test_name_pattern = r"^.*-Dtest=(\S+).*$"
    result_pattern = r"^.*BUILD (SUCCESS|FAILURE)$"

    for line in log.split("\n"):
        test_name_match = re.match(test_name_pattern, line.strip())
        if test_name_match:
            pending_tests.append(test_name_match.groups()[0])

        result_match = re.match(result_pattern, line.strip())
        if result_match:
            status = result_match.groups()[0]
            if pending_tests:
                test_name = pending_tests.pop(0)
                if status == "SUCCESS":
                    test_status_map[test_name] = TestStatus.PASSED.value
                elif status == "FAILURE":
                    test_status_map[test_name] = TestStatus.FAILED.value
            else:
                unmatched_results.append(status)

    while pending_tests and unmatched_results:
        test_name = pending_tests.pop(0)
        status = unmatched_results.pop(0)
        if status == "SUCCESS":
            test_status_map[test_name] = TestStatus.PASSED.value
        elif status == "FAILURE":
            test_status_map[test_name] = TestStatus.FAILED.value

    if pending_tests:
        print(
            f"[WARNING] Maven log parser: {len(pending_tests)} test(s) had no BUILD result: "
            f"{pending_tests}"
        )

    return test_status_map


def parse_log_ant(log: str, test_spec: TestSpec) -> dict[str, str]:
    test_status_map = {}

    pattern = r"^\s*\[junit\]\s+\[(PASS|FAIL|ERR)\]\s+(.*)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == "PASS":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status in ["FAIL", "ERR"]:
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_gradle_custom(log: str, test_spec: TestSpec) -> dict[str, str]:
    test_status_map = {}

    full_pattern = r"^([^>].+)\s+(PASSED|FAILED)$"
    test_name_pattern = r"^([^>]\S*\s+>\s+\S+)$"
    status_only_pattern = r"^(PASSED|FAILED)$"

    pending_test_name = None

    for line in log.split("\n"):
        stripped = line.strip()

        match = re.match(full_pattern, stripped)
        if match:
            test_name, status = match.groups()
            if status == "PASSED":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "FAILED":
                test_status_map[test_name] = TestStatus.FAILED.value
            pending_test_name = None
            continue

        test_name_match = re.match(test_name_pattern, stripped)
        if test_name_match:
            pending_test_name = test_name_match.group(1)
            continue

        if pending_test_name:
            status_match = re.match(status_only_pattern, stripped)
            if status_match:
                status = status_match.group(1)
                if status == "PASSED":
                    test_status_map[pending_test_name] = TestStatus.PASSED.value
                elif status == "FAILED":
                    test_status_map[pending_test_name] = TestStatus.FAILED.value
                pending_test_name = None

    if pending_test_name:
        print(
            f"[WARNING] Gradle log parser: test had no status result: {pending_test_name}"
        )

    return test_status_map


MAP_REPO_TO_PARSER_JAVA = {
    "google/gson": parse_log_maven,
    "apache/druid": parse_log_maven,
    "javaparser/javaparser": parse_log_maven,
    "projectlombok/lombok": parse_log_ant,
    "apache/lucene": parse_log_gradle_custom,
    "reactivex/rxjava": parse_log_gradle_custom,
}
