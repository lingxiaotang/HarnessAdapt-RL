"""Optional batching alternative for related bounded source observations."""
import importlib.util
import json
from pathlib import Path

TOOL_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'inspect_windows',
        'description': 'Read 2–4 related text windows in one action, such as a function and its caller. Uses read_file restrictions. Total output is bounded to 6000 characters; incomplete windows are marked. Use read_file for one window or for omitted relevant lines. This tool only reads.',
        'parameters': {
            'type': 'object',
            'properties': {
                'windows': {
                    'type': 'array', 'minItems': 2, 'maxItems': 4,
                    'items': {
                        'type': 'object',
                        'properties': {
                            'path': {'type': 'string'},
                            'offset': {'type': 'integer', 'minimum': 1},
                            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 60},
                        },
                        'required': ['path'], 'additionalProperties': False,
                    },
                },
            },
            'required': ['windows'], 'additionalProperties': False,
        },
    },
}
SMOKE_ARGUMENTS = {'windows': [{'path': 'package/module.py', 'limit': 15},
                              {'path': 'package/caller.py', 'limit': 15}]}


def _reader():
    # Always use this candidate snapshot's reader, never a mutable external
    # release. All source-file access still occurs inside the task environment.
    path = Path(__file__).with_name('read_file.py')
    spec = importlib.util.spec_from_file_location('bundle_snapshot_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(arguments, environment):
    def error(message):
        return {'output': 'inspect_windows: ' + message,
                'returncode': 2, 'exception_info': ''}
    if not isinstance(arguments, dict) or set(arguments) != {'windows'}:
        return error('Provide only windows, an array of 2–4 read requests.')
    windows = arguments['windows']
    if not isinstance(windows, list) or not 2 <= len(windows) <= 4:
        return error('Provide 2–4 related windows; use read_file for one window.')
    normalized = []
    for window in windows:
        if not isinstance(window, dict) or 'path' not in window or set(window) - {'path', 'offset', 'limit'}:
            return error('Each window needs path and optionally offset and limit.')
        path, offset, limit = window['path'], window.get('offset', 1), window.get('limit', 15)
        if not isinstance(path, str) or not path or len(path) > 1024:
            return error('Use a nonempty path of at most 1024 characters.')
        if type(offset) is not int or not 1 <= offset <= 10000000 or type(limit) is not int or not 1 <= limit <= 60:
            return error('offset must be positive and limit must be 1–60.')
        normalized.append(dict(path=path, offset=offset, limit=limit))
    reader = _reader()
    # Fixed per-window shares prevent one long observation from hiding all
    # subsequent windows. The existing reader bounds individual operations.
    share = 6000 // len(windows)
    output = []
    failed = False
    for index, window in enumerate(normalized, 1):
        result = reader.execute(window, environment)
        failed |= result['returncode'] != 0
        heading = f'WINDOW {index}\n'
        text = str(result.get('output') or '')
        budget = share - len(heading) - 1
        if len(text) > budget:
            notice = '\n[Bundle output truncated; this window is incomplete. Use read_file for omitted relevant lines.]'
            prefix = text[:budget - len(notice)]
            newline = prefix.rfind('\n')
            if newline > 0:
                prefix = prefix[:newline]
            text = prefix + notice
        output.append(heading + text)
    return {'output': '\n'.join(output), 'returncode': 2 if failed else 0,
            'exception_info': ''}
