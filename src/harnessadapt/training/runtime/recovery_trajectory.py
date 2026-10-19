"""Preserve sampled segments across recovery without re-tokenizing training ids.

Only enabled for new experiments through R2E_PRESERVE_RECOVERY_SEGMENTS=1.
Routing, malformed-response demotion and shared-prefix de-duplication remain
the stock manager's responsibility. Ordinary no-recovery REALIGN is unchanged.
"""
from slime.agent.trajectory import TrajectoryManager, DriftKind, _SampleBuilder


class RecoveryTrajectoryManager(TrajectoryManager):
    def _split_chain_into_builders(self, chain):
        builders = []
        previous_index = -1
        for index, node in enumerate(chain):
            if node.role != 'assistant' or node.turn is None:
                continue
            trained = not node.response_trained
            node.response_trained = True
            crossed_recovery = any(
                (item.metadata.get('demoted_response') or {}).get('reason') == 'format_error_recovery'
                or (item.metadata.get('merged_rewrite') or {}).get('reason') == 'format_error_recovery'
                for item in chain[previous_index + 1:index])
            kind = builders[-1].classify_token_drift(node.turn) if builders else DriftKind.FORK
            # Preserve the exact previous response even for short re-render
            # drift at a recovery boundary. Never splice new ids onto old logps.
            if kind is DriftKind.FORK or (crossed_recovery and kind is DriftKind.REALIGN):
                builders.append(_SampleBuilder(self._fork_threshold))
                kind = DriftKind.CLEAN
            builders[-1].append_turn(node.turn, kind, trained=trained)
            previous_index = index
        return builders
