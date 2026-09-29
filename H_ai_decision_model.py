"""Legal, deterministic rule-based baseline; weights can be trained later."""
from Decision_Strategy_engine import StrategyEngine, default_threat_level


class DecisionModel:
    def __init__(self, weights=None):
        self.engine = StrategyEngine(weights=weights)

    def choose_discard(self, state):
        if not state.legal_discards:
            raise ValueError('no legal discard')
        threat = default_threat_level(state.melds, state.player, state.wall_remaining)
        decision = self.engine.choose_discard(
            list(state.hand), list(state.own_melds), list(state.visible_tiles), threat
        )
        if decision.tile not in state.legal_discards:
            raise ValueError('strategy returned an illegal tile')
        return decision

    def should_call(self, state, tile, call_type):
        hand = list(state.hand)
        melds = list(state.own_melds)
        visible = list(state.visible_tiles)
        threat = default_threat_level(state.melds, state.player, state.wall_remaining)
        if call_type == 'pong':
            return hand.count(tile) >= 2 and self.engine.should_pong(hand, melds, tile, visible, threat).action == 'pong'
        if call_type == 'kong':
            return hand.count(tile) >= 3 and self.engine.should_kong(hand, melds, tile, visible, threat).action == 'kong'
        if call_type == 'chi':
            previous = (state.player - 1) % 4
            return bool(state.discards[previous] and state.discards[previous][-1] == tile and
                        self.engine.should_chi(hand, melds, tile, visible, threat).action == 'chi')
        raise ValueError(f'unsupported call: {call_type}')
