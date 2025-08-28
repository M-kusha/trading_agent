import asyncio
import time

from modules.utils.info_bus import InfoBusManager


def test_duplicate_writer_event_logged():
    bus = InfoBusManager.get_instance()

    # Ensure clean state
    bus.shutdown()
    bus = InfoBusManager.get_instance()

    # Subscribe to events to catch duplicate_writer
    captured = []
    def on_event(evt):
        if evt.get('type') == 'duplicate_writer' and evt.get('base_key') == 'market_regime':
            captured.append(evt)

    bus.subscribe('event_logged', on_event)

    # First provider writes market_regime
    bus.set('market_regime', 'trending', module='FirstProvider', thesis='first')

    # Second provider attempts to write same key
    bus.set('market_regime', 'volatile', module='SecondProvider', thesis='second')

    # Give a brief moment for event propagation
    time.sleep(0.01)

    assert any(evt for evt in captured if evt.get('attempting_module') == 'SecondProvider'), "Expected duplicate_writer event for market_regime"


def test_committee_publishes_trade_vote_smoke():
    # This is a smoke test: ensure the bus accepts trade_vote writes from coordinator
    bus = InfoBusManager.get_instance()

    # Safe re-init
    bus.shutdown()
    bus = InfoBusManager.get_instance()

    # Simulate committee writing canonical trade_vote
    bus.set('trade_vote', {
        'action': 'buy',
        'confidence': 0.8,
        'source': 'EnhancedVotingCommitteeCoordinator',
        'timestamp': 'now'
    }, module='EnhancedVotingCommitteeCoordinator', thesis='canonical trade vote', confidence=0.8)

    tv = bus.get('trade_vote', 'Test')
    assert isinstance(tv, dict) and tv.get('action') in {'buy', 'sell', 'abstain'}


