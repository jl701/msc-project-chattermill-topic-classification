"""CPU-only verification of label-free, completion-window monitoring."""
import json

from watch_serving_input_shift import OnlineWindows, completed_window, primary_reference, waiting_failure_reason


def outcome(index, start, finish, length_bin, success=True):
    return {'request_id': f'private-{index}', 'sample_id': 'private-sample',
            'scheduled_s': start, 'finished_s': finish, 'success': success,
            'latency_ms': (finish - start) * 1000, 'input_chars': 1000,
            'length_bin': length_bin, 'response': {'predictions': []},
            'text': 'PRIVATE REVIEW NEVER EXPORTED'}


def test_completion_window_not_arrival_window_and_failures_stay_in_denominator():
    rows = [outcome(0, 10, 70, 9), outcome(1, 20, 75, 9, success=False)]
    reference = {'length_bin_counts': {'0': 10}}
    assert completed_window(rows, 0, reference)['completed_attempts'] == 0
    later = completed_window(rows, 1, reference)
    assert later['completed_attempts'] == 2
    assert later['completed_failed'] == 1
    assert later['success_fraction_of_completed_attempts'] == 0.5
    assert later['input_length_js_bits'] == 1
    assert later['input_length_shift_signal']
    assert later['low_sample_warning']
    encoded = json.dumps(later)
    assert 'private-' not in encoded and 'PRIVATE REVIEW' not in encoded
    assert 'micro_f1' not in encoded


def test_online_emission_real_timestamp_late_revision_and_backfill_are_distinct():
    reference = {'length_bin_counts': {'0': 10}}
    tracker = OnlineWindows(reference, attachment_elapsed_s=1)
    row = outcome(0, 1, 55, 9)
    assert tracker.update([row], 59, 'actual-utc-before-boundary') == []
    first = tracker.update([row], 60.5, 'actual-utc-after-boundary')
    assert first[0]['emitted_utc'] == 'actual-utc-after-boundary'
    assert first[0]['live_observation']
    assert first[0]['emission_lag_after_window_end_s'] == 0.5
    assert tracker.update([row], 61, 'another-utc') == []
    late = outcome(1, 2, 59, 0)
    revised = tracker.update([row, late], 62, 'actual-late-persistence-utc')
    assert revised[0]['event'] == 'completion_window_revision'
    assert revised[0]['revision'] == 1
    late_tracker = OnlineWindows(reference, attachment_elapsed_s=70)
    backfill = late_tracker.update([row], 71, 'actual-late-attachment-utc')
    assert backfill[0]['attachment_was_after_window_end']
    assert not backfill[0]['live_observation']
    much_later = OnlineWindows(reference, attachment_elapsed_s=3700)
    late_final = much_later.update([row], 3701, 'actual-late-final-utc', final=True, final_run_elapsed_s=60)
    assert len(late_final) == 1
    assert late_final[0]['emission_lag_after_window_end_s'] == 3641


def test_reference_reads_completed_development_outcomes_without_labels(tmp_path, monkeypatch):
    from pathlib import Path

    primary = tmp_path / 'primary'
    primary.mkdir()
    (primary / 'plan.json').write_text(json.dumps({
        'sessions': [{'name': 'hybrid', 'server': {'variant': 'hybrid'}}]
    }), encoding='utf-8')
    for repeat in range(1, 4):
        run = primary / 'runs' / f'hybrid__closed_c1_r{repeat}'
        run.mkdir(parents=True)
        (run / 'summary.json').write_text(json.dumps({'interrupted': False, 'attempts': 1}), encoding='utf-8')
        (run / 'requests.jsonl').write_text(json.dumps(outcome(repeat, 0, 1, 0)) + '\n', encoding='utf-8')
    # Reading any labels would fail: the sidecar must derive reference solely
    # from the predeclared development traffic's completed outcomes.
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        assert 'label' not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', guarded_read)
    reference = primary_reference(tmp_path)
    assert len(reference['sources']) == 3
    assert reference['reference']['length_bin_counts'] == {'0': 3}
    assert 'private-' not in json.dumps(reference)


def test_upstream_failure_stops_wait_without_resilience_directory(tmp_path):
    assert waiting_failure_reason(tmp_path) is None
    primary = tmp_path / 'primary'
    primary.mkdir()
    primary_state = primary / 'state.json'
    primary_state.write_text(json.dumps({'status': 'failed'}), encoding='utf-8')
    assert waiting_failure_reason(tmp_path) == 'primary_failed_before_input_shift'
    primary_state.write_text(json.dumps({'status': 'completed'}), encoding='utf-8')
    followup = tmp_path / 'followup_controller'
    followup.mkdir()
    (followup / 'state.json').write_text(json.dumps({'status': 'failed'}), encoding='utf-8')
    assert waiting_failure_reason(tmp_path) == 'followup_controller_failed_before_input_shift'
