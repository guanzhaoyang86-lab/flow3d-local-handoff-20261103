"""Republish coherent checksums before mutations to test structural validation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dataset_gate as gate
spec = importlib.util.spec_from_file_location('repaired_dps_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture
def published(tmp_path):
    raw, adapted = tmp_path / 'raw', tmp_path / 'adapted'
    raw.mkdir(); (adapted / 'fields').mkdir(parents=True)
    splits, checksums = {}, {}
    counter = 0
    for name, count in gate.COUNTS.items():
        splits[name] = []
        for _ in range(count):
            case = f'case_{counter:05d}'; counter += 1
            splits[name].append({'case_id': case, 'flow_group_id': 'flow_' + case, 'path': case + '.npz'})
            checksums[case + '.npz'] = ('%064x' % counter)
            (adapted / 'fields' / (case + '.npz')).touch()
    base = {'mode': 'scientific', 'split_unit': 'flow_group_id', 'allow_overlap_for_smoke_test': False}
    raw_manifest = dict(base, splits=copy.deepcopy(splits))
    adapter_manifest = dict(base, schema='flow3d_lbm_frozen_manifest_v1', dataset_kind='sparse_projected_particle_flow',
        dataset_id='lbm_fixture', source_dataset_root=str(raw), velocity_units='lattice cells per LBM step',
        time_units='LBM steps', unit_scale=[1.,1.,1.],
        observation_protocol={'N':12,'probe_count':32,'observation_seed':47,'view':'xz','training_seed':31},
        splits={name: [dict(row, path='fields/' + row['path'], source_case_id=row['case_id'],
                            sha256=checksums[row['path']]) for row in rows] for name, rows in splits.items()})
    raw_marker = {'status':'completed','num_cases':3125,'num_independent_flow_groups':3125,
        'physical_trajectories':500000,'single_view_trajectories':500000,'projection_names':['xz'],
        'split_counts':dict(gate.COUNTS),'manifest':str(raw/'manifest.json')}
    adapter_marker = {'status':'completed','dataset_id':'lbm_fixture','verified_fields':3125,
        'physical_trajectories':500000,'split_counts':dict(gate.COUNTS),'arrays_modified':False,
        'source_data_mutated':False,'training_or_rank_selection_performed':False,'adapter_sha256':'1'*64}
    config = {'dataset':'lbm','dataset_id':'lbm_fixture','raw_data_root':str(raw),
        'manifest':str(adapted/'manifest.json'),'completion_marker':str(adapted/'COMPLETE.json'),
        'expected_split_counts':dict(gate.COUNTS),'expected_adapter_sha256':'1'*64,
        'deferred_dataset_adoption':True}
    def publish():
        runtime.write(raw/'manifest.json', raw_manifest)
        checksums['manifest.json'] = gate.sha(raw/'manifest.json')
        (raw/'SHA256SUMS').write_text(''.join(digest+'  '+name+'\n' for name,digest in sorted(checksums.items())))
        version = gate.sha(raw/'SHA256SUMS')
        raw_marker['dataset_version'] = version
        runtime.write(raw/'COMPLETE.json', raw_marker)
        config.update(expected_raw_manifest_sha256=gate.sha(raw/'manifest.json'),
            expected_raw_complete_sha256=gate.sha(raw/'COMPLETE.json'),expected_raw_dataset_version=version)
        adapter_manifest['source_dataset_version'] = version
        runtime.write(adapted/'manifest.json', adapter_manifest)
        adapter_marker.update(manifest_sha256=gate.sha(adapted/'manifest.json'), source_dataset_version=version,
                              source_complete_sha256=gate.sha(raw/'COMPLETE.json'))
        runtime.write(adapted/'COMPLETE.json', adapter_marker)
    publish()
    return SimpleNamespace(config=config, raw=raw, adapted=adapted, raw_manifest=raw_manifest,
        adapter_manifest=adapter_manifest, raw_marker=raw_marker, adapter_marker=adapter_marker, publish=publish)


def test_actual_split_and_full_provenance_are_accepted(published):
    result = gate.verify_adapter(published.config)
    assert result['split_counts'] == {'train':2499,'validation':313,'test':313}
    assert result['verified_fields'] == 3125


@pytest.mark.parametrize('stage', ['train','calibrate','test'])
def test_old_split_rejected_before_cuda_in_every_production_entry(published, tmp_path, monkeypatch, stage):
    for manifest in (published.raw_manifest, published.adapter_manifest):
        manifest['splits']['train'].append(manifest['splits']['validation'].pop(0))
    published.raw_marker['split_counts'] = {'train':2500,'validation':312,'test':313}
    published.adapter_marker['split_counts'] = dict(published.raw_marker['split_counts'])
    published.publish()
    runtime.write(tmp_path/'config.json', published.config)
    runtime.write(tmp_path/'frozen_manifest.json', {'files':{},'manifest_sha256':None})
    monkeypatch.setattr(runtime, 'HERE', tmp_path)
    monkeypatch.setenv('FLOW3D_DPS_FROZEN_SHA256', runtime.sha(tmp_path/'frozen_manifest.json'))
    monkeypatch.setattr(sys, 'argv', ['runtime.py',stage])
    monkeypatch.setattr(runtime, 'cuda_check', lambda: pytest.fail('CUDA started before rejecting the split'))
    with pytest.raises(ValueError, match='split counts|2499/313/313'):
        runtime.main()


def test_adapter_cannot_reorder_the_published_split(published):
    rows = published.adapter_manifest['splits']['validation']; rows[0], rows[1] = rows[1], rows[0]
    published.publish()
    with pytest.raises(ValueError, match='case order'):
        gate.verify_adapter(published.config)


def test_cross_split_group_overlap_rejected_even_with_new_valid_hashes(published):
    group = published.raw_manifest['splits']['train'][0]['flow_group_id']
    for manifest in (published.raw_manifest,published.adapter_manifest):
        manifest['splits']['validation'][0]['flow_group_id'] = group
    published.publish()
    with pytest.raises(ValueError, match='overlap'):
        gate.verify_adapter(published.config)


@pytest.mark.parametrize('change,pattern', [('units','lattice units'),('observation','protocol differs'),
    ('case_checksum','case checksum'),('adapter_code','adapter code'),('arrays_changed','preserve arrays'),
    ('incomplete','incomplete'),('missing_field','file is missing')])
def test_adapter_contract_mismatch_rejected(published, change, pattern):
    if change == 'units': published.adapter_manifest['velocity_units'] = 'm/s'
    elif change == 'observation': published.adapter_manifest['observation_protocol']['N'] = 2
    elif change == 'case_checksum': published.adapter_manifest['splits']['test'][0]['sha256'] = '0'*64
    elif change == 'adapter_code': published.adapter_marker['adapter_sha256'] = '2'*64
    elif change == 'arrays_changed': published.adapter_marker['arrays_modified'] = True
    elif change == 'incomplete': published.adapter_marker['status'] = 'running'
    else: next((published.adapted/'fields').iterdir()).unlink()
    published.publish()
    with pytest.raises(ValueError, match=pattern):
        gate.verify_adapter(published.config)


def test_preflight_reports_waiting_without_claiming_adapter_validation(published, tmp_path, monkeypatch):
    (published.adapted/'COMPLETE.json').unlink()
    runtime.write(tmp_path/'config.json', published.config)
    runtime.write(tmp_path/'frozen_manifest.json', {'files':{},'manifest_sha256':None})
    monkeypatch.setattr(runtime, 'HERE', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['runtime.py','preflight'])
    monkeypatch.setattr(runtime, 'cuda_check', lambda: pytest.fail('CPU preflight used CUDA'))
    runtime.main()
    report = runtime.read(tmp_path/'reports/preflight.json')
    assert report['status'] == 'waiting_for_completed_adapter'
    assert report['adapter_data_validation_complete'] is False and report['prior_training'] == 'not_run'
    assert report['raw_validation']['split_counts'] == gate.COUNTS
