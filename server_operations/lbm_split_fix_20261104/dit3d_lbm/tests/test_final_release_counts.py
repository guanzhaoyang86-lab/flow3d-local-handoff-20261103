"""Regression against the real completed raw release; never opens NPZ fields."""
import copy
import pytest
from test_lbm_binding import m


def real_release():
    manifest=m.read(m.RAW/'manifest.json')
    complete=m.read(m.RAW/'COMPLETE.json')
    assert complete['status']=='completed'
    assert complete['num_cases']==complete['num_independent_flow_groups']==3125
    assert complete['physical_trajectories']==complete['single_view_trajectories']==500000
    assert complete['dataset_version']==m.sha(m.RAW/'SHA256SUMS')
    checksums={line.split('  ',1)[1]:line.split('  ',1)[0]
               for line in (m.RAW/'SHA256SUMS').read_text().splitlines()}
    assert checksums['manifest.json']==m.sha(m.RAW/'manifest.json')
    return manifest,complete


def test_finalized_release_counts_and_independent_groups():
    manifest,complete=real_release()
    assert m.COUNTS=={'train':2499,'validation':313,'test':313}
    assert m.validate_split_counts(manifest,complete,complete)==m.COUNTS
    groups=[{row['flow_group_id'] for row in manifest['splits'][name]} for name in m.COUNTS]
    assert sum(map(len,groups))==3125
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])


def test_old_rounded_counts_still_refused_even_if_markers_agree():
    manifest,complete=real_release()
    wrong=copy.deepcopy(manifest)
    wrong['splits']['train'].append(wrong['splits']['validation'].pop())
    marker={**complete,'split_counts':{'train':2500,'validation':312,'test':313}}
    with pytest.raises(ValueError,match='LBM split counts differ'):
        m.validate_split_counts(wrong,marker,marker)


def test_marker_count_disagreement_still_refused():
    manifest,complete=real_release()
    wrong={**complete,'split_counts':{'train':2500,'validation':312,'test':313}}
    with pytest.raises(ValueError,match='LBM split counts differ'):
        m.validate_split_counts(manifest,wrong,complete)
