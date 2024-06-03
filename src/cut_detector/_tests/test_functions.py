from cut_detector.utils.mid_body_spot import MidBodySpot
from cut_detector.utils.mid_body_track import MidBodyTrack


def test_mid_body_track_fill_gaps_standard():
    """Test the fill_gaps method of the MidBodyTrack class."""
    track = MidBodyTrack(0)
    track.add_spot(MidBodySpot(0, 0, 0, 0.0, 0.0, 0.0, 0.0))
    track.add_spot(MidBodySpot(2, 2, 2, 2.0, 2.0, 2.0, 2.0))
    track.add_spot(MidBodySpot(3, 3, 3, 3.0, 3.0, 3.0, 3.0))
    track.fill_gaps()
    assert track.spots[1].x == 1


def test_mid_body_track_fill_gaps_none():
    """Same with missing attributes."""
    track = MidBodyTrack(0)
    track.add_spot(MidBodySpot(0, 0, 0, 0.0))
    track.add_spot(MidBodySpot(2, 2, 2, 2.0))
    track.fill_gaps()
    assert track.spots[1].x == 1
    assert track.spots[1].area is None


def test_mid_body_track_fill_gaps_bigger():
    """Same with bigger gap."""
    track = MidBodyTrack(0)
    track.add_spot(MidBodySpot(2, 2, 2, 2.0))
    track.add_spot(MidBodySpot(5, 5, 5, 5.0))
    track.fill_gaps()
    assert track.spots[3].x == 3
    assert track.spots[4].x == 4


def test_mid_body_track_fill_gaps_all():
    """Same with all previous conditions."""
    track = MidBodyTrack(0)
    track.add_spot(MidBodySpot(0, 0, 0, 0.0))
    track.add_spot(MidBodySpot(2, 2, 2, 2.0))
    track.add_spot(MidBodySpot(5, 5, 5, 5.0))
    track.fill_gaps()
    assert track.spots[1].x == 1
    assert track.spots[3].x == 3
    assert track.spots[4].x == 4
