"""Regression tests for bggen-fix4 (batch2 7c801ee, kubectl traces):

A. t08 d1 / d2: release VDM '?' -> R9 2D geometry 0.7 -> next keyframe VDM 0 with no geometry at that event -> the
   pair's remembered positive geometry (valid: nothing grasped / moved since) makes it an R10 conflict, re-checked
B. t08 d3: the release frame puts the cheese 13.3 cm above the basket (still falling) -> the geometry is taken at the
   first settled frame (inside); with no settled frame the release frame is used with height vetoing nothing
C. B1K 040: a full category's detection 4 m from every track is dropped (capped_dropped), 0.8 m is assigned; LIBERO
   (no caps) unchanged
D. aggregate: an episode with no re-observation has no rate; decided (graded statements) beside rel. accuracy
"""

from __future__ import annotations

import json

import numpy as np

from capx.bggen import aggregate as agg
from capx.bggen.configs import b1k as b1k_cfg
from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.events import Event
from capx.bggen.generator import Generator
from capx.bggen.geometry import GeoMemory
from capx.bggen.perception import Detection
from capx.bggen.records import ArmState, CameraFrame, FrameRecord
from capx.bggen.tracks import Tracker
from capx.bggen.vocab import category

CHEESE, BASKET = "cream_cheese_A", "basket_A"
K = ("in", CHEESE, BASKET)
CAM = "agentview"


def _det(cat, x, y, z=0.04, size=(0.12, 0.12, 0.05), score=0.9, lo=None, hi=None, px=None, box=(0, 0, 1, 1)):
    d = Detection(cat, score, px or (100 - 200 * y, 100 * x), box,
                  None if z is None else np.array([x, y, z], float), np.array(size, float))
    d.lo, d.hi = (None if lo is None else np.array(lo, float)), (None if hi is None else np.array(hi, float))
    return d


def _basket(top=0.08):
    """A shallow basket: median point z 0.05, mask depth points 0 .. ``top``."""
    return _det("basket", 0.60, 0.0, 0.05, (0.25, 0.18, 0.08), lo=(0.475, -0.09, 0.0), hi=(0.725, 0.09, top))


class _Ep:
    """Minimal episode: one arm, one camera with an image at every frame, scripted EEF / gripper width."""

    fps, mobile, cameras, arms = 20.0, False, [CAM], ["main"]
    instruction = "put the cream cheese in the basket"

    def __init__(self, n, eef, width):
        self.n_frames, self._eef, self._w = n, eef, width
        self.image_reads = 0

    def robot(self, t):
        return FrameRecord(t, t / self.fps, {"main": ArmState(float(self._w(t)), np.asarray(self._eef(t), float))})

    def frame(self, t, cams=(), causal=True):
        fr = self.robot(t)
        fr.cams = {c: CameraFrame(c, t, np.zeros((120, 160, 3), np.uint8)) for c in cams if c == CAM}
        return fr


class _Det:
    """Scripted detector: ``at(t)`` -> detections."""

    name = "scripted"

    def __init__(self, at):
        self.at, self.calls = at, 0

    def detect(self, cam, cats):
        self.calls += 1
        keys = {c.key for c in cats}
        return [d for d in self.at(cam.t) if d.category in keys]


def _gen(ep, det, mode="A", init_dets=None):
    gen = Generator(ep, det, models=None, cfg=lm_cfg(mode))
    gen.categories = [c for c in (category("cream_cheese"), category("basket")) if c is not None]
    assert len(gen.categories) == 2
    gen.tracker.initialize(0, CAM, init_dets if init_dets is not None else det.at(0))
    gen.geo.observe_sizes()
    return gen


CATS = {CHEESE: "cream_cheese", BASKET: "basket"}


# --- A: t08 d1 / d2 ----------------------------------------------------------------------------------------------

def _d1_world():
    """2D path (no depth): the cheese and basket are tracked by pixels only, the camera is fixed; the front camera sees
    the basket's wall, the cheese's mask lies above the basket's box (no 2D evidence at the keyframe)."""
    ep = _Ep(400, lambda t: [0.6, 0.0, 0.2], lambda t: 0.08)
    det = _Det(lambda t: [_det("basket", 0.6, 0.0, None, px=(80.0, 60.0), box=(60, 50, 100, 70)),
                          _det("cream_cheese", 0.6, 0.0, None, px=(82.0, 38.0), box=(75, 30, 90, 45))])
    return _gen(ep, det)


def _release_r9(gen, t=198):
    """The release: VDM '?' for in(cheese, basket), the 2D release geometry positive (0.051 < r 0.074) -> R9 0.7."""
    g = {"mode": "2d", "release_d_xy": 0.051, "r": 0.074, "positive": True, "frame": "release"}
    gen.verifier.begin()
    assert gen.verifier.geo_fill(K, g, t, CATS)
    gen._remember_geometry({K: g}, t, "release")
    assert gen.graph.p(K) == 0.7 and gen.graph.get(K).source == "exp"


def test_t08_d1_keyframe_zero_after_r9_is_a_conflict_from_the_remembered_geometry():
    gen = _d1_world()
    _release_r9(gen)
    # t=229 keyframe: VDM 0, this event computes no positive geometry (2D, nothing released)
    vals = {K: "0"}
    gen.verifier.begin()
    geom = gen._geometry_for(vals, CAM)
    assert not gen.verifier.geo_positive(K, geom.get(K), CATS)
    mem = gen._geometry_memory(vals, geom, CAM)
    rec = mem["(in cream_cheese_A basket_A)"]
    assert rec["valid"] and rec["from_t"] == 198 and rec["src"] == "release" and rec["grasps_since"] == {}
    assert set(rec["movement"]) == {CHEESE, BASKET}  # the validity evidence is recorded per object
    assert geom[K]["remembered"]["valid"]
    gen._observe_apply(vals, 229, {}, geom, set())
    assert gen.graph.p(K) == 0.5 and gen.graph.get(K).source == "conflict" and gen.verifier.new_conflicts == [K]
    hold = [d for d in gen.verifier.decisions if d.rule == "R10-hold"]
    assert hold and "remembered" in hold[0].why
    # the re-check (wrist camera / hindsight) cannot tell: pending, asked again; the next 0 lowers it
    assert not gen.verifier.recheck(K, "?", 229, "look robot0_eye_in_hand")
    assert K in gen.verifier.recheck_keys()
    gen.verifier.begin()
    geom = gen._geometry_for({K: "0"}, CAM)
    assert gen._geometry_memory({K: "0"}, geom, CAM) == {}  # pending conflict: the verifier's R10 path, not the memory
    gen._observe_apply({K: "0"}, 258, {}, geom, set())
    assert gen.graph.p(K) == 0.1 and [d.rule for d in gen.verifier.decisions] == ["R10-confirm"]
    # the remembered geometry raised its one conflict: a later 0 is an ordinary observation
    gen.verifier.begin()
    geom = gen._geometry_for({K: "0"}, CAM)
    assert gen._geometry_memory({K: "0"}, geom, CAM)["(in cream_cheese_A basket_A)"]["valid"] is False
    gen._observe_apply({K: "0"}, 300, {}, geom, set())
    assert gen.graph.p(K) == 0.1 and [d.rule for d in gen.verifier.decisions] == ["R1"]


def test_t08_d2_recheck_that_sees_it_inside_raises_it():
    gen = _d1_world()
    _release_r9(gen, 197)
    vals = {K: "0"}
    gen.verifier.begin()
    geom = gen._geometry_for(vals, CAM)
    gen._geometry_memory(vals, geom, CAM)
    gen._observe_apply(vals, 228, {}, geom, set())
    assert gen.verifier.recheck(K, "1", 228, "hindsight") and gen.graph.p(K) == 0.9 and gen.graph.get(K).source == "obs"


def test_remembered_geometry_is_void_after_a_grasp_or_a_move():
    # grasped since the release -> the Observe 0 is an observation (R1 0.1), as in fix3
    gen = _d1_world()
    _release_r9(gen)
    gen.grasped_at[CHEESE] = [210]
    vals = {K: "0"}
    gen.verifier.begin()
    geom = gen._geometry_for(vals, CAM)
    rec = gen._geometry_memory(vals, geom, CAM)["(in cream_cheese_A basket_A)"]
    assert not rec["valid"] and rec["grasps_since"] == {CHEESE: [210]} and "remembered" not in geom[K]
    gen._observe_apply(vals, 229, {}, geom, set())
    assert gen.graph.p(K) == 0.1 and [d.rule for d in gen.verifier.decisions] == ["R1"]
    assert K not in gen.geo_mem.entries  # dropped once void
    # moved since (fixed camera, 2D): the cheese's pixel moved 40 px
    gen = _d1_world()
    _release_r9(gen)
    gen.tracker.tracks[CHEESE].pixel[CAM] = (122.0, 38.0)
    geom = gen._geometry_for(vals, CAM)
    rec = gen._geometry_memory(vals, geom, CAM)["(in cream_cheese_A basket_A)"]
    assert not rec["valid"] and rec["moved"] == {CHEESE: 40.0}


def test_memory_reference_is_the_first_detection_after_an_undetected_release():
    """The released cheese is not detected at the release (its track has the hand's position): its fall into the basket
    is no movement; the first detection after it becomes the reference, later motion voids the memory."""
    tr = Tracker(lm_cfg().tracks)
    tr.initialize(0, CAM, [_basket(), _det("cream_cheese", 0.6, 0.0, 0.2, (0.06, 0.04, 0.03))])
    mem = GeoMemory()
    tr.tracks[CHEESE].seen_now = False  # not detected at the release frame
    mem.remember(K, {"mode": "2d", "positive": True}, 198, tr.tracks, "release", True)
    assert mem.entries[K]["snap"][CHEESE]["ref_t"] is None
    tr.update(229, CAM, [_basket(), _det("cream_cheese", 0.6, 0.01, 0.06, (0.06, 0.04, 0.03))])  # landed 14 cm lower
    mem.refresh(tr.tracks, 229)
    g, rec = mem.lookup(K, tr.tracks, {}, CAM)
    assert rec["valid"] and g["remembered"] is rec
    tr.update(260, CAM, [_basket(), _det("cream_cheese", 0.6, 0.08, 0.06, (0.06, 0.04, 0.03))])  # pushed 7 cm
    g, rec = mem.lookup(K, tr.tracks, {}, CAM)
    assert g is None and rec["moved"] == {CHEESE: 0.07}


def test_depth_geometry_now_that_says_outside_keeps_the_zero():
    tr = Tracker(lm_cfg().tracks)
    tr.initialize(0, CAM, [_basket(), _det("cream_cheese", 0.6, 0.0, 0.06, (0.06, 0.04, 0.03))])
    ep = _Ep(400, lambda t: [0.6, 0.0, 0.2], lambda t: 0.08)
    det = _Det(lambda t: [_basket(), _det("cream_cheese", 0.6, 0.30, 0.04, (0.06, 0.04, 0.03))])
    gen = _gen(ep, det, init_dets=[_basket(), _det("cream_cheese", 0.6, 0.0, 0.06, (0.06, 0.04, 0.03))])
    _release_r9(gen)
    gen.tracker.update(229, CAM, det.at(229))  # now measured 30 cm away in 3D
    vals = {K: "0"}
    gen.verifier.begin()
    geom = gen._geometry_for(vals, CAM)
    assert geom[K]["mode"] == "depth" and geom[K]["inside"] is False
    assert gen._geometry_memory(vals, geom, CAM) == {}
    gen._observe_apply(vals, 229, {}, geom, set())
    assert gen.graph.p(K) == 0.1


# --- B: t08 d3 -- settled frame -----------------------------------------------------------------------------------

def _d3(fall_until=243, hand_leaves=260, still=True):
    """Release at t=235 with the cheese 13.3 cm above the basket's median point, falling into it by ``fall_until``;
    the fingers open over 235-241, the hand rises from ``hand_leaves``."""
    rel = np.array([0.6, 0.038, 0.183])

    def eef(t):
        return rel + [0, 0, 0.01 * max(t - hand_leaves, 0)]

    def width(t):
        return 0.03 + 0.05 * min(max(t - 235, 0), 6) / 6

    def cheese_z(t):
        if not still:
            return 0.183 - 0.006 * max(t - 235, 0)  # keeps moving 2.4 cm per check (never still)
        return 0.183 - (0.183 - 0.05) * min(max(t - 235, 0), fall_until - 235) / (fall_until - 235)

    det = _Det(lambda t: [_basket(), _det("cream_cheese", 0.6, 0.038, cheese_z(t), (0.06, 0.04, 0.03))])
    ep = _Ep(600, eef, width)
    gen = _gen(ep, det)
    gen.tracker.hold(CHEESE, "main", rel)
    assert gen.tracker.release("main", 235, rel) == CHEESE
    return gen, rel


def test_t08_d3_geometry_at_the_settled_frame_is_inside():
    gen, rel = _d3()
    # what fix3 computed at the release frame: 13.3 cm above the median point, above the 8 cm basket -> outside
    old = gen.geo.pair(K, CAM, rel)
    assert old["inside_xy"] is True and old["inside_z"] is False and old["inside"] is False
    tmp, info = gen._settle(Event(235, "release", "main"), "main", CHEESE, [BASKET], 400)
    assert info["frame"] == "settle" and info["why"] == "object still" and 243 <= info["t"] <= 255
    assert info["supports_detected"] == [BASKET] and all("hand_away_m" in c for c in info["checks"])
    geom = gen._geometry_for({K: "0"}, CAM, release={CHEESE: rel}, settle=(tmp, info))
    g = geom[K]
    assert g["frame"] == "settle" and g["t_geo"] == info["t"] and g["inside"] is True and g["inside_z"] is True
    assert "height_veto" not in g
    # the tracker itself is untouched (the event's frame state)
    assert gen.tracker.tracks[CHEESE].pos[2] > 0.18
    # VDM 0 at the release vs inside at the settled frame -> R10 conflict
    gen.verifier.begin()
    gen._observe_apply({K: "0"}, 235, {}, geom, set())
    assert gen.graph.p(K) == 0.5 and gen.verifier.new_conflicts == [K]


def test_no_settled_frame_falls_back_to_the_release_frame_without_a_height_veto():
    gen, rel = _d3(still=False, hand_leaves=10_000)  # never still, the hand never leaves within 2 s
    tmp, info = gen._settle(Event(235, "release", "main"), "main", CHEESE, [BASKET], 400)
    assert tmp == {} and info["frame"] == "release" and info["why"].startswith("no settled frame")
    g = gen._geometry_for({K: "0"}, CAM, release={CHEESE: rel}, settle=(tmp, info))[K]
    assert g["frame"] == "release" and g["inside_z"] is False and g["inside"] is True and g["height_veto"].startswith("skipped")
    # the settle search stops before the next gripper event
    tmp, info = gen._settle(Event(235, "release", "main"), "main", CHEESE, [BASKET], 240)
    assert info["frame"] == "release" and all(c["t"] < 240 for c in info["checks"])


def test_hand_leaving_is_a_settled_frame_and_mode_b_uses_the_release_frame():
    gen, rel = _d3(still=False, hand_leaves=245)  # never still, but the hand is 5 cm away at t=250
    tmp, info = gen._settle(Event(235, "release", "main"), "main", CHEESE, [BASKET], 400)
    assert info["frame"] == "settle" and info["why"] == "hand left" and info["checks"][-1]["hand_away_m"] >= 0.05
    gen.cfg.mode = "B"
    tmp, info = gen._settle(Event(235, "release", "main"), "main", CHEESE, [BASKET], 400)
    assert tmp == {} and info["frame"] == "release" and info["why"].startswith("mode B")


def test_mode_b_keeps_the_release_frame_rim_veto(tmp_path):
    """The swap episode's 2nd release (bowl 6.6 cm off the plate center at the release, then sliding): mode B has no later
    frame -> release-frame geometry, height skipped, the horizontal rim veto (R5) as before."""
    from capx.bggen.fake_models import FakeServer
    from capx.bggen.models import Models
    from capx.bggen.pipeline import run_episode
    from tests.bggen_fakes import FakeDetector, TruthBrain, write_swap_episode

    w = write_swap_episode(tmp_path)
    with FakeServer(TruthBrain(w)) as srv:
        run_episode(tmp_path, tmp_path / "out", lm_cfg("B"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    trace = [json.loads(x) for x in (tmp_path / "out" / "trace.jsonl").read_text().splitlines()]
    rel = [t for t in trace if t["event"]["kind"] == "release"][1]
    assert rel["tools"]["settle"]["frame"] == "release"
    g = rel["tools"]["geometry"]["(on bowl_B plate_B)"]
    assert g["frame"] == "release" and "dz" not in g and "dz_unsettled" in g
    assert any(d["rule"] == "R5" and d["key"] == "(on bowl_B plate_B)" for d in rel["decisions"])


# --- C: cap assignment radius ---------------------------------------------------------------------------------------

MW = (0.5, 0.35, 0.3)


def _b1k_tracker():
    tr = Tracker(b1k_cfg().tracks)
    tr.initialize(0, "head", [_det("countertop", 1.0, 0.0, 0.9, (2.0, 0.6, 0.9))])
    tr.expect("microwave", "microwave_A")
    tr.update(1000, "head", [_det("microwave", 1.0, 0.5, 1.2, MW)])  # bound
    tr.tracks["microwave_A"].seen_now = False
    return tr


def test_full_category_detection_4m_away_is_dropped_and_1m_is_assigned():
    tr = _b1k_tracker()
    assert tr.cfg.cap_assign_radius == 1.0
    pos0 = tr.tracks["microwave_A"].pos.copy()
    # B1K 040 t1229: a microwave detection 4.1 m (box gap) from microwave_A -> dropped, not microwave_A
    rep = tr.update(1229, "head", [_det("microwave", 5.6, 0.5, 1.2, MW)])
    assert rep["new"] == [] and rep["dropped"] == 1 and "microwave_A" not in rep.get("matched", {})
    assert tr.counts["capped_dropped"] == 1 and np.allclose(tr.tracks["microwave_A"].pos, pos0)
    d = tr.capped_dropped[-1]
    assert d["t"] == 1229 and d["category"] == "microwave" and d["nearest"] == "microwave_A" and d["unit"] == "m"
    assert abs(d["distance"] - 4.1) < 0.01 and d["limit"] == 1.0
    # 0.8 m (box gap 0.3 m < 1 m): assigned as before
    rep = tr.update(1300, "head", [_det("microwave", 1.8, 0.5, 1.2, MW)])
    assert tr.counts["capped_assigned"] == 1 and tr.capped[-1]["into"] == "microwave_A"
    assert np.allclose(tr.tracks["microwave_A"].pos[:2], [1.8, 0.5])


def test_cap_radius_measures_a_released_object_from_its_release_place():
    tr = Tracker(b1k_cfg().tracks)
    tr.initialize(0, "head", [_det("countertop", 1.0, 0.0, 0.9, (2.0, 0.6, 0.9))])
    tr.set_caps({"hotdog": 1})
    tr.update(10, "head", [_det("hotdog", 1.0, 0.0, 0.95, (0.15, 0.05, 0.05))])
    a = "hotdog_A"
    tr.tracks[a].pos = np.array([3.0, 0.0, 0.95])  # stale position far from the release place
    tr.tracks[a].released_at = (500, np.array([1.0, 2.0, 0.95]), None)
    tr.tracks[a].seen_now = False
    rep = tr.update(510, "head", [_det("hotdog", 1.3, 2.4, 0.95, (0.15, 0.05, 0.05))])  # 0.5 m from the release place
    assert rep["new"] == [] and tr.counts["capped_dropped"] == 0
    assert a in rep.get("matched", {}) or any(r["into"] == a for r in rep.get("merged", []) + rep.get("identified", []))
    # 2.55 m from both the track and the release place (040 t5892): dropped
    tr.tracks[a].seen_now = False
    tr.tracks[a].released_at = (600, np.array([1.3, 2.4, 0.95]), None)
    rep = tr.update(5892, "head", [_det("hotdog", 1.3, 4.95, 0.95, (0.15, 0.05, 0.05))])
    assert tr.counts["capped_dropped"] == 1 and tr.capped_dropped[-1]["distance"] >= 2.5


def test_libero_has_no_caps_and_closed_world_assignment_is_unchanged():
    cfg = lm_cfg().tracks
    assert cfg.cap_assign_radius == 0.0 and not cfg.max_instances and not cfg.cap_from_init
    tr = Tracker(cfg)
    tr.initialize(0, CAM, [_det("bowl", 0.6, 0.0, 0.04), _det("bowl", 0.6, -0.2, 0.04)])
    tr.tracks["bowl_A"].seen_now = False
    # bowl_B seen in place, a bowl detection 1.5 m from bowl_A: closed world -> still bowl_A (no radius), no drop
    rep = tr.update(50, CAM, [_det("bowl", 0.6, -0.2, 0.04), _det("bowl", 2.1, 0.0, 0.04)])
    assert rep["new"] == [] and tr.counts["capped_dropped"] == 0 and tr.counts["closed_assigned"] == 1
    assert np.allclose(tr.tracks["bowl_A"].pos[:2], [2.1, 0.0])


def test_countertop_is_exempt_from_the_cap_radius():
    tr = Tracker(b1k_cfg().tracks)
    tr.initialize(0, "head", [_det("countertop", 1.0, 0.0, 0.9, (2.0, 0.6, 0.9))])
    tr.tracks["countertop_A"].seen_now = False
    rep = tr.update(20, "head", [_det("countertop", 3.5, 3.0, 0.9, (2.0, 0.6, 0.9))])  # an L-shaped counter's far slab
    assert tr.counts["capped_dropped"] == 0 and rep["new"] == []


# --- D: aggregate -------------------------------------------------------------------------------------------------------

def _ep_dir(root, name, reobs, decided, acc):
    d = root / name
    d.mkdir(parents=True)
    m = {"relations": {"accuracy": acc, "decided": decided, "undecided": 0, "undecided_rate": 0.0, "per_pred": {}},
         "subtask_boundary": {}, "reobservation": reobs}
    (d / "eval.json").write_text(json.dumps({"forward": m, "forward+backward": m}))
    (d / "summary.json").write_text(json.dumps({"goal": {}, "models": {}}))


def test_aggregate_no_reobservation_is_no_rate_and_decided_is_beside_accuracy(tmp_path):
    _ep_dir(tmp_path, "libero_mem_t00_demo_1", {"rate": 0.0, "n": 0}, 120, 0.8)  # an old eval.json wrote 0.0 for n=0
    _ep_dir(tmp_path, "libero_mem_t00_demo_2", {"rate": None, "n": 0}, 100, 0.9)
    res = agg.aggregate(tmp_path)
    g = res["groups"]["libero_mem t00"]["forward"]
    assert g["reobs"] is None and g["reobs_n"] == 0 and g["decided"] == 220
    md = agg.to_markdown(res)
    row = next(x for x in md.splitlines() if x.startswith("| libero_mem t00 | fwd |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3] == "0.850 [0.800, 0.900]" and cells[4] == "220" and cells[-1] == "- (0 cases)"
    assert "0.00 (" not in row
    ep = next(x for x in md.splitlines() if x.startswith("| libero_mem_t00_demo_1 |"))
    assert "| 120 |" in ep and "- (0)" in ep
    # one disagreeing case in one episode: shown as a count of cases and episodes, not like a rate over episodes
    _ep_dir(tmp_path, "libero_mem_t00_demo_3", {"rate": 0.0, "n": 1}, 90, 0.7)
    row = next(x for x in agg.to_markdown(agg.aggregate(tmp_path)).splitlines() if x.startswith("| libero_mem t00 | fwd |"))
    assert row.rstrip().endswith("| 0.00 (1 case in 1 ep) |")
