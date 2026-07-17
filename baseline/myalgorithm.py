# myalgorithm.py
# OGC 2026 solver -- entry point for the evaluation server.
#
# The server calls `algorithm(prob_info, timelimit)` and scores the returned
# solution with the UNMODIFIED utils.check_feasibility.  All of our logic lives
# in this single file; only validated pieces are imported from utils.py (which
# must never be modified).  See CLAUDE.md for the full design and roadmap.
#
# Structure (light OO so LNS / incremental state in later layers fit cleanly):
#   Placement   -- one block's committed decision (bay, pos, orient, times)
#   Solver      -- holds instance + time budget + best solution; solve() runs
#                  the layered pipeline and always returns a feasible dict.
#
# Layer 0 (done): guaranteed-feasible "serial-per-bay" floor.  Each bay holds
#   at most one block at a time, so |N(t,j)| <= 1 and the layer-collision and
#   crane-operation constraints hold BY CONSTRUCTION.  On the leaderboard
#   infeasible/timed-out/crashed = -1 while any feasible >= 1, so never
#   returning -1 is the highest-value property; this floor secures it.
#
# Layer 1 (done): crane-aware coexistence construction.  Blocks share a bay
#   whenever geometry AND crane access allow.  The key fix over the provided
#   baseline: when placing block N we also verify that N does not obstruct the
#   crane moments (entry/exit) of already-placed blocks that occur DURING N's
#   stay.  The baseline only checked the forward direction (others obstructing
#   N), which is exactly why 97% of its infeasibilities were crane stages 2/3
#   and its repair loop thrashed.  Checking both directions up front makes the
#   construction feasible-by-search; the official check_feasibility still
#   validates the result before it can replace the floor.
#
# Layer 2 (done): regime-adaptive LNS (destroy-repair) on the leftover time
#   budget.  Feasibility is preserved by construction (removal only relaxes
#   the pairwise constraints; reinsertion reuses the Layer-1 bidirectional
#   search), the objective is tracked with EXACTLY the checker's arithmetic,
#   and the final result is still validated once by check_feasibility before
#   it may replace the incumbent.
#
# Layer 3 (done): Gurobi timing MILP.  Placements stay fixed; entry/exit
#   times of each bay are re-optimised by an independent small MILP (bays
#   have independent cranes, and with fixed placements only obj1 depends on
#   time).  Warm-started from the incumbent, hard time-sliced, and accepted
#   under the same feasible-and-better-or-rejected contract.  Without
#   gurobipy or a license it degrades to a no-op.

import time
import math
import random
from dataclasses import dataclass

import shapely

from utils import (Bay, Block, check_feasibility,
                   _resolve_layers, _bounding_box, _poly_from_verts)


# Fraction of the wall-clock limit we allow ourselves to use overall.  Below
# 1.0 to leave margin (the dev machine may be slower than the server, and
# building the operations dict + final feasibility check also cost time).  All
# time budgeting is RELATIVE to `timelimit` -- no absolute constants tuned to
# the training instances.
TIME_SAFETY = 0.90

# Layer-1 construction must stop at this fraction of the time limit; the gap
# up to LNS_SAFETY belongs to Layer-2 improvement, and the gap up to
# TIME_SAFETY is reserved for building the operations dict and running the
# official check_feasibility on the final candidate.
CONSTRUCT_SAFETY = 0.72
LNS_SAFETY = 0.82

# Layer-3 (Gurobi timing MILP) runs in the [LNS_SAFETY, MILP_SAFETY] window;
# the remainder up to TIME_SAFETY stays reserved for the final official check.
# If gurobipy is unavailable the window is simply unused (extra margin).
MILP_SAFETY = 0.88


# ---------------------------------------------------------------------------
# One block's committed decision
# ---------------------------------------------------------------------------

@dataclass
class Placement:
    block_id: int
    bay_id: int
    x: int
    y: int
    orient_idx: int
    entry_time: int
    exit_time: int


# ---------------------------------------------------------------------------
# Per-bay record of a committed block (Layer-1 state)
# ---------------------------------------------------------------------------

@dataclass
class _Rec:
    entry: int
    exit: int
    bid: int
    block: object          # utils.Block (world-coordinate layers cached)
    bbox: tuple            # world bbox (x0, y0, x1, y1)
    geo: list | None = None  # lazy fast-predicate geometry (see Solver._rec_geo)


# ---------------------------------------------------------------------------
# Small validated helpers (inlined from baseline_greedy so the submission
# depends only on myalgorithm.py + utils.py)
# ---------------------------------------------------------------------------

def _block_bbox(block_data: dict, orient_idx: int) -> tuple[float, float, float, float]:
    """Local bbox (min_x, min_y, max_x, max_y) of a block relative to its
    reference point (first vertex of first layer = (0, 0))."""
    raw_layers = block_data["shape"][orient_idx]["layers"]
    layers = _resolve_layers(raw_layers)
    if not layers:
        return (0.0, 0.0, 1.0, 1.0)
    all_verts = [v for l in layers for v in l]
    return _bounding_box(all_verts)


def _build_operations(placements: list[Placement]) -> dict:
    """Group placements into the solution "operations" dict: keyed by integer
    time (as str), EXIT before ENTRY at each time, then ordered by block_id.
    NOTE: the same-time tie-break rules in the Layer-1 crane checks assume
    exactly this ordering (EXITs then ENTRYs, block_id ascending within type)."""
    buckets: dict[int, list[tuple]] = {}
    for p in placements:
        buckets.setdefault(p.exit_time, []).append((0, "EXIT", p.block_id, p.bay_id, None, None, None))
        buckets.setdefault(p.entry_time, []).append((1, "ENTRY", p.block_id, p.bay_id, p.x, p.y, p.orient_idx))

    operations: dict[str, list[dict]] = {}
    for t in sorted(buckets):
        ops = sorted(buckets[t], key=lambda r: (r[0], r[2]))
        result = []
        for _, kind, bid, bay, x, y, oi in ops:
            op: dict = {"type": kind, "block_id": bid, "bay_id": bay}
            if kind == "ENTRY":
                op["x"] = x
                op["y"] = y
                op["orient_idx"] = oi
            result.append(op)
        operations[str(t)] = result
    return operations


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class Solver:
    """Holds the instance, time budget, and best-so-far solution."""

    def __init__(self, prob_info: dict, timelimit: float):
        self.prob = prob_info
        self.timelimit = float(timelimit)
        self.t_start = time.time()
        self.deadline = self.t_start + max(1.0, self.timelimit) * TIME_SAFETY

        self.blocks = prob_info["blocks"]
        self.bays = [Bay.from_dict(d, i) for i, d in enumerate(prob_info["bays"])]
        w = prob_info.get("weights", {})
        self.w1 = w.get("w1", 1.0)
        self.w2 = w.get("w2", 1.0)
        self.w3 = w.get("w3", 1.0)

        self._bbox_cache: dict = {}
        self._geo_cache: dict = {}  # (id(block_data), orient) -> (local layers, local layer bboxes)
        # Crane-flag cache keyed by the RELATIVE placement of two shapes (see
        # _flag): the sweep predicate depends only on shape identities and the
        # integer offset between reference points, so results survive across
        # LNS iterations that reinsert the same block near the same neighbours
        # (measured hit-rate ~73% on prob_23, ~3x the iterations/second).
        # Cleared on overflow to bound memory under the server's 16GB cap.
        self._flag_cache: dict = {}
        self._flag_cache_cap = 2_000_000
        self.best_sol: dict | None = None  # always feasible once the floor is built

    # -- time ----------------------------------------------------------------
    def time_left(self) -> float:
        return self.deadline - time.time()

    def elapsed(self) -> float:
        return time.time() - self.t_start

    # -- geometry ------------------------------------------------------------
    def _bbox(self, block_data: dict, orient_idx: int):
        key = (id(block_data), orient_idx)
        v = self._bbox_cache.get(key)
        if v is None:
            v = _block_bbox(block_data, orient_idx)
            self._bbox_cache[key] = v
        return v

    # -- fast crane predicate (search-only mirror of utils.check_entry) ------
    #
    # utils.check_entry is pure geometry (no time dependence): mover layer k
    # obstructed by existing layer j >= k iff their polygons intersect with
    # area > 0.  The official checker still validates every accepted solution;
    # this mirror only replaces check_entry inside the SEARCH hot path, with
    # identical decisions but (a) per-layer bbox prefilters (pure arithmetic),
    # (b) shapely prepared-geometry `intersects` gates, and (c) polygons built
    # once per committed block / candidate instead of per call.
    # The bay-containment branch of check_entry is dropped deliberately: every
    # candidate position is generated inside [px_lo, px_hi] x [py_lo, py_hi],
    # so footprints are inside the bay by construction (it never fired).

    _UNBUILT = object()  # sentinel: polygon not constructed yet (None = degenerate)

    def _local_geo(self, block_data: dict, orient_idx: int):
        """(local layer vertex lists, local layer bboxes), reference point at
        the origin -- world coordinates are local + (x, y), exactly as
        utils.Block.__post_init__ translates them."""
        key = (id(block_data), orient_idx)
        g = self._geo_cache.get(key)
        if g is None:
            layers = _resolve_layers(block_data["shape"][orient_idx]["layers"])
            if layers:
                rx, ry = layers[0][0] if layers[0] else (0.0, 0.0)
                layers = [[(vx - rx, vy - ry) for vx, vy in l] for l in layers]
            bboxes = [_bounding_box(l) for l in layers]
            g = (layers, bboxes)
            self._geo_cache[key] = g
        return g

    def _make_geo(self, block_data: dict, orient_idx: int, x: float, y: float) -> list:
        """Geometry bundle [world bboxes, lazy polys, local layers, x, y,
        shape_key] for one placed shape.  Polygons are built (and prepared) on
        first use.  shape_key = (id(block_data), orient) is stable for the
        solver's lifetime (block dicts live in self.blocks) and identifies the
        translation-invariant shape for the _flag cache."""
        loc, lb = self._local_geo(block_data, orient_idx)
        wb = [(b0 + x, b1 + y, b2 + x, b3 + y) for b0, b1, b2, b3 in lb]
        return [wb, [Solver._UNBUILT] * len(loc), loc, x, y,
                (id(block_data), orient_idx)]

    def _rec_geo(self, rec: _Rec) -> list:
        g = getattr(rec, "geo", None)
        if g is None:
            g = self._make_geo(self.blocks[rec.bid], rec.block.orient_idx,
                               rec.block.x, rec.block.y)
            rec.geo = g
        return g

    @staticmethod
    def _geo_poly(geo: list, i: int):
        p = geo[1][i]
        if p is Solver._UNBUILT:
            loc, x, y = geo[2], geo[3], geo[4]
            p = _poly_from_verts([[vx + x, vy + y] for vx, vy in loc[i]])
            if p is not None:
                shapely.prepare(p)
            geo[1][i] = p
        return p

    @staticmethod
    def _sweep_blocked(mover_geo: list, exist_geo: list) -> bool:
        """True iff the mover's vertical sweep is obstructed by `exist`:
        exists (k, j), j >= k, with area(mover layer k  ∩  exist layer j) > 0.
        Decision-identical to utils.check_entry conditions 2 & 3."""
        m_wb = mover_geo[0]
        e_wb = exist_geo[0]
        n_e = len(e_wb)
        for k in range(len(m_wb)):
            mk = m_wb[k]
            mp = Solver._UNBUILT
            for j in range(k, n_e):
                ej = e_wb[j]
                if mk[2] <= ej[0] or ej[2] <= mk[0] or mk[3] <= ej[1] or ej[3] <= mk[1]:
                    continue  # layer bboxes disjoint -> polygons disjoint
                if mp is Solver._UNBUILT:
                    mp = Solver._geo_poly(mover_geo, k)
                if mp is None:
                    break  # degenerate mover layer obstructs nothing
                ep = Solver._geo_poly(exist_geo, j)
                if ep is None:
                    continue
                if not shapely.intersects(mp, ep):
                    continue  # prepared-geometry gate (cheap)
                # For polygonal geometry area(A ∩ B) > 0  <=>  the interiors
                # meet (an open non-empty 2D set has positive area), i.e.
                # intersects AND NOT touches -- both prepared-accelerated,
                # ~50x cheaper than materialising the intersection.
                if not shapely.touches(mp, ep):
                    return True
        return False

    def _flag(self, mover_geo: list, exist_geo: list) -> bool:
        """Cached _sweep_blocked.  The sweep is translation-invariant: its
        result depends only on the two shapes and the integer offset between
        their reference points, so (shape_mover, shape_exist, dx, dy) keys a
        result that stays valid whenever the same block is reinserted at the
        same relative placement to the same neighbour -- the common case in
        LNS reinsertion.  Positions are integers throughout the search, so dx,
        dy are exact.  The cache is cleared wholesale on overflow (recent
        offsets dominate the hit-rate, so the re-warm cost is small)."""
        key = (mover_geo[5], exist_geo[5],
               exist_geo[3] - mover_geo[3], exist_geo[4] - mover_geo[4])
        c = self._flag_cache
        v = c.get(key)
        if v is None:
            v = 1 if Solver._sweep_blocked(mover_geo, exist_geo) else 0
            if len(c) >= self._flag_cache_cap:
                c.clear()
            c[key] = v
        return v == 1

    def _fit_position(self, block_data: dict, bay: Bay) -> tuple[int, int, int] | None:
        """First orientation with a valid integer reference-point position in
        `bay`, placed at the minimum valid position; None if it fits in none.

        Valid integer position for orientation oi exists iff
            ceil(-lx0) <= floor(W - lx1)  and  ceil(-ly0) <= floor(H - ly1).
        Bay containment then holds by arithmetic (the bay is a rectangle, so
        polygon-in-bay <=> bbox-in-bay)."""
        for oi in range(len(block_data["shape"])):
            lx0, ly0, lx1, ly1 = self._bbox(block_data, oi)
            px_lo = math.ceil(-lx0)
            px_hi = math.floor(bay.width - lx1)
            py_lo = math.ceil(-ly0)
            py_hi = math.floor(bay.height - ly1)
            if px_lo <= px_hi and py_lo <= py_hi:
                return (oi, max(0, px_lo), max(0, py_lo))
        return None

    # ======================================================================
    # Layer 0: guaranteed-feasible serial-per-bay floor
    # ======================================================================

    def build_floor(self) -> list[Placement]:
        """Serial-per-bay construction in EDD order.  Each block goes to the
        fitting bay that minimises a tardiness-first score and is scheduled
        right after that bay's current free time, so no two blocks are ever
        co-present in the same bay -> feasible by construction."""
        n_bays = len(self.bays)

        fit: dict[tuple[int, int], tuple[int, int, int]] = {}
        for bi, bd in enumerate(self.blocks):
            for bay in self.bays:
                f = self._fit_position(bd, bay)
                if f is not None:
                    fit[(bi, bay.id)] = f

        free_time = [0] * n_bays  # earliest time each bay is empty again
        order = sorted(
            range(len(self.blocks)),
            key=lambda i: (self.blocks[i]["due_date"], self.blocks[i]["processing_time"]),
        )

        placements: list[Placement] = []
        for bi in order:
            bd = self.blocks[bi]
            r = bd["release_time"]
            proc = bd["processing_time"]
            due = bd["due_date"]
            prefs = bd["bay_preferences"]
            s_max = max(prefs)

            cand_bays = [bay.id for bay in self.bays if (bi, bay.id) in fit]

            if cand_bays:
                best = None  # (score, bay_id, entry, exit_t)
                for bay_id in cand_bays:
                    entry = max(r, free_time[bay_id])
                    exit_t = entry + proc
                    tardiness = max(0, exit_t - due)
                    pref_pen = s_max - prefs[bay_id]
                    score = self.w1 * tardiness + self.w3 * pref_pen
                    if best is None or score < best[0]:
                        best = (score, bay_id, entry, exit_t)
                _, bay_id, entry, exit_t = best
                oi, x, y = fit[(bi, bay_id)]
            else:
                # Block fits in no bay (malformed instance): still return a
                # dict rather than crash; the official check will flag it.
                bay_id = max(range(n_bays), key=lambda j: prefs[j])
                oi, x, y = 0, 0, 0
                entry = max(r, free_time[bay_id])
                exit_t = entry + proc
                print(f"[ogc]   WARNING: block {bi} fits in no bay -- last-resort placement")

            free_time[bay_id] = exit_t
            placements.append(Placement(bi, bay_id, int(x), int(y), oi, int(entry), int(exit_t)))

        return placements

    # ======================================================================
    # Layer 1: crane-aware coexistence construction
    # ======================================================================
    #
    # For every candidate (bay, orientation, position, entry time) we verify,
    # pairwise against each committed block whose stay interacts with ours,
    # exactly the conditions check_feasibility enforces:
    #
    #   * OUR crane moments (stages 2/3): any block present when we enter or
    #     exit must not intersect our vertical sweep (utils.check_entry --
    #     the j>=k predicate is identical for ENTRY and EXIT per the problem
    #     definition and the utils docstring).
    #   * THEIR crane moments -- the check the provided baseline misses: if a
    #     committed block enters or exits while WE are in the bay, we must not
    #     obstruct ITS sweep.  This omission caused 97% of the baseline's
    #     infeasibilities (crane stages 2/3) and its repair thrashing.
    #   * Same-layer collision (stage 4) needs no separate test: the j>=k
    #     sweep includes j==k, and every time-overlapping pair triggers at
    #     least one crane check in some direction (proof: if a<t the pair
    #     triggers "present at our entry"; if a==t the equal-entry branch; if
    #     a>t the "we present at their entry" branch), so collision-freedom is
    #     implied.
    #
    # Same-time boundaries (equal entry or exit times) are checked in BOTH
    # directions: slightly conservative, but safe regardless of replay order.
    #
    # Anytime: a per-block budget derived from the remaining construction time
    # caps the search; on exhaustion the incumbent is committed or the
    # guaranteed empty-window fallback is used, so construction always
    # completes before CONSTRUCT_SAFETY * timelimit.

    def build_coexist(self, bay_of: dict[int, int] | None = None,
                      deadline: float | None = None,
                      order_rng: random.Random | None = None,
                      order_noise: float = 0.08) -> list[Placement]:
        """Coexistence construction.  With `bay_of` (block_id -> bay) the
        search is restricted to the given bay per block, so a Layer-3 master
        assignment can be realised by a full rebuild instead of local patches.
        With `order_rng` the EDD insertion order is perturbed by
        +-`order_noise` of the due-date span -- used by the restart rounds to
        escape the greedy construction's own quality ceiling on congested
        instances (the driver cycles the magnitude: gentle nudges first, then
        aggressive reshuffles).
        """
        if deadline is None:
            deadline = self.t_start + self.timelimit * CONSTRUCT_SAFETY
        n_bays = len(self.bays)
        areas = [b.width * b.height for b in self.bays]
        avg_area = sum(areas) / n_bays
        u = [avg_area / a for a in areas]

        state: list[list[_Rec]] = [[] for _ in range(n_bays)]
        loads = [0.0] * n_bays
        if order_rng is None:
            order = sorted(
                range(len(self.blocks)),
                key=lambda i: (self.blocks[i]["due_date"], self.blocks[i]["processing_time"]),
            )
        else:
            dues = [b["due_date"] for b in self.blocks]
            span = float(max(dues) - min(dues)) or 1.0
            noisy = {i: self.blocks[i]["due_date"]
                     + order_rng.uniform(-order_noise, order_noise) * span
                     for i in range(len(self.blocks))}
            order = sorted(range(len(self.blocks)),
                           key=lambda i: (noisy[i], self.blocks[i]["processing_time"]))

        placements: list[Placement] = []
        n_fallback = 0
        for k, bi in enumerate(order):
            forced = bay_of.get(bi) if bay_of is not None else None
            now = time.time()
            p = None
            if now < deadline:
                budget = max(0.002, (deadline - now) / (len(order) - k))
                p = self._place_coexist(bi, state, loads, u, now + budget,
                                        only_bay=forced)
            if p is None:
                p = self._fallback_place(bi, state, only_bay=forced)
                n_fallback += 1
            self._commit(p, state, loads)
            placements.append(p)

        if n_fallback and bay_of is None and order_rng is None:
            print(f"[ogc]   Layer1: {n_fallback} fallback placement(s)")
        return placements

    def _place_coexist(self, bi: int, state, loads, u, t_deadline,
                       thorough: bool = False, rng=None,
                       only_bay: int | None = None) -> Placement | None:
        """Best placement for block bi against the committed state.

        thorough=False (Layer-1 construction): accept the first minimal-
          tardiness slot in the most-preferred bay -- fast, tardiness-first.
        thorough=True (Layer-2 repair): explore every non-pruned bay so the
          w2 (imbalance) and w3 (preference) trade-off is evaluated exactly;
          `rng` additionally applies +-5% score noise to diversify repairs.
        only_bay (Layer-3 reassignment): restrict the search to one bay so a
          MILP-chosen assignment is realised exactly.
        """
        bd = self.blocks[bi]
        r = bd["release_time"]
        proc = bd["processing_time"]
        due = bd["due_date"]
        workload = bd["workload"]
        prefs = bd["bay_preferences"]
        s_max = max(prefs)
        w1, w2, w3 = self.w1, self.w2, self.w3
        n_bays = len(self.bays)
        min_tard = max(0, r + proc - due)   # unavoidable tardiness lower bound

        best = None  # (score, bay_id, x, y, oi, entry, exit_t)

        def imbalance_with(bay_id: int, new_load: float) -> float:
            worst = 0.0
            for j1 in range(n_bays):
                l1 = new_load if j1 == bay_id else loads[j1]
                v1 = u[j1] * l1
                for j2 in range(j1 + 1, n_bays):
                    l2 = new_load if j2 == bay_id else loads[j2]
                    d = abs(v1 - u[j2] * l2)
                    if d > worst:
                        worst = d
            return worst

        if only_bay is not None:
            bay_order = [only_bay]
        else:
            bay_order = sorted(range(n_bays), key=lambda j: prefs[j], reverse=True)
        for bay_id in bay_order:
            pref_pen = s_max - prefs[bay_id]
            # Bound: even with minimal tardiness this bay cannot beat the
            # incumbent -> skip it entirely.
            if best is not None and w1 * min_tard + w3 * pref_pen >= best[0]:
                continue
            bay = self.bays[bay_id]
            # Blocks already gone before our release never interact with us.
            recs = [rec for rec in state[bay_id] if rec.exit > r]
            exit_ts = sorted({rec.exit for rec in recs})

            found_min_tard = False
            for oi in range(len(bd["shape"])):
                # HARD budget guard: overrunning it compounds catastrophically
                # (a crowded bay makes every later search slower), so honour it
                # even with no incumbent -- the caller then uses the fallback.
                if time.time() > t_deadline:
                    return self._as_placement(bi, best)
                bb = self._bbox(bd, oi)
                lx0, ly0, lx1, ly1 = bb
                px_lo = math.ceil(-lx0)
                px_hi = math.floor(bay.width - lx1)
                py_lo = math.ceil(-ly0)
                py_hi = math.floor(bay.height - ly1)
                if px_lo > px_hi or py_lo > py_hi:
                    continue

                # Candidate positions: bottom-left fill against active blocks.
                # Both edge sets are capped (left-most / bottom-most kept) so
                # the conflict-count pass below stays bounded regardless of how
                # crowded the bay is.
                xs = {max(0, px_lo)}
                ys = {max(0, py_lo)}
                for rec in recs:
                    x_c = math.ceil(rec.bbox[2] - lx0)
                    if px_lo <= x_c <= px_hi:
                        xs.add(x_c)
                    y_c = math.ceil(rec.bbox[3] - ly0)
                    if py_lo <= y_c <= py_hi:
                        ys.add(y_c)
                xs = sorted(xs)[:30]
                ys = sorted(ys)[:30]

                # Order positions by (bbox-conflict count, top edge): positions
                # whose bbox overlaps nothing active can enter at release time
                # with zero shapely work, so they are tried first.
                cands = []
                for x in xs:
                    wx0 = x + lx0
                    wx1 = x + lx1
                    for y in ys:
                        wy0 = y + ly0
                        wy1 = y + ly1
                        cnt = 0
                        for rec in recs:
                            rb = rec.bbox
                            if not (wx1 <= rb[0] or rb[2] <= wx0 or wy1 <= rb[1] or rb[3] <= wy0):
                                cnt += 1
                        cands.append((cnt, wy1, x, y))
                cands.sort()

                for cnt, top_y, x, y in cands:
                    if time.time() > t_deadline:
                        return self._as_placement(bi, best)  # hard budget: commit incumbent (may be None)
                    bound = best[0] if best is not None else None
                    slot = self._earliest_slot(bd, oi, bb, x, y, recs,
                                               exit_ts, r, proc, due, pref_pen, bound,
                                               t_deadline)
                    if slot is None:
                        continue
                    entry, exit_t = slot
                    tard = max(0, exit_t - due)
                    score = (w1 * tard
                             + w2 * imbalance_with(bay_id, loads[bay_id] + workload)
                             + w3 * pref_pen
                             + 1e-4 * top_y)
                    if rng is not None:
                        score *= 1.0 + 0.05 * rng.random()
                    if best is None or score < best[0]:
                        best = (score, bay_id, x, y, oi, entry, exit_t)
                    if tard == min_tard:
                        found_min_tard = True
                        break  # tardiness cannot improve further in this bay
                if found_min_tard:
                    break
            if found_min_tard and pref_pen == 0 and not thorough:
                break  # minimal tardiness in the most-preferred bay: accept

        return self._as_placement(bi, best)

    @staticmethod
    def _as_placement(bi: int, best) -> Placement | None:
        if best is None:
            return None
        _, bay_id, x, y, oi, entry, exit_t = best
        return Placement(bi, bay_id, int(x), int(y), oi, int(entry), int(exit_t))

    def _earliest_slot(self, bd: dict, oi: int, bb, x: int, y: int,
                       recs: list[_Rec], exit_ts: list[int],
                       r: int, proc: int, due: int, pref_pen: float,
                       score_bound: float | None,
                       t_deadline: float = float("inf")) -> tuple[int, int] | None:
        """Earliest entry time t >= r at which the block at (x, y, oi) is
        feasible against all committed blocks in this bay, mirroring
        check_feasibility pairwise IN BOTH DIRECTIONS (see class comment).
        Candidate entries: r and every later exit time (bay state only changes
        at exits).  The last candidate (= max exit) always succeeds because the
        bay is empty from then on, so this returns None only when `score_bound`
        proves every remaining slot worse than the incumbent."""
        lx0, ly0, lx1, ly1 = bb
        wx0 = x + lx0
        wy0 = y + ly0
        wx1 = x + lx1
        wy1 = y + ly1
        # Blocks whose bbox is disjoint from ours can never interact spatially
        # (bbox-disjoint => polygon interiors disjoint at every layer, which
        # clears both the collision and the crane predicates).
        conf = [rec for rec in recs
                if not (wx1 <= rec.bbox[0] or rec.bbox[2] <= wx0 or
                        wy1 <= rec.bbox[1] or rec.bbox[3] <= wy0)]
        if not conf:
            return r, r + proc

        # The crane predicate is pure geometry -- independent of t -- so each
        # (candidate, committed block) pair is evaluated at most once and the
        # time loop below is pure arithmetic over the cached flags.
        # (Batching all pairs into vectorised shapely calls was tried and is
        # SLOWER here: array conversion costs ~140us per call, which loses to
        # lazy per-pair calls at the ~10 pairs a candidate typically has.)
        cand_geo = None  # built lazily: many candidates die on the first t
        flags: dict[int, tuple[bool, bool]] = {}

        def pair_flags(rec: _Rec) -> tuple[bool, bool]:
            fl = flags.get(rec.bid)
            if fl is None:
                nonlocal cand_geo
                if cand_geo is None:
                    cand_geo = self._make_geo(bd, oi, x, y)
                rg = self._rec_geo(rec)
                fl = (self._flag(cand_geo, rg),   # rec obstructs OUR sweep
                      self._flag(rg, cand_geo))   # we obstruct THEIRS
                flags[rec.bid] = fl
            return fl

        w1, w3 = self.w1, self.w3
        for t in [r] + exit_ts:
            # HARD time escape (see the budget lesson in CLAUDE.md): a crowded
            # bay can make even one candidate expensive, and overruns compound.
            if time.time() > t_deadline:
                return None
            # Later entries only increase tardiness: once even the tardiness
            # term alone is no better than the incumbent, stop.
            if score_bound is not None and \
               w1 * max(0, t + proc - due) + w3 * pref_pen >= score_bound:
                return None
            et = t + proc
            ok = True
            for rec in conf:
                a = rec.entry
                e = rec.exit
                if e <= t or a >= et:
                    continue  # no temporal interaction (EXIT-before-ENTRY makes boundaries safe)
                # OUR crane moments: rec present at our entry/exit sweep.
                if (a < t < e) or a == t or (a < et < e) or e == et:
                    if pair_flags(rec)[0]:
                        ok = False
                        break
                # THEIR crane moments: we are present at rec's entry/exit sweep.
                if (t < a < et) or a == t or (t < e < et) or e == et:
                    if pair_flags(rec)[1]:
                        ok = False
                        break
            if ok:
                return t, et
        return None

    def _fallback_place(self, bi: int, state, only_bay: int | None = None) -> Placement:
        """Guaranteed-feasible placement: empty-bay window (no committed block
        overlaps [entry, entry+proc)), minimum valid position, preferred bay
        first.  Later blocks validate against this one when they are placed."""
        bd = self.blocks[bi]
        r = bd["release_time"]
        proc = bd["processing_time"]
        prefs = bd["bay_preferences"]
        n_bays = len(self.bays)

        if only_bay is not None:
            bay_ids = [only_bay]
        else:
            bay_ids = sorted(range(n_bays), key=lambda j: prefs[j], reverse=True)
        for bay_id in bay_ids:
            f = self._fit_position(bd, self.bays[bay_id])
            if f is None:
                continue
            oi, x, y = f
            entry = int(r)
            changed = True
            while changed:  # iterative push past every overlapping interval
                changed = False
                et = entry + proc
                for rec in state[bay_id]:
                    if rec.entry < et and entry < rec.exit:
                        entry = max(entry, rec.exit)
                        changed = True
            return Placement(bi, bay_id, x, y, oi, int(entry), int(entry + proc))

        # Fits nowhere (malformed instance) -- mirror the floor's last resort.
        bay_id = max(range(n_bays), key=lambda j: prefs[j])
        print(f"[ogc]   WARNING: block {bi} fits in no bay -- last-resort placement")
        return Placement(bi, bay_id, 0, 0, 0, int(r), int(r + proc))

    def _commit(self, p: Placement, state, loads) -> _Rec:
        bd = self.blocks[p.block_id]
        blk = Block(block_id=p.block_id, block_data=bd,
                    x=p.x, y=p.y, orient_idx=p.orient_idx)
        lx0, ly0, lx1, ly1 = self._bbox(bd, p.orient_idx)
        rec = _Rec(p.entry_time, p.exit_time, p.block_id, blk,
                   (p.x + lx0, p.y + ly0, p.x + lx1, p.y + ly1))
        state[p.bay_id].append(rec)
        loads[p.bay_id] += bd["workload"]
        return rec

    # ======================================================================
    # Layer 2: LNS improvement (destroy - repair)
    # ======================================================================
    #
    # Regime-adaptive large-neighbourhood search on top of the incumbent
    # placements.  Feasibility is preserved by construction: removing blocks
    # never invalidates the rest (all cross-block constraints are pairwise, so
    # removal only relaxes them), and every reinsertion passes the same
    # bidirectional crane checks as Layer 1.  The objective is tracked with
    # EXACTLY the arithmetic utils.check_feasibility uses (incl. the floor()
    # on obj2) and only strictly improving iterations are accepted, so the
    # single official check at the end protects against bugs rather than being
    # part of the loop.
    #
    # Destroy operators attack whichever objective term currently dominates
    # (roulette over w1*obj1 / w2*obj2 / w3*obj3, plus a random share):
    #   tardy   -- worst-tardiness blocks, plus one block "squatting" on each
    #              one's earlier time window (bbox-overlapping in its bay)
    #   timewin -- whole-time-window repack: EVERY block of an offender's bay
    #              whose stay intersects the offender's window, re-sequenced in
    #              a randomised order (tardiness on congested instances is
    #              sequencing-bound -- single moves are zero-sum shifts)
    #   pref    -- blocks placed outside their most-preferred bay
    #   balance -- random blocks from the most (normalized-)loaded bay
    #   random  -- uniform random blocks
    # Repair reinserts the removed blocks in EDD order via the Layer-1 search.

    def lns_improve(self, placements: list[Placement],
                    deadline: float | None = None,
                    seed: int = 0xC0FFEE,
                    stall_frac: float | None = None,
                    do_sweep: bool = True,
                    temp: float = 0.0) -> tuple[list[Placement], int, int]:
        """Improve `placements` until `deadline` (default LNS_SAFETY *
        timelimit).  `seed` keeps runs reproducible; callers resuming a state
        that already consumed the default stream pass a different one so the
        roulette does not replay proposals that were already exhausted.
        With `stall_frac`, return early once no STRICT improvement has been
        found for that fraction of this call's window -- the caller can then
        spend the freed time on construction restarts or a fresh seed instead
        of replaying a plateaued search to the deadline.
        `do_sweep=False` skips the deterministic tardiness sweep -- callers
        resuming a branch whose sweep already ran to exhaustion pass this (a
        sweep pass costs tens of seconds on congested instances and is
        near-deterministic, so replaying it on an unchanged branch is waste).
        `temp > 0` enables simulated-annealing acceptance: a worsening delta
        is accepted with probability exp(-delta / T), where T = temp * (running
        mean of the positive deltas seen so far) * (fraction of this call's
        window remaining) -- self-scaling across instances (no absolute
        constants) and cooling to pure descent by the deadline.  Used by the
        round driver against sequencing-bound tardiness (prob_23 class), where
        greedy descent has NO improving neighbourhood and only a temporarily
        worse re-sequencing can unlock progress.  The best placements SEEN are
        returned, never the (possibly worse) final SA state.
        Returns (improved placements, accepted iterations, total iterations)."""
        if deadline is None:
            deadline = self.t_start + self.timelimit * LNS_SAFETY
        t_call = time.time()
        stall_window = None
        if stall_frac is not None:
            stall_window = max(stall_frac * (deadline - t_call), 0.02 * self.timelimit)
        n = len(self.blocks)
        n_bays = len(self.bays)
        if time.time() >= deadline or n == 0 or not placements:
            return placements, 0, 0

        rng = random.Random(seed)  # deterministic for reproducibility
        areas = [b.width * b.height for b in self.bays]
        avg_area = sum(areas) / n_bays
        u = [avg_area / a for a in areas]

        # -- mutable state rebuilt from the incumbent ------------------------
        state: list[list[_Rec]] = [[] for _ in range(n_bays)]
        loads = [0.0] * n_bays
        cur: dict[int, Placement] = {}
        rec_of: dict[int, _Rec] = {}
        for p in placements:
            rec_of[p.block_id] = self._commit(p, state, loads)
            cur[p.block_id] = p

        def exact_terms() -> tuple[float, float, float, float]:
            """(total, obj1, obj2, obj3) with the checker's exact arithmetic."""
            obj1 = 0.0
            obj3 = 0.0
            for bi, p in cur.items():
                bd = self.blocks[bi]
                obj1 += max(0.0, p.exit_time - bd["due_date"])
                prefs = bd["bay_preferences"]
                obj3 += max(prefs) - prefs[p.bay_id]
            if n_bays >= 2:
                obj2 = math.floor(max(
                    abs(u[j1] * loads[j1] - u[j2] * loads[j2])
                    for j1 in range(n_bays) for j2 in range(j1 + 1, n_bays)))
            else:
                obj2 = 0.0
            return self.w1 * obj1 + self.w2 * obj2 + self.w3 * obj3, obj1, obj2, obj3

        def remove(bi: int) -> tuple[Placement, _Rec]:
            p = cur.pop(bi)
            rec = rec_of.pop(bi)
            state[p.bay_id].remove(rec)
            loads[p.bay_id] -= self.blocks[bi]["workload"]
            return p, rec

        def restore(p: Placement, rec: _Rec) -> None:
            state[p.bay_id].append(rec)
            loads[p.bay_id] += self.blocks[p.block_id]["workload"]
            cur[p.block_id] = p
            rec_of[p.block_id] = rec

        def top_random_mix(cands: list[tuple[float, int]], k: int) -> list[int]:
            """Half the worst offenders, half random among the rest."""
            cands.sort(reverse=True)
            n_top = max(1, k // 2)
            top = [bi for _, bi in cands[:n_top]]
            rest = [bi for _, bi in cands[n_top:]]
            return top + rng.sample(rest, min(len(rest), k - len(top)))

        # Unavoidable tardiness (release + proc > due) cannot be improved by
        # ANY placement -- exclude it from targeting so the roulette and the
        # tardy operator only chase the improvable part.
        min_tard_of = {
            bi: max(0, bd["release_time"] + bd["processing_time"] - bd["due_date"])
            for bi, bd in enumerate(self.blocks)
        }
        total_min_tard = float(sum(min_tard_of.values()))

        def select(op: str, k: int) -> list[int]:
            if op == "tardy":
                cands = [(cur[bi].exit_time - self.blocks[bi]["due_date"] - min_tard_of[bi], bi)
                         for bi in cur
                         if cur[bi].exit_time - self.blocks[bi]["due_date"] > min_tard_of[bi]]
                if not cands:
                    return select("random", k)
                # Small tardy neighbourhoods only: reinsertion quality degrades
                # sharply with size (diagnosed: big sets wreck the relocated
                # blockers' own tardiness and always get rejected).
                sel = top_random_mix(cands, min(k, 2))
                # Remove ALL blocks squatting on the top offender's earlier
                # window (capped): a single re-place never helps, but clearing
                # every bbox-overlapping blocker in [release, entry) drops the
                # tardiness to its minimum.
                extra: list[int] = []
                for bi in sel[:1]:
                    p = cur[bi]
                    my = rec_of[bi].bbox
                    r = self.blocks[bi]["release_time"]
                    n_taken = 0
                    for rec in state[p.bay_id]:
                        if n_taken >= 3:
                            break
                        if rec.bid == bi or rec.bid in extra:
                            continue
                        if rec.exit <= r or rec.entry >= p.entry_time:
                            continue  # does not occupy the earlier window
                        rb = rec.bbox
                        if not (my[2] <= rb[0] or rb[2] <= my[0] or
                                my[3] <= rb[1] or rb[3] <= my[1]):
                            extra.append(rec.bid)
                            n_taken += 1
                return sel + extra
            if op == "timewin":
                # Whole-time-window repack: on congested instances tardiness
                # is sequencing-bound (pulling one block forward pushes
                # another -- zero-sum), so remove EVERY block of the
                # offender's bay whose stay intersects the offender's window
                # and let the repair re-sequence them together.
                cands = [(cur[bi].exit_time - self.blocks[bi]["due_date"] - min_tard_of[bi], bi)
                         for bi in cur
                         if cur[bi].exit_time - self.blocks[bi]["due_date"] > min_tard_of[bi]]
                if not cands:
                    return select("random", k)
                cands.sort(reverse=True)
                _, off = cands[rng.randrange(min(3, len(cands)))]
                p_off = cur[off]
                r_off = self.blocks[off]["release_time"]
                members = [rec.bid for rec in state[p_off.bay_id]
                           if rec.exit > r_off and rec.entry < p_off.exit_time]
                if len(members) > 10:  # keep the offender (distance 0) + nearest stays
                    members.sort(key=lambda bi: abs(cur[bi].entry_time - p_off.entry_time))
                    members = members[:10]
                return members
            if op == "pref":
                cands = []
                for bi in cur:
                    prefs = self.blocks[bi]["bay_preferences"]
                    pen = max(prefs) - prefs[cur[bi].bay_id]
                    if pen > 0:
                        cands.append((float(pen), bi))
                if not cands:
                    return select("random", k)
                return top_random_mix(cands, k)
            if op == "balance":
                heavy = max(range(n_bays), key=lambda j: u[j] * loads[j])
                members = [rec.bid for rec in state[heavy]]
                if not members:
                    return select("random", k)
                return rng.sample(members, min(len(members), k))
            pool = list(cur.keys())
            return rng.sample(pool, min(len(pool), k))

        cur_obj, o1, o2, o3 = exact_terms()
        start_obj = cur_obj
        k_max = max(3, min(30, n // 8))
        n_iter = n_acc = 0

        # -- deterministic tardiness sweep first ------------------------------
        # For each improvable offender (worst first): remove it together with
        # the blocks squatting on its earlier window, reinsert offender-first,
        # keep on improvement.  Diagnosed on prob_4: these single-offender
        # repairs accept reliably, while large random neighbourhoods do not --
        # so harvest them deterministically and leave only the rarer
        # multi-block rearrangements to the stochastic loop below.
        for _ in range(3 if do_sweep else 0):
            improved_any = False
            offenders = sorted(
                ((max(0, cur[bi].exit_time - self.blocks[bi]["due_date"]) - min_tard_of[bi], bi)
                 for bi in cur), reverse=True)
            for imp, off in offenders:
                if imp <= 0 or time.time() >= deadline:
                    break
                n_iter += 1
                p_off = cur[off]
                my = rec_of[off].bbox
                r_off = self.blocks[off]["release_time"]
                victims = [off]
                for rec in state[p_off.bay_id]:
                    if len(victims) >= 4:
                        break
                    if rec.bid == off:
                        continue
                    if rec.exit <= r_off or rec.entry >= p_off.entry_time:
                        continue
                    rb = rec.bbox
                    if not (my[2] <= rb[0] or rb[2] <= my[0] or
                            my[3] <= rb[1] or rb[3] <= my[1]):
                        victims.append(rec.bid)
                saved = [remove(bi) for bi in victims]
                added = []
                for bi in victims:  # offender first, then its blockers
                    budget = max(0.002, min(0.5, (deadline - time.time()) / 8))
                    p = self._place_coexist(bi, state, loads, u, time.time() + budget)
                    if p is None:
                        p = self._fallback_place(bi, state)
                    rec_of[bi] = self._commit(p, state, loads)
                    cur[bi] = p
                    added.append(bi)
                new_obj, n1, n2, n3 = exact_terms()
                if new_obj < cur_obj:
                    cur_obj, o1, o2, o3 = new_obj, n1, n2, n3
                    n_acc += 1
                    improved_any = True
                else:
                    for bi in added:
                        remove(bi)
                    for p, rec in saved:
                        restore(p, rec)
            if not improved_any or time.time() >= deadline:
                break

        # -- stochastic destroy-repair loop -----------------------------------
        last_progress = time.time()  # sweep improvements count as progress
        window = max(deadline - t_call, 1e-9)
        best_loc_obj = cur_obj                 # best placements SEEN so far
        best_pl = list(cur.values())           # snapshot: SA may wander below it
        d_hist: list[float] = []               # recent positive deltas (SA scale)
        while time.time() < deadline:
            if stall_window is not None and time.time() - last_progress > stall_window:
                break  # plateaued: hand the remaining window back to the caller
            n_iter += 1
            # regime roulette: attack the currently dominant IMPROVABLE term
            T = self.w1 * max(0.0, o1 - total_min_tard)
            B = self.w2 * o2
            P = self.w3 * o3
            tot = T + B + P
            if tot <= 0 or rng.random() < 0.15:
                op = "random"
            else:
                x = rng.random() * tot
                op = "tardy" if x < T else ("balance" if x < T + B else "pref")
            if op == "tardy" and rng.random() < 0.25:
                op = "timewin"  # sequencing move instead of single-offender move

            # Small neighbourhoods iterate fast; occasionally go large.
            k = rng.randint(2, 6) if rng.random() < 0.7 else rng.randint(2, k_max)
            victims = list(dict.fromkeys(select(op, k)))
            if not victims:
                continue
            # old improvable tardiness, needed for the offender-first order
            old_imp_tard = {
                bi: max(0, cur[bi].exit_time - self.blocks[bi]["due_date"]) - min_tard_of[bi]
                for bi in victims
            }
            saved = [remove(bi) for bi in victims]

            # Repair mode: thorough (all bays, exact w2/w3 trade-off) only for
            # the operators that need cross-bay moves; tardy/random iterations
            # use the fast tardiness-first search so they stay cheap.
            thorough = op in ("balance", "pref")

            # Reinsert order: tardy iterations put offenders FIRST so they grab
            # the freed slot before their old blockers reclaim it; other
            # operators reinsert in plain EDD order.
            if op == "tardy":
                order = sorted(victims, key=lambda i: (-old_imp_tard[i],
                                                       self.blocks[i]["due_date"],
                                                       self.blocks[i]["processing_time"]))
            elif op == "timewin":
                # The whole point is a DIFFERENT sequence: noisy-EDD over the
                # removed set so ties and near-ties reshuffle every draw.
                dues = {i: self.blocks[i]["due_date"] for i in victims}
                span = float(max(dues.values()) - min(dues.values())) or 1.0
                order = sorted(victims,
                               key=lambda i: dues[i] + rng.uniform(-0.4, 0.4) * span)
            else:
                order = sorted(victims, key=lambda i: (self.blocks[i]["due_date"],
                                                       self.blocks[i]["processing_time"]))
            added: list[int] = []
            for j, bi in enumerate(order):
                p = None
                now = time.time()
                if now < deadline:
                    # one iteration may use at most ~1/4 of the remaining time
                    budget = max(0.002, min(0.5, (deadline - now) / (4 * (len(order) - j))))
                    p = self._place_coexist(bi, state, loads, u, now + budget,
                                            thorough=thorough, rng=rng)
                if p is None:
                    p = self._fallback_place(bi, state)
                rec_of[bi] = self._commit(p, state, loads)
                cur[bi] = p
                added.append(bi)

            new_obj, n1, n2, n3 = exact_terms()
            d = new_obj - cur_obj
            # Accept improvements; walk sideways occasionally (equal objective,
            # different configuration); with temp > 0 also accept worsenings
            # with the Metropolis probability -- the returned solution is the
            # BEST seen, so the caller never receives worse than its input.
            accept = d < 0 or (d == 0 and rng.random() < 0.2)
            if not accept and temp > 0.0 and d > 0:
                d_hist.append(d)
                if len(d_hist) > 64:
                    d_hist.pop(0)
                # MEDIAN delta as the temperature scale: the delta distribution
                # is heavy-tailed (one wrecked reinsertion can cost 100x the
                # typical move), and a mean-scaled T degenerates into a random
                # walk.  The median keeps huge worsenings near-impossible while
                # typical ones pass at the intended Metropolis rate.
                med = sorted(d_hist)[len(d_hist) // 2]
                frac = max(0.0, (deadline - time.time()) / window)  # cooling
                T_abs = temp * med * frac
                accept = T_abs > 0 and rng.random() < math.exp(-d / T_abs)
            if accept:
                if d < 0:
                    n_acc += 1
                cur_obj, o1, o2, o3 = new_obj, n1, n2, n3
                if cur_obj < best_loc_obj:
                    best_loc_obj = cur_obj
                    best_pl = list(cur.values())
                    last_progress = time.time()
            else:  # revert to the exact previous state (same _Rec objects)
                for bi in added:
                    remove(bi)
                for p, rec in saved:
                    restore(p, rec)

        if best_loc_obj < cur_obj:
            out, out_obj = best_pl, best_loc_obj  # SA wandered: return the best seen
        else:
            out, out_obj = list(cur.values()), cur_obj
        tag = f", SA temp={temp:.2f}" if temp > 0 else ""
        print(f"[ogc] Layer2 LNS    : {n_iter} iters, {n_acc} accepted{tag}, "
              f"exact obj {start_obj:.0f} -> {out_obj:.0f}  elapsed={self.elapsed():.2f}s")
        return out, n_acc, n_iter

    # ======================================================================
    # Layer 3: Gurobi timing MILP (decomposed subproblem)
    # ======================================================================
    #
    # Placement (bay, position, orientation) is FIXED; only entry/exit times
    # are re-optimised.  This decomposition is exact for the objective split:
    # the checker accumulates bay loads from the constant block["workload"]
    # (utils.py:1412), so obj2 and obj3 depend on the assignment only -- with
    # placements fixed, time moves obj1 (tardiness) alone, and any per-bay
    # tardiness reduction is a strict improvement of the total objective.
    # Bays have independent cranes, so each bay is an independent small MILP.
    #
    # Pairwise model (same conservative boundary rules as Layer-1's
    # _earliest_slot, so every MILP-feasible timing passes our own predicate;
    # the official check_feasibility still gates acceptance):
    #   c_ij = "i's crane sweep is obstructed while j is present"
    #        = check_entry(bay, [j], i)   (precomputed once per pair; bbox-
    #          disjoint pairs are skipped -- they can never interact)
    #   For every conflicting ordered pair (i mover, j static):
    #     ENTRY:  E_i <= E_j - 1   OR   E_i >= X_j      (one binary, big-M)
    #     EXIT :  X_i <= E_j       OR   X_i >= X_j + 1  (one binary, big-M)
    #   Same-layer collision needs no extra constraint: j>=k includes j==k,
    #   so spatially colliding pairs have BOTH c_ij and c_ji set, and the op
    #   constraints then forbid any temporal overlap.
    #
    # The incumbent times are MILP-feasible (they satisfy the same rules by
    # construction) and are supplied as a warm start, so the model is anytime:
    # whatever Gurobi finds within its slice is a certified-or-rejected gain.

    def _exact_obj(self, placements: list[Placement]) -> float:
        """Full objective of a placement list with the checker's arithmetic."""
        n_bays = len(self.bays)
        areas = [b.width * b.height for b in self.bays]
        avg_area = sum(areas) / n_bays
        u = [avg_area / a for a in areas]
        obj1 = obj3 = 0.0
        loads = [0.0] * n_bays
        for p in placements:
            bd = self.blocks[p.block_id]
            obj1 += max(0.0, p.exit_time - bd["due_date"])
            loads[p.bay_id] += bd["workload"]
            prefs = bd["bay_preferences"]
            obj3 += max(prefs) - prefs[p.bay_id]
        obj2 = math.floor(max(
            abs(u[j1] * loads[j1] - u[j2] * loads[j2])
            for j1 in range(n_bays) for j2 in range(j1 + 1, n_bays))) if n_bays >= 2 else 0.0
        return self.w1 * obj1 + self.w2 * obj2 + self.w3 * obj3

    # -- Layer 3a: bay-reassignment master ---------------------------------
    #
    # The timing MILP below PROVED (on the training set) that with placements
    # fixed our constructive timing is already optimal -- remaining gains need
    # blocks to MOVE.  This master chooses a bay per block:
    #   z[i,j] = 1  <=>  block i assigned to (fitting) bay j
    #   obj3 exact linear:  sum (s_max_i - pref_ij) z[i,j]
    #   obj2 exact epigraph:  I >= +-(u_j * load_j - u_k * load_k), load_j =
    #     sum workload_i z[i,j]  (floor() is monotone, so minimising the real
    #     value minimises the checker's floored value too)
    #   obj1 is schedule-dependent, so the MILP cannot see it -- it is guarded
    #   by a trust region (at most ~10% of blocks move) and by realising the
    #   assignment afterwards: moved blocks are removed and reinserted into
    #   their MILP bay via the Layer-1 search, and the candidate is kept only
    #   if the EXACT full objective (and then the official checker) improves.

    def milp_reassign(self, placements: list[Placement], deadline: float,
                      ) -> tuple[list[Placement], str, list[Placement] | None]:
        """One master-reassignment round.  Returns (certain, status, spec):
          certain / status "better" -- placements that strictly improve the
            exact objective (realised via rebuild or stepwise moves); with
            status "none", certain is just the input.
          spec -- optionally, a SPECULATIVE rebuild that realises the master
            assignment but still carries recoverable tardiness.  Its provable
            floor (w1 * unavoidable tardiness + its exact obj2/obj3) beats the
            incumbent, so it is worth part of the LNS budget; the caller must
            hedge it (time-boxed trial + official gate), never trust it.
        """
        if time.time() >= deadline or len(self.bays) < 2:
            return placements, "none", None
        if self.w2 <= 0 and self.w3 <= 0:
            return placements, "none", None  # the master can only improve obj2/obj3

        try:
            import gurobipy as gp
            from gurobipy import GRB
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
        except Exception as exc:
            print(f"[ogc] Layer3 master  : gurobi unavailable ({exc!r}) -- skipped")
            return placements, "none", None

        n_bays = len(self.bays)
        areas = [b.width * b.height for b in self.bays]
        avg_area = sum(areas) / n_bays
        u = [avg_area / a for a in areas]
        cur_bay = {p.block_id: p.bay_id for p in placements}
        n = len(placements)
        # If the window affords a full constrained rebuild (measured Layer-1
        # cost as the estimate), let the master reassign freely -- the rebuild
        # re-packs every bay from scratch, so big moves are safe.  Otherwise
        # moves are realised as local patches and need a trust region.
        l1_secs = getattr(self, "_l1_secs", None)
        rebuild_ok = (l1_secs is not None
                      and (deadline - time.time()) > 1.3 * l1_secs + 0.2)
        max_moves = n if rebuild_ok else max(2, math.ceil(0.10 * n))

        try:
            m = gp.Model(env=env)
            z: dict[tuple[int, int], object] = {}
            for p in placements:
                bd = self.blocks[p.block_id]
                fit_bays = [b.id for b in self.bays
                            if b.id == p.bay_id or self._fit_position(bd, b) is not None]
                for j in fit_bays:
                    z[(p.block_id, j)] = m.addVar(vtype=GRB.BINARY)
                m.addConstr(gp.quicksum(z[(p.block_id, j)] for j in fit_bays) == 1)
            load = {}
            for j in range(n_bays):
                load[j] = gp.quicksum(self.blocks[bi]["workload"] * v
                                      for (bi, jj), v in z.items() if jj == j)
            I = m.addVar(lb=0.0)
            for j1 in range(n_bays):
                for j2 in range(j1 + 1, n_bays):
                    m.addConstr(I >= u[j1] * load[j1] - u[j2] * load[j2])
                    m.addConstr(I >= u[j2] * load[j2] - u[j1] * load[j1])
            pref_pen = gp.quicksum(
                (max(self.blocks[bi]["bay_preferences"]) - self.blocks[bi]["bay_preferences"][j]) * v
                for (bi, j), v in z.items())
            m.addConstr(gp.quicksum(v for (bi, j), v in z.items()
                                    if j != cur_bay[bi]) <= max_moves)
            for (bi, j), v in z.items():   # warm start = stay put
                v.Start = 1.0 if j == cur_bay[bi] else 0.0
            m.setObjective(self.w2 * I + self.w3 * pref_pen, GRB.MINIMIZE)
            # MILP gets at most half the window (relative floor, no absolute
            # tuning): the other half belongs to realising the moves.
            m.Params.TimeLimit = max(0.05, min((deadline - time.time()) * 0.5,
                                               max(1.0, 0.05 * self.timelimit)))
            m.Params.Threads = 4
            m.optimize()
            if m.SolCount == 0:
                return placements, "none", None
            target = {bi: j for (bi, j), v in z.items() if v.X > 0.5}
        except Exception as exc:
            print(f"[ogc] Layer3 master  : MILP error ({exc!r})")
            return placements, "none", None
        finally:
            try:
                env.dispose()
            except Exception:
                pass

        moved = [bi for bi, j in target.items() if j != cur_bay[bi]]
        if not moved:
            return placements, "none", None
        spec: list[Placement] | None = None

        # -- Path A: full constrained rebuild ---------------------------------
        # Re-run the Layer-1 constructor with every block pinned to its MILP
        # bay.  Local patching (Path B) keeps the other blocks frozen, which
        # often denies the movers a tardiness-free landing; a rebuild re-packs
        # each bay from scratch and realises the assignment much more often
        # (measured: it reproduces the promised obj2/obj3 almost exactly, with
        # only a few units of tail tardiness left over).
        if rebuild_ok:
            base_obj = self._exact_obj(placements)
            cand = self.build_coexist(bay_of=target, deadline=deadline)
            new_obj = self._exact_obj(cand)
            if new_obj < base_obj:
                print(f"[ogc] Layer3 master  : rebuild with {len(moved)} move(s), exact obj "
                      f"{base_obj:.0f} -> {new_obj:.0f}  elapsed={self.elapsed():.2f}s")
                return cand, "better", None
            # Speculative branch: the rebuild's PROVABLE floor -- unavoidable
            # tardiness (no schedule can beat it) plus its exact obj2/obj3 --
            # beats the incumbent, and its excess tardiness is exactly the
            # kind the LNS tardiness sweep removes.  Kept ALONGSIDE the
            # stepwise realisation below: the certain gain is banked first,
            # the gamble gets a time-boxed LNS trial in the caller.
            total_min_tard = sum(
                max(0, bd["release_time"] + bd["processing_time"] - bd["due_date"])
                for bd in self.blocks)
            o23 = new_obj - self.w1 * sum(
                max(0.0, p.exit_time - self.blocks[p.block_id]["due_date"]) for p in cand)
            floor_obj = self.w1 * total_min_tard + o23
            if floor_obj < base_obj:
                print(f"[ogc] Layer3 master  : speculative rebuild kept (obj {new_obj:.0f}, "
                      f"provable floor {floor_obj:.0f} < incumbent {base_obj:.0f})  "
                      f"elapsed={self.elapsed():.2f}s")
                spec = cand
            # fall through to stepwise realisation of the same assignment

        # -- Path B: realise the assignment ONE MOVE AT A TIME as undoable STEPS, tracking
        # the exact full objective (obj1 side-effects included) after each
        # step, then undo in reverse order back to the best prefix.  This
        # keeps coordinated multi-moves (a swap may pass through a worse
        # intermediate) while never returning anything worse than the
        # incumbent -- realising all moves at once and judging once was
        # diagnosed to fail exactly like oversized LNS neighbourhoods.
        #
        # Each step is a list of relocations (bi, old_p, old_rec, new_p,
        # new_rec).  A plain step relocates just the mover into its MILP bay.
        # If that does not improve, an ENRICHED step is tried: the blocks
        # squatting on the mover's landing window in the target bay are pulled
        # out with it, the mover lands first, the blockers re-place freely --
        # the same window-clearing that made the LNS tardy operator work.
        # Undoing a step in reverse entry order restores the exact prior state.
        state: list[list[_Rec]] = [[] for _ in range(n_bays)]
        loads = [0.0] * n_bays
        cur: dict[int, Placement] = {}
        rec_of: dict[int, _Rec] = {}
        for p in placements:
            rec_of[p.block_id] = self._commit(p, state, loads)
            cur[p.block_id] = p

        def relocate(bi: int, only_bay: int | None, t_deadline: float):
            old_p, old_rec = cur[bi], rec_of[bi]
            state[old_p.bay_id].remove(old_rec)
            loads[old_p.bay_id] -= self.blocks[bi]["workload"]
            p = self._place_coexist(bi, state, loads, u, t_deadline,
                                    thorough=True, only_bay=only_bay)
            if p is None:
                p = self._fallback_place(bi, state, only_bay=only_bay)
            rec = self._commit(p, state, loads)
            cur[bi] = p
            rec_of[bi] = rec
            return (bi, old_p, old_rec, p, rec)

        def undo(entries) -> None:
            for bi, old_p, old_rec, new_p, new_rec in reversed(entries):
                state[new_p.bay_id].remove(new_rec)
                loads[new_p.bay_id] -= self.blocks[bi]["workload"]
                state[old_p.bay_id].append(old_rec)
                loads[old_p.bay_id] += self.blocks[bi]["workload"]
                cur[bi] = old_p
                rec_of[bi] = old_rec

        base_obj = self._exact_obj(placements)
        best_obj = base_obj
        steps: list[list] = []
        best_idx = 0
        n_tried = 0
        order = sorted(moved, key=lambda i: (self.blocks[i]["due_date"],
                                             self.blocks[i]["processing_time"]))
        for idx, bi in enumerate(order):
            now = time.time()
            if now >= deadline:
                break
            budget = max(0.002, min(0.5, (deadline - now) / (len(order) - idx)))
            n_tried += 1

            # -- plain step: relocate the mover alone -------------------------
            step = [relocate(bi, target[bi], now + budget)]
            obj_a = self._exact_obj(list(cur.values()))

            if obj_a >= best_obj and time.time() < deadline:
                # -- enriched step: clear the landing window ------------------
                landed_p, landed_rec = cur[bi], rec_of[bi]
                lb = landed_rec.bbox
                r_mover = self.blocks[bi]["release_time"]
                blockers = []
                for rec in state[target[bi]]:
                    if len(blockers) >= 3:
                        break
                    if rec.bid == bi:
                        continue
                    if rec.exit <= r_mover or rec.entry >= landed_p.entry_time:
                        continue  # not on the earlier window
                    rb = rec.bbox
                    if not (lb[2] <= rb[0] or rb[2] <= lb[0] or
                            lb[3] <= rb[1] or rb[3] <= lb[1]):
                        blockers.append(rec.bid)
                if blockers:
                    undo(step)
                    t_end = time.time() + budget
                    step = [relocate(bi, target[bi], t_end)]
                    for bj in blockers:  # blockers re-place freely, any bay
                        step.append(relocate(bj, None, t_end))
                    obj_b = self._exact_obj(list(cur.values()))
                    if obj_b >= obj_a:  # enrichment did not pay: redo plain
                        undo(step)
                        step = [relocate(bi, target[bi], time.time() + budget)]
                        obj_a = self._exact_obj(list(cur.values()))
                    else:
                        obj_a = obj_b

            steps.append(step)
            if obj_a < best_obj:
                best_obj = obj_a
                best_idx = len(steps)

        for step in reversed(steps[best_idx:]):
            undo(step)

        if best_idx == 0:
            print(f"[ogc] Layer3 master  : {n_tried}/{len(moved)} move(s) tried, "
                  f"no improving prefix -- keeping incumbent")
            return placements, "none", spec
        print(f"[ogc] Layer3 master  : kept {best_idx}/{len(moved)} move(s), exact obj "
              f"{base_obj:.0f} -> {best_obj:.0f}  elapsed={self.elapsed():.2f}s")
        return list(cur.values()), "better", spec

    # -- Layer 3b: per-bay timing MILP --------------------------------------

    def milp_refine_times(self, placements: list[Placement]) -> tuple[list[Placement], int]:
        """Per-bay entry/exit re-timing via Gurobi.  Returns (placements,
        number of bays improved); on any failure returns the input unchanged."""
        deadline = self.t_start + self.timelimit * MILP_SAFETY
        if self.w1 <= 0 or time.time() >= deadline:
            return placements, 0

        by_bay: dict[int, list[Placement]] = {}
        for p in placements:
            by_bay.setdefault(p.bay_id, []).append(p)

        def improvable_tard(ps: list[Placement]) -> float:
            s = 0.0
            for p in ps:
                bd = self.blocks[p.block_id]
                s += (max(0, p.exit_time - bd["due_date"])
                      - max(0, bd["release_time"] + bd["processing_time"] - bd["due_date"]))
            return s

        todo = [(improvable_tard(ps), bay_id) for bay_id, ps in by_bay.items()]
        todo = sorted(((imp, b) for imp, b in todo if imp > 0), reverse=True)
        if not todo:
            return placements, 0

        try:
            import gurobipy as gp
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
        except Exception as exc:
            print(f"[ogc] Layer3 MILP    : gurobi unavailable ({exc!r}) -- skipped")
            return placements, 0

        new_times: dict[int, tuple[int, int]] = {}
        n_improved = 0
        gain = 0.0
        for idx, (_, bay_id) in enumerate(todo):
            now = time.time()
            if now >= deadline:
                break
            # Even time split over the remaining bays (worst offenders first).
            slice_deadline = now + (deadline - now) / (len(todo) - idx)
            got = self._milp_bay_times(gp, env, by_bay[bay_id], bay_id, slice_deadline)
            if got is not None:
                times, g = got
                new_times.update(times)
                n_improved += 1
                gain += g

        try:
            env.dispose()
        except Exception:
            pass
        if not new_times:
            print(f"[ogc] Layer3 MILP    : no timing improvement  elapsed={self.elapsed():.2f}s")
            return placements, 0

        out = [Placement(p.block_id, p.bay_id, p.x, p.y, p.orient_idx,
                         *new_times[p.block_id]) if p.block_id in new_times else p
               for p in placements]
        print(f"[ogc] Layer3 MILP    : {n_improved}/{len(todo)} bay(s) improved, "
              f"tardiness -{gain:.0f}  elapsed={self.elapsed():.2f}s")
        return out, n_improved

    def _milp_bay_times(self, gp, env, ps: list[Placement], bay_id: int,
                        t_deadline: float) -> tuple[dict[int, tuple[int, int]], float] | None:
        """Optimal re-timing of one bay's fixed placements.  Returns
        ({block_id: (entry, exit)}, tardiness gain) on strict improvement,
        None otherwise (including timeouts and oversized models)."""
        GRB = gp.GRB
        n = len(ps)
        if n <= 1:
            return None

        # World bboxes + fast-predicate geometry for the pairwise conflict flags.
        recs = []
        for p in ps:
            bd = self.blocks[p.block_id]
            geo = self._make_geo(bd, p.orient_idx, p.x, p.y)
            lx0, ly0, lx1, ly1 = self._bbox(bd, p.orient_idx)
            recs.append((p, bd, geo, (p.x + lx0, p.y + ly0, p.x + lx1, p.y + ly1)))

        conf: list[tuple[int, int]] = []  # (mover, static): mover blocked while static present
        for i in range(n):
            if time.time() > t_deadline:
                return None  # hard budget: pair precompute is shapely-bound
            _, _, geo_i, bb_i = recs[i]
            for j in range(i + 1, n):
                _, _, geo_j, bb_j = recs[j]
                if (bb_i[2] <= bb_j[0] or bb_j[2] <= bb_i[0] or
                        bb_i[3] <= bb_j[1] or bb_j[3] <= bb_i[1]):
                    continue
                if self._flag(geo_i, geo_j):
                    conf.append((i, j))
                if self._flag(geo_j, geo_i):
                    conf.append((j, i))

        if 2 * len(conf) > 30000:
            return None  # model would dwarf the remaining budget

        cur_tard = sum(max(0, p.exit_time - bd["due_date"]) for p, bd, _, _ in recs)
        # Horizon: the incumbent fits below it (warm start stays feasible) and
        # a little slack lets non-tardy blocks step aside for tardy ones.
        max_p = max(bd["processing_time"] for _, bd, _, _ in recs)
        H = max(p.exit_time for p, _, _, _ in recs) + max_p

        try:
            m = gp.Model(env=env)
            E, X, T = {}, {}, {}
            for i, (p, bd, _, _) in enumerate(recs):
                E[i] = m.addVar(lb=bd["release_time"], ub=H, vtype=GRB.INTEGER)
                X[i] = m.addVar(lb=0, ub=H, vtype=GRB.INTEGER)
                T[i] = m.addVar(lb=0.0)
                m.addConstr(X[i] >= E[i] + bd["processing_time"])
                m.addConstr(T[i] >= X[i] - bd["due_date"])
                E[i].Start = p.entry_time
                X[i].Start = p.exit_time
            for i, j in conf:
                b1 = m.addVar(vtype=GRB.BINARY)
                m.addConstr(E[i] <= E[j] - 1 + H * b1)
                m.addConstr(E[i] >= X[j] - H * (1 - b1))
                b2 = m.addVar(vtype=GRB.BINARY)
                m.addConstr(X[i] <= E[j] + H * b2)
                m.addConstr(X[i] >= X[j] + 1 - H * (1 - b2))
            m.setObjective(gp.quicksum(T.values()), GRB.MINIMIZE)
            m.Params.TimeLimit = max(0.05, t_deadline - time.time())
            m.Params.Threads = 4          # server cap (<=4 cores)
            m.Params.MIPFocus = 1         # warm-started: hunt better incumbents
            m.optimize()
            if m.SolCount == 0 or m.ObjVal > cur_tard - 0.5:
                return None
            times = {recs[i][0].block_id: (int(round(E[i].X)), int(round(X[i].X)))
                     for i in range(n)}
            return times, cur_tard - m.ObjVal
        except Exception as exc:
            print(f"[ogc]   Layer3 bay {bay_id}: MILP error ({exc!r})")
            return None  # any Gurobi hiccup: keep the incumbent times

    # ======================================================================
    # Pipeline
    # ======================================================================

    def solve(self) -> dict:
        print(f"[ogc] Instance : {self.prob.get('name', '?')}  "
              f"bays={len(self.bays)}  blocks={len(self.blocks)}  timelimit={self.timelimit:.1f}s")

        # ---- Layer 0: feasible floor (validated by the OFFICIAL checker) ----
        placements = self.build_floor()
        best_placements = placements
        self.best_sol = {"operations": _build_operations(placements)}
        res = check_feasibility(self.prob, self.best_sol)
        if res["feasible"]:
            best_obj = res["objective"]
            print(f"[ogc] Layer0 floor   : FEASIBLE  obj={best_obj:.0f}  "
                  f"(obj1={res['obj1']:.1f} obj2={res['obj2']:.1f} obj3={res['obj3']:.1f})  "
                  f"elapsed={self.elapsed():.2f}s")
        else:
            best_obj = float("inf")
            print(f"[ogc] Layer0 floor   : INFEASIBLE stage={res['stage']} (unexpected)")
            for v in res["violations"][:5]:
                print(f"[ogc]   {v}")

        # ---- Layer 1: crane-aware coexistence -------------------------------
        # Contract: only replace best_sol when the official checker confirms
        # the candidate feasible AND strictly better.  Any failure keeps the
        # floor, so we can never regress below the feasibility guarantee.
        try:
            if self.time_left() > 0:
                _t1 = time.time()
                cand_placements = self.build_coexist()
                self._l1_secs = time.time() - _t1  # cost estimate for Layer-3a rebuilds
                cand = {"operations": _build_operations(cand_placements)}
                res1 = check_feasibility(self.prob, cand)
                if res1["feasible"] and res1["objective"] < best_obj:
                    self.best_sol = cand
                    best_obj = res1["objective"]
                    best_placements = cand_placements
                    print(f"[ogc] Layer1 coexist : FEASIBLE  obj={res1['objective']:.0f}  "
                          f"(obj1={res1['obj1']:.1f} obj2={res1['obj2']:.1f} obj3={res1['obj3']:.1f})  "
                          f"elapsed={self.elapsed():.2f}s")
                elif res1["feasible"]:
                    print(f"[ogc] Layer1 coexist : feasible but not better "
                          f"(obj={res1['objective']:.0f}) -- keeping floor")
                else:
                    print(f"[ogc] Layer1 coexist : INFEASIBLE stage={res1['stage']} "
                          f"-- keeping floor")
                    for v in res1["violations"][:3]:
                        print(f"[ogc]   {v}")
        except Exception as exc:  # defensive: floor survives any Layer-1 bug
            print(f"[ogc] Layer1 failed ({exc!r}) -- keeping floor")

        # ---- Layer 3a: bay-reassignment master (BEFORE the LNS) --------------
        # The master fixes assignment-level obj2/obj3 (measured headroom is
        # often several times the LNS's reach); running it here lets the whole
        # LNS budget then work on recovering any tardiness the moves induced.
        # Accept contract as always: exact-objective improvement inside, then
        # the official checker before the incumbent is replaced.
        lns_input = best_placements
        spec = None
        try:
            if self.time_left() > 0:
                m_deadline = min(time.time() + 0.08 * self.timelimit,
                                 self.t_start + self.timelimit * 0.78)
                r_placements, status, spec = self.milp_reassign(best_placements, m_deadline)
                if status == "better":
                    cand = {"operations": _build_operations(r_placements)}
                    res_m = check_feasibility(self.prob, cand)
                    if res_m["feasible"] and res_m["objective"] < best_obj:
                        self.best_sol = cand
                        best_obj = res_m["objective"]
                        best_placements = r_placements
                        lns_input = r_placements
                        print(f"[ogc] Layer3a accept: FEASIBLE  obj={res_m['objective']:.0f}  "
                              f"(obj1={res_m['obj1']:.1f} obj2={res_m['obj2']:.1f} obj3={res_m['obj3']:.1f})  "
                              f"elapsed={self.elapsed():.2f}s")
                    else:
                        why = "not better" if res_m["feasible"] else f"INFEASIBLE stage={res_m['stage']}"
                        print(f"[ogc] Layer3a accept: rejected ({why}) -- keeping incumbent")
        except Exception as exc:  # defensive: incumbent survives any master bug
            print(f"[ogc] Layer3a failed ({exc!r}) -- keeping incumbent")

        # ---- Layer 2: LNS improvement ----------------------------------------
        # Same contract: the LNS result replaces the incumbent only if the
        # official checker confirms it feasible AND strictly better.
        #
        # Hedged speculative trial: if the master produced a speculative
        # rebuild, it gets HALF the LNS window to recover its excess
        # tardiness.  If it has overtaken the incumbent by then, the LNS
        # continues on that branch; otherwise the remaining half runs on the
        # incumbent as usual -- the loss is bounded at half the LNS budget.
        spec_adopted = False
        try:
            if spec is not None and self.time_left() > 0:
                lns_deadline = self.t_start + self.timelimit * LNS_SAFETY
                mid = time.time() + 0.5 * (lns_deadline - time.time())
                # The trial runs in three segments with a linear-extrapolation
                # abort: if the measured recovery rate cannot close the gap to
                # the incumbent within the trial window, stop gambling NOW and
                # give the time back to the incumbent branch (measured on
                # prob_23/26: the trial otherwise burns its full half-window
                # on an arithmetically hopeless chase).  Overtaking ends the
                # trial early in the winning direction instead.
                spec_pl = spec
                for si in range(3):
                    now = time.time()
                    if now >= mid:
                        break
                    seg_deadline = now + (mid - now) / (3 - si)
                    o_before = self._exact_obj(spec_pl)
                    spec_pl, _, _ = self.lns_improve(
                        spec_pl, deadline=seg_deadline,
                        seed=0xC0FFEE + 7919 * si, stall_frac=0.25)
                    o_after = self._exact_obj(spec_pl)
                    if o_after < best_obj:
                        break  # overtaken -- no need to finish the trial
                    rate = (o_before - o_after) / max(time.time() - now, 1e-9)
                    if rate * (mid - time.time()) < o_after - best_obj:
                        break  # projected miss -- abort the gamble
                if self._exact_obj(spec_pl) < best_obj:
                    lns_input = spec_pl
                    spec_adopted = True
                    print(f"[ogc] Layer2 spec   : rebuild branch overtook the incumbent "
                          f"-- continuing on it")
                else:
                    print(f"[ogc] Layer2 spec   : rebuild branch did not overtake "
                          f"-- back to the incumbent")
        except Exception as exc:  # defensive: incumbent survives the gamble
            print(f"[ogc] Layer2 spec failed ({exc!r}) -- back to the incumbent")

        try:
            if self.time_left() > 0:
                # Round loop: each LNS round returns early once it plateaus
                # (stall_frac); the freed window funds perturbed-order
                # construction RESTARTS (adopted only when strictly better by
                # the exact objective) and fresh-seed LNS rounds.  This
                # attacks the two measured stall modes at once: the greedy
                # construction's own quality ceiling (prob_23/26 class) and
                # seed exhaustion in long windows (prob_4 class).
                lns_deadline = self.t_start + self.timelimit * LNS_SAFETY
                margin = 0.02 * self.timelimit
                cur = lns_input
                cur_ox = self._exact_obj(cur)
                progressed = spec_adopted
                base_seed = 0x5EED if spec_adopted else 0xC0FFEE
                round_i = 0
                n_hopeless = 0  # consecutive restarts far off the incumbent
                sweep_next = True  # sweep only fresh branches, not replays
                stall_n = 0  # consecutive rounds without strict improvement
                while time.time() < lns_deadline - margin:
                    # Temperature ladder: descent first; after each stalled
                    # round escalate the SA temperature so sequencing-bound
                    # tardiness (zero-sum shifts with no improving neighbour)
                    # can pass through temporarily worse re-sequencings.
                    pl, _, n_it = self.lns_improve(
                        cur, deadline=lns_deadline,
                        seed=base_seed + 7919 * round_i, stall_frac=0.25,
                        do_sweep=sweep_next,
                        temp=(0.0, 0.5, 1.0, 1.5)[min(stall_n, 3)])
                    sweep_next = False
                    round_i += 1
                    o = self._exact_obj(pl)
                    if o < cur_ox:
                        progressed = True
                        stall_n = 0
                    else:
                        stall_n += 1
                    cur, cur_ox = pl, o  # a round never returns worse than its input
                    if n_it == 0:
                        break  # nothing left to iterate on
                    remaining = lns_deadline - time.time()
                    if remaining <= margin:
                        break
                    # Plateaued with time to spare: perturbed restart if the
                    # window affords a construction (measured Layer-1 cost).
                    est = getattr(self, "_l1_secs", None)
                    if est is not None and n_hopeless < 3 and \
                            remaining > 1.6 * est + margin:
                        # Budget floor RELATIVE to the timelimit: candidate
                        # quality is wildly non-monotone in the budget when it
                        # sits barely above the per-block minimum (measured on
                        # prob_8: same seed gives 12412 or 2.0M depending on
                        # jitter), and two unlucky draws used to disable
                        # restarts for good.
                        cand = self.build_coexist(
                            deadline=time.time() + min(remaining * 0.5,
                                                       max(1.5 * est, 0.03 * self.timelimit)),
                            order_rng=random.Random(0xA11CE + round_i),
                            order_noise=(0.08, 0.25, 0.6)[round_i % 3])
                        co = self._exact_obj(cand)
                        if co < cur_ox:
                            cur, cur_ox = cand, co
                            progressed = True
                            n_hopeless = 0
                            stall_n = 0
                            sweep_next = True  # fresh branch: sweep it once
                            print(f"[ogc] Layer2 restart: perturbed construction adopted "
                                  f"(exact obj {co:.0f})  elapsed={self.elapsed():.2f}s")
                        elif co > 1.3 * cur_ox:
                            # Construction-scale quality is far off the
                            # incumbent (e.g. after a big master gain): two
                            # such misses in a row stop further restarts.
                            n_hopeless += 1
                lns_placements = cur
                if progressed:
                    cand = {"operations": _build_operations(lns_placements)}
                    res2 = check_feasibility(self.prob, cand)
                    if res2["feasible"] and res2["objective"] < best_obj:
                        self.best_sol = cand
                        best_obj = res2["objective"]
                        best_placements = lns_placements
                        print(f"[ogc] Layer2 accept : FEASIBLE  obj={res2['objective']:.0f}  "
                              f"(obj1={res2['obj1']:.1f} obj2={res2['obj2']:.1f} obj3={res2['obj3']:.1f})  "
                              f"elapsed={self.elapsed():.2f}s")
                    else:
                        why = "not better" if res2["feasible"] else f"INFEASIBLE stage={res2['stage']}"
                        print(f"[ogc] Layer2 accept : rejected ({why}) -- keeping incumbent")
        except Exception as exc:  # defensive: incumbent survives any Layer-2 bug
            print(f"[ogc] Layer2 failed ({exc!r}) -- keeping incumbent")

        # ---- Layer 3b: Gurobi timing MILP (fixed placements, per bay) --------
        # Same contract again: candidate replaces the incumbent only when the
        # official checker confirms it feasible AND strictly better.  With no
        # gurobipy / no license / no improvable tardiness this is a no-op.
        try:
            if self.time_left() > 0:
                m_placements, n_imp = self.milp_refine_times(best_placements)
                if n_imp > 0:
                    cand = {"operations": _build_operations(m_placements)}
                    res3 = check_feasibility(self.prob, cand)
                    if res3["feasible"] and res3["objective"] < best_obj:
                        self.best_sol = cand
                        best_obj = res3["objective"]
                        best_placements = m_placements
                        print(f"[ogc] Layer3b accept: FEASIBLE  obj={res3['objective']:.0f}  "
                              f"(obj1={res3['obj1']:.1f} obj2={res3['obj2']:.1f} obj3={res3['obj3']:.1f})  "
                              f"elapsed={self.elapsed():.2f}s")
                    else:
                        why = "not better" if res3["feasible"] else f"INFEASIBLE stage={res3['stage']}"
                        print(f"[ogc] Layer3b accept: rejected ({why}) -- keeping incumbent")
        except Exception as exc:  # defensive: incumbent survives any Layer-3 bug
            print(f"[ogc] Layer3b failed ({exc!r}) -- keeping incumbent")

        return self.best_sol


# ---------------------------------------------------------------------------
# Entry point (signature fixed by the competition)
# ---------------------------------------------------------------------------

def algorithm(prob_info: dict, timelimit: float = 60) -> dict:
    """Return a feasible solution dict for `prob_info` within `timelimit`.

    Never raises: any unexpected error still returns the feasible floor if it
    was built, because a crash scores -1 (same as infeasible) on the
    leaderboard.
    """
    solver = Solver(prob_info, timelimit)
    try:
        return solver.solve()
    except Exception as exc:  # pragma: no cover - defensive last resort
        print(f"[myalgorithm] solve() failed ({exc!r})")
        if solver.best_sol is not None:
            return solver.best_sol
        return {"operations": _build_operations(solver.build_floor())}
