"""CloseDrawerDisambigThree: 3-way drawer disambiguation.

Extends CloseDrawerDisambig from 2 candidates (left/right) to 3 candidates
(leftmost/middle/rightmost). Random-pick baselines drop from ~50% to ~33%,
making the language-conditioning gap easier to detect even with smaller n.

Discovery: finds the largest same-y group of TOP_DRAWERs (the main drawer
row of the layout) and takes the 3 leftmost-to-rightmost in that group.
target_index ∈ {0, 1, 2} for the 3 ordered drawers.

Reset: opens all 3 candidate drawers fully so the prompted-but-not-target
drawers stay un-closed (the disambig signal: succeed only when target
matches the prompt).
"""
import numpy as np

from robocasa.environments.kitchen.single_stage.kitchen_drawer import CloseDrawer
from robocasa.models.fixtures import FixtureType, fixture_is_type


ORDINALS_3 = ("leftmost", "middle", "rightmost")


class CloseDrawerDisambigThree(CloseDrawer):
    def __init__(self, target_index=0, separation_min=0.4,
                 open_all=False, randomize_target=False, *args, **kwargs):
        """target_index: which of the 3 ordered drawers the prompt refers to.
        open_all: if True, open ALL 3 candidate drawers at reset. If False
        (default), open only the target drawer — matches CloseDrawer policy
        training distribution (one open drawer in view), keeps oracle SR high.
        randomize_target: if True, target_index is re-sampled uniformly each
        reset. Used during MimicGen data generation so demos cover all 3
        ordinals; ignored at eval time."""
        assert target_index in (0, 1, 2), target_index
        self.target_index = int(target_index)
        self._separation_min = separation_min
        self.open_all = open_all
        self.randomize_target = randomize_target
        self._candidate_drawers = None
        self._target_drawer = None
        super().__init__(*args, **kwargs)

    def _pick_three_drawers(self):
        """Return the 3 leftmost-to-rightmost top drawers in the layout's
        largest same-y group, or None if fewer than 3 candidates."""
        drawers = []
        for n, f in self.fixtures.items():
            if fixture_is_type(f, FixtureType.TOP_DRAWER):
                drawers.append((n, f))
        if len(drawers) < 3:
            return None
        by_y = {}
        for n, f in drawers:
            ykey = round(f.pos[1] / 0.3) * 0.3
            by_y.setdefault(ykey, []).append((n, f))
        best_group = max(by_y.values(), key=len)
        if len(best_group) < 3:
            return None
        best_group.sort(key=lambda nf: nf[1].pos[0])
        # Check overall extent is at least separation_min
        if best_group[-1][1].pos[0] - best_group[0][1].pos[0] < self._separation_min:
            return None
        # If the group has more than 3, take leftmost / median / rightmost
        if len(best_group) == 3:
            picked = best_group
        else:
            picked = [best_group[0], best_group[len(best_group) // 2], best_group[-1]]
        return picked

    def _reset_internal(self):
        super()._reset_internal()
        picked = self._pick_three_drawers()
        if picked is None:
            raise RuntimeError(
                "CloseDrawerDisambigThree: layout "
                f"{getattr(self, 'layout_and_style_ids', '?')} has no "
                f"3 same-row TOP_DRAWERs spanning >= {self._separation_min} m.")
        self._candidate_drawers = [f for _, f in picked]
        self._candidate_drawer_names = [n for n, _ in picked]
        if self.randomize_target:
            self.target_index = int(self.rng.integers(3))
        self._target_drawer = self._candidate_drawers[self.target_index]
        # Make `self.drawer` (used by base CloseDrawer + MimicGen MG_CloseDrawer
        # interface) point at the target candidate, so subtask waypoints
        # follow the prompted drawer.
        self.drawer = self._target_drawer
        if self.open_all:
            for f in self._candidate_drawers:
                f.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        else:
            # match CloseDrawer training distribution: only target is open
            self._target_drawer.set_door_state(
                min=0.90, max=1.0, env=self, rng=self.rng)
        self.sim.forward()

    def _check_success(self):
        if self._target_drawer is None:
            return False
        door_state = self._target_drawer.get_door_state(env=self)
        for joint_p in door_state.values():
            if joint_p > 0.05:
                return False
        return True

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"close the {ORDINALS_3[self.target_index]} drawer"
        return ep_meta

    def compute_target_oracle_candidates(self):
        """Collision-free placements for the target drawer. Mirrors
        CloseDrawerDisambig.compute_target_oracle_candidates."""
        d = self._target_drawer
        if d is None:
            raise RuntimeError("compute_target_oracle called before reset")
        x_ofs = (d.width / 2) + 0.3
        cands = []
        for sign in (-1, +1):
            pos, ori = self.compute_robot_base_placement_pose(
                ref_fixture=d, offset=(sign * x_ofs, -0.23))
            test, _ = self.compute_robot_base_placement_pose(
                ref_fixture=d, offset=(sign * (x_ofs + 0.3), -0.23))
            if not self.check_fxtr_contact(test) and \
               not self.check_sidewall_contact(test):
                cands.append(np.array([pos[0], pos[1], ori[2]], dtype=np.float64))
        if not cands:
            raise RuntimeError(
                f"no collision-free placement for drawer index {self.target_index}")
        return cands

    def get_drawer_states(self):
        if not self._candidate_drawers:
            return {}
        return {
            ORDINALS_3[i]: max(self._candidate_drawers[i].get_door_state(env=self).values())
            for i in range(3)
        }


class CloseDrawerDisambigThreeRandTarget(CloseDrawerDisambigThree):
    """Train-time variant: target_index is uniformly resampled each reset and
    open_all defaults to False, matching the CloseDrawer policy's training
    distribution. Used by MimicGen retarget so generated demos cover all 3
    candidate drawers."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("randomize_target", True)
        kwargs.setdefault("open_all", False)
        super().__init__(*args, **kwargs)
