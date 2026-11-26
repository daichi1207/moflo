"""CloseDrawerDisambig --- a multi-fixture CloseDrawer variant for the
within-task disambiguation rollout-SR experiment.

The base RoboCasa ``CloseDrawer`` env registers a single ``TOP_DRAWER`` as
``self.drawer``, opens only that one, and ``_check_success`` checks only that
drawer's joint. That makes it impossible to test the question "given the
instruction, does the robot close the *prompted* drawer?" --- the un-prompted
drawer is never even open.

``CloseDrawerDisambig`` fixes both:
  1. At ``_reset_internal``, after the base reset, it discovers the left- and
     right-most top drawers in the layout's main row and opens *both*.
  2. It accepts a ``target_side`` ("left" / "right") and overrides
     ``_check_success`` to check only the prompted drawer's joint state.
  3. ``get_ep_meta`` reports lang = "close the {target_side} drawer".

The eval harness teleports the robot to the placement method's predicted pose,
so the env's own robot-placement logic (around ``self.drawer``) is irrelevant
here; we do not override it.

Discovery logic mirrors ``moflo/eval/disambiguation/probe_disambig_scoring.py``'s
``_pick_left_right_drawers`` so the (left, right) pair is identical to the one
the scoring-only probe uses.
"""
from robocasa.environments.kitchen.single_stage.kitchen_drawer import CloseDrawer
from robocasa.models.fixtures import FixtureType, fixture_is_type


class CloseDrawerDisambig(CloseDrawer):
    def __init__(self, target_side="left", separation_min=1.0, *args, **kwargs):
        assert target_side in ("left", "right"), target_side
        self.target_side = target_side
        self._separation_min = separation_min
        self._left_drawer = None
        self._right_drawer = None
        self._target_drawer = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Left/right top-drawer discovery (mirrors probe_disambig_scoring).
    # ------------------------------------------------------------------ #
    def _pick_left_right_drawers(self):
        drawers = []
        for n, f in self.fixtures.items():
            if fixture_is_type(f, FixtureType.TOP_DRAWER):
                drawers.append((n, f))
        if len(drawers) < 2:
            return None
        # Group by y (within 0.3 m); the largest group is the main row.
        by_y = {}
        for n, f in drawers:
            ykey = round(f.pos[1] / 0.3) * 0.3
            by_y.setdefault(ykey, []).append((n, f))
        best_group = max(by_y.values(), key=len)
        if len(best_group) < 2:
            return None
        best_group.sort(key=lambda nf: nf[1].pos[0])
        left = best_group[0]
        right = best_group[-1]
        if right[1].pos[0] - left[1].pos[0] < self._separation_min:
            return None
        return left, right

    # ------------------------------------------------------------------ #
    # Reset: open BOTH candidate drawers, pin the prompted one as target.
    # ------------------------------------------------------------------ #
    def _reset_internal(self):
        # Base reset: opens self.drawer (some TOP_DRAWER) + places objects.
        super()._reset_internal()

        pair = self._pick_left_right_drawers()
        if pair is None:
            raise RuntimeError(
                "CloseDrawerDisambig: layout "
                f"{getattr(self, 'layout_and_style_ids', '?')} has no "
                "left/right top-drawer pair >= "
                f"{self._separation_min} m apart."
            )
        (lname, lfxtr), (rname, rfxtr) = pair
        self._left_drawer = lfxtr
        self._right_drawer = rfxtr
        self._left_drawer_name = lname
        self._right_drawer_name = rname

        # Open both candidate drawers fully. (Done after super() so the base
        # object-placement step does not drop objects through an open drawer
        # we created --- the base only places an object in self.drawer.)
        lfxtr.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        rfxtr.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        self.sim.forward()

        self._target_drawer = (
            lfxtr if self.target_side == "left" else rfxtr
        )

    # ------------------------------------------------------------------ #
    # Success: only the prompted drawer must be closed.
    # ------------------------------------------------------------------ #
    def _check_success(self):
        if self._target_drawer is None:
            return False
        door_state = self._target_drawer.get_door_state(env=self)
        for joint_p in door_state.values():
            if joint_p > 0.05:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Language: the disambiguating instruction.
    # ------------------------------------------------------------------ #
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"close the {self.target_side} drawer"
        return ep_meta

    # ------------------------------------------------------------------ #
    # Oracle base pose for the target drawer.
    # Mirrors ManipulateDrawer._load_model: try a left-of-drawer and a
    # right-of-drawer placement, collision/sidewall-check each, keep the
    # valid one(s). The probe's "default to right side" shortcut placed
    # the robot in unreachable space for layout-rightmost drawers.
    # ------------------------------------------------------------------ #
    def compute_target_oracle_candidates(self):
        """Return ALL collision-free base placements for the target drawer,
        as a list of (x, y, theta) arrays. Mirrors ManipulateDrawer._load_model:
        a left-of-drawer and a right-of-drawer placement, each collision- and
        sidewall-checked. A drawer typically yields one or two candidates.

        The collision check guarantees *reachability* but not *policy success*:
        empirically, only the placement facing the layout interior is reliable.
        The caller (smoke test / rollout harness) determines which candidate is
        the policy-favorable one --- the oracle is "best reachable placement"."""
        import numpy as np
        d = self._target_drawer
        if d is None:
            raise RuntimeError("compute_target_oracle called before reset")
        x_ofs = (d.width / 2) + 0.3
        cands = []

        pos_l, ori_l = self.compute_robot_base_placement_pose(
            ref_fixture=d, offset=(-x_ofs, -0.23))
        test_l, _ = self.compute_robot_base_placement_pose(
            ref_fixture=d, offset=(-x_ofs - 0.3, -0.23))
        if not self.check_fxtr_contact(test_l) and \
           not self.check_sidewall_contact(test_l):
            cands.append(np.array([pos_l[0], pos_l[1], ori_l[2]], dtype=np.float64))

        pos_r, ori_r = self.compute_robot_base_placement_pose(
            ref_fixture=d, offset=(x_ofs, -0.23))
        test_r, _ = self.compute_robot_base_placement_pose(
            ref_fixture=d, offset=(x_ofs + 0.3, -0.23))
        if not self.check_fxtr_contact(test_r) and \
           not self.check_sidewall_contact(test_r):
            cands.append(np.array([pos_r[0], pos_r[1], ori_r[2]], dtype=np.float64))

        if not cands:
            raise RuntimeError(
                f"no collision-free placement for {self.target_side} drawer")
        return cands

    def compute_target_oracle(self):
        """Backward-compatible: return the first collision-free placement.
        Prefer compute_target_oracle_candidates() + policy calibration."""
        return self.compute_target_oracle_candidates()[0]

    # ------------------------------------------------------------------ #
    # Convenience: per-drawer open percentage, for the smoke test + harness.
    # ------------------------------------------------------------------ #
    def get_drawer_states(self):
        out = {}
        if self._left_drawer is not None:
            out["left"] = max(self._left_drawer.get_door_state(env=self).values())
        if self._right_drawer is not None:
            out["right"] = max(self._right_drawer.get_door_state(env=self).values())
        return out
