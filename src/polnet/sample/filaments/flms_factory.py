from .fiber_unit import FiberUnitSDimer, MTUnit
from .flms_gen import FlmsParamGen, HxParamGenBranched
from .flms_network import NetHelixFiber, NetHelixFiberB

class FlmsFactory:

    @classmethod
    def create(cls, hx_type: str, params: dict, v_size: float):
        """Returns (fiber_unit, param_gen, NetworkClass, net_kwargs)."""
        if hx_type == "mt":
            fiber_unit = MTUnit(
                sph_rad=params["HX_MMER_RAD"],
                mt_rad=params["MT_RAD"],
                n_units=int(params["MT_NUNITS"]),
                v_size=v_size,
            )
            param_gen = FlmsParamGen()
            net_cls = NetHelixFiber
            net_kwargs = dict(
                unit_diam=(params["MT_RAD"] + 0.5 * params["HX_MMER_RAD"]) * 2.4,
            )
        elif hx_type == "actin":
            fiber_unit = FiberUnitSDimer(
                sph_rad=params["HX_MMER_RAD"],
                v_size=v_size,
            )
            param_gen = HxParamGenBranched()
            net_cls = NetHelixFiberB
            net_kwargs = dict(
                b_prop=params["A_BPROP"],
                max_p_branch=int(params["A_MAX_P_BRANCH"]),
            )
        else:
            raise ValueError(f"Unsupported helix type: {hx_type}")

        return fiber_unit, param_gen, net_cls, net_kwargs