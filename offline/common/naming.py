from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MoeTag:
    layer: int
    kind: str  # "router" | "fc1" | "fc2"
    expert_id: int | None


_MOE_ROUTER_RE = re.compile(r"\.decoder\.layers\.(?P<layer>\d+)\.mlp\.router\.weight$")
_MOE_FC1_RE = re.compile(
    r"\.decoder\.layers\.(?P<layer>\d+)\.mlp\.experts\.linear_fc1\.weight(?P<expert>\d+)$"
)
_MOE_FC2_RE = re.compile(
    r"\.decoder\.layers\.(?P<layer>\d+)\.mlp\.experts\.linear_fc2\.weight(?P<expert>\d+)$"
)
_LAYER_RE = re.compile(r"\.decoder\.layers\.(?P<layer>\d+)\.")


def parse_moe_tag(param_name: str) -> MoeTag | None:
    m = _MOE_ROUTER_RE.search(param_name)
    if m:
        return MoeTag(layer=int(m.group("layer")), kind="router", expert_id=None)
    m = _MOE_FC1_RE.search(param_name)
    if m:
        return MoeTag(layer=int(m.group("layer")), kind="fc1", expert_id=int(m.group("expert")))
    m = _MOE_FC2_RE.search(param_name)
    if m:
        return MoeTag(layer=int(m.group("layer")), kind="fc2", expert_id=int(m.group("expert")))
    return None


def fix_moe_expert_param_name(param_name: str, *, ep_rank: int, experts_per_ep_rank: int) -> str:
    """Rewrite MoE expert-id suffix to global expert id across EP ranks."""
    if experts_per_ep_rank <= 0 or ep_rank < 0:
        return param_name
    tag = parse_moe_tag(param_name)
    if tag is None or tag.expert_id is None:
        return param_name
    global_expert = int(ep_rank) * int(experts_per_ep_rank) + int(tag.expert_id)
    return re.sub(r"(linear_fc[12]\.weight)\d+$", rf"\g<1>{global_expert}", param_name)


def classify_param_group(param_name: str) -> str:
    tag = parse_moe_tag(param_name)
    if tag is not None:
        if tag.kind == "router":
            return "moe/router"
        return f"moe/{tag.kind}"
    if ".attention." in param_name:
        return "attention"
    if ".mlp." in param_name:
        return "mlp(non-moe)"
    if "embedding" in param_name:
        return "embedding"
    if "layernorm" in param_name.lower() or "ln_" in param_name.lower():
        return "layernorm"
    if "output_layer" in param_name:
        return "output"
    return "other"


def parse_decoder_layer_id(param_name: str) -> int | None:
    m = _LAYER_RE.search(param_name)
    if not m:
        return None
    try:
        return int(m.group("layer"))
    except (TypeError, ValueError):
        return None
