import os
from argparse import Namespace
from enum import Enum


class SparseState(Enum):
    NOT_INITIALIZED = "not_initialized"
    OBSERVE = "observe"
    UPDATE = "update"
    UPDATE_AND_VALIDATE = "update_and_validate"
    UPDATE_AND_OBSERVE = "update_and_observe"
    UPDATE_AND_VALIDATE_AND_OBSERVE = "update_and_validate_and_observe"


_SPARSE_STATE = SparseState.NOT_INITIALIZED


def get_sparse_state() -> SparseState | None:
    """Get the current sparse state. HardCode to OBSERVER for now."""
    global _SPARSE_STATE

    if _SPARSE_STATE == SparseState.NOT_INITIALIZED:
        _SPARSE_STATE = None
        sparse_state_env = str(os.getenv("SPARSERL_STATE", "None"))
        for state in SparseState:
            if str(state.value).lower() == sparse_state_env.lower():
                _SPARSE_STATE = state
        assert _SPARSE_STATE != SparseState.NOT_INITIALIZED, "Invalid sparse state"

    return _SPARSE_STATE


def get_sparse_args() -> Namespace:
    """Get booleans that can drive external call sites."""
    sparse_state_obj = get_sparse_state()
    if sparse_state_obj is None:
        return Namespace(sparse_update=False, sparse_validate=False, sparse_observe=False)

    sparse_state = sparse_state_obj.value
    update = "update" in sparse_state
    validate = "validate" in sparse_state
    observe = "observe" in sparse_state

    if validate:
        assert update, "validate requires update"

    return Namespace(
        sparse_update=update,
        sparse_validate=validate,
        sparse_observe=observe,
    )
