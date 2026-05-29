from .sasrec   import SASRec
from .tisasrec import TiSASRec
from .cl4srec  import CL4SRec
from .bsarec   import BSARec
from .saferec  import SAFERec
from .mbstr    import MBSTRec


def build_model(cfg, n_items: int):
    name = cfg.name.lower()
    if name == "sasrec":
        return SASRec(cfg, n_items)
    elif name == "tisasrec":
        return TiSASRec(cfg, n_items)
    elif name == "cl4srec":
        return CL4SRec(cfg, n_items)
    elif name == "bsarec":
        return BSARec(cfg, n_items)
    elif name == "saferec":
        return SAFERec(cfg, n_items)
    elif name == "mbstr":
        return MBSTRec(cfg, n_items)
    else:
        raise ValueError(f"Unknown model: {cfg.name}")
