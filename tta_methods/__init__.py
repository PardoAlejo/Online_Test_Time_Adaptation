from tta_methods.basic import Basic_Wrapper
from tta_methods.tent import Tent
from tta_methods.eata import EATA
from tta_methods.cotta import CoTTA
from tta_methods.adabn import AdaBn
from tta_methods.shot_im import SHOTIM
from tta_methods.shot import SHOT
from tta_methods.bn_adaptation import BN_Adaptation
from tta_methods.sar import SAR

_all_methods = {
    'basic': Basic_Wrapper,
    'tent': Tent,
    'eta': EATA,
    'eata': EATA,
    'cotta':CoTTA,
    'adabn': AdaBn,
    'shotim': SHOTIM,
    'shot': SHOT,
    'bn_adaptation': BN_Adaptation,
    'sar': SAR,
}