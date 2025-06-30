import sys
import os

from rdd.RDD.utils.misc import read_config
from rdd.RDD.RDD import build
from rdd.RDD.RDD_helper import RDD_helper
from LiftFeat.models.liftfeat_wrapper import LiftFeat


liftfeat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LiftFeat'))
if liftfeat_path not in sys.path:
    sys.path.insert(0, liftfeat_path)

def setupRDD():
    RDD_model = build(weights='./downloads/RDD-v2.pth', config=read_config('./rdd/configs/default.yaml'))
    RDD_model.eval()
    RDD = RDD_helper(RDD_model)
    return RDD

def setupLiftFeat():
    liftfeat = LiftFeat(detect_threshold=0.05)    
    return liftfeat

if __name__ == '__main__':

    # Models instances
    # RDD = setupRDD()
    LF = setupLiftFeat()

