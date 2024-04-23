from cut_detector.utils.mb_support import detection

from mbpkg.better_detector import Detector

class Detectors:
    cur_log = Detector(detection.cur_log)
    lapgau = Detector(detection.lapgau)
    log2_wider = Detector(detection.log2_wider)
    rshift_log = Detector(detection.rshift_log)
    cur_dog = Detector(detection.cur_dog)
    diffgau = Detector(detection.diffgau)
    cur_doh = Detector(detection.cur_doh)
    hessian = Detector(detection.hessian)


class StrDetectors:
    cur_log = Detector("cur_log")
    lapgau = Detector("lapgau")
    log2_wider = Detector("log2_wider")
    rshift_log = Detector("rshift_log")
    cur_dog = Detector("cur_dog")
    diffgau = Detector("diffgau")
    cur_doh = Detector("cur_doh")
    hessian = Detector("hessian")

    

