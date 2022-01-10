# from .house import House
# from .bbox import BBox
# from .panorama import Panorama

import logging
import coloredlogs
logger = logging.getLogger('panotools_configs')
coloredlogs.install(level="INFO",
                    logger=logger,
                    fmt='%(name)s, %(message)s')
logging.root.setLevel(logging.INFO)
