
import os
from natsort import natsorted
from enum import Enum

class ListDirPolicyEnum(Enum):
    ALL = 0
    DIRS_ONLY = 1
    FILES_ONLY = 2

def list_dir(path, policy=ListDirPolicyEnum.ALL):
    filter = None
    if policy == ListDirPolicyEnum.ALL:
        filter = lambda x: True
    elif policy == ListDirPolicyEnum.DIRS_ONLY:
        filter =  lambda x: os.path.isdir(x)
    elif policy == ListDirPolicyEnum.FILES_ONLY:
        filter = lambda x: os.path.isfile(x)

    if not filter:
        filter = lambda x: True

    return natsorted([os.path.join(path, f) for f in os.listdir(path) 
            if filter(os.path.join(path, f))
    ])

