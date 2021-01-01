from django.apps import AppConfig
import sys, os

class FastbertConfig(AppConfig):
    name = 'fastbert'
    print("In App")
    ROOT_DIR = os.path.abspath("..\\")
    print(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN-master\\"))
    import violence
    ml = violence.MLmodel()