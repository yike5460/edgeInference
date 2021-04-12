import gluoncv as gcv
from gluoncv.utils import export_block

import glob

net = gcv.model_zoo.get_model('yolo3_mobilenet1.0_voc', pretrained=True)

export_block('yolo3_mobilenet1.0_voc', net, preprocess=True, layout='HWC')
print('Done.')


print(glob.glob('*.json') + glob.glob('*.params'))