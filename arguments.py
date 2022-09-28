import argparse


class Arguments:
    def __init__(self, stage='demo'):
        self._parser = argparse.ArgumentParser(description='Arguments for SCARP Demo')
        self.add_common_args()

    def add_common_args(self):
        ### data related
        self._parser.add_argument('--class_choice', type=str, default='plane', help='plane|car|chair|watercraft|bottle|bowl|mug|can|basket')
        self._parser.add_argument('--demo_dataset_path', type=str,default='./demo_data', help='Path for the Demo Data')

        
        ### TreeGAN architecture related
        self._parser.add_argument('--DEGREE', type=int, default=[1,  2,   2,   2,   2,   2,   32], nargs='+', help='Upsample degrees for generator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[128, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
        self._parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
        self._parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))
        
        ### others
        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        self._parser.add_argument('--ckpt_load', type=str, default=None, help='Checkpoint name to load. (default:None)')
    
    def parser(self):
        return self._parser
    


    
   


