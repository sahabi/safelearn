import os
a = 3
b = 300
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( a, b))