from utils import *
from ray import *
from cli import render

#----BIRD----
sphere = Material(vec([0,0,0]))
bird_blue = Material(vec([0.1,0.2,0.9]))
beak_yellow = Material(vec([1.0, 0.72, 0.21]))
eyes = Material(vec([0,0,0]))

#---CAKE----
cake = Material(vec([0.85, 0.77, 0.549]))
cake_strawberry = Material(vec([0.411, 0.078, 0.023]))
vanilla = Material(vec([0.96,0.94,0.85]))
strawberry = Material(vec([0.988, 0.27, 0.1882]))

#---BASKET + BREAD----
brown = Material(vec([0.6, 0.3, 0]), k_m=0.4, k_s=0.3)

#---GRAPE---
magenta = Material(vec([0.4, 0.2, 0.5]), k_m=0.3)
green = Material(vec([0,0.8,0.1]))

#---CUPS----
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
magenta = Material(vec([0.4, 0.2, 0.5]), k_m=0.5)

#-----GENERAL----
plate_reflect = Material(vec([0.5, 0.5, 0.5]), k_m=0.8)





bird_blue_night = Material(vec([0.03, 0.07, 0.30]))
beak_yellow_night = Material(vec([0.35, 0.26, 0.07]))
eyes_night = Material(vec([0.0, 0.0, 0.0]))
cake_night = Material(vec([0.22, 0.20, 0.14]))
cake_strawberry_night = Material(vec([0.11, 0.02, 0.006]))
vanilla_night = Material(vec([0.22, 0.21, 0.19]))
strawberry_night = Material(vec([0.25, 0.07, 0.05]))
brown_night = Material(vec([0.12, 0.06, 0.0]), k_m=0.4)
magenta_night = Material(vec([0.10, 0.05, 0.13]), k_m=0.3)
green_night = Material(vec([0.0, 0.18, 0.03]))
blue_night    = Material(vec([0.06, 0.06, 0.15]), k_m=0.5)
magenta_night = Material(vec([0.10, 0.05, 0.13]), k_m=0.5)
green_night   = Material(vec([0.0, 0.14, 0.04]), k_m=0.5)
plate_reflect_night = Material(vec([0.12, 0.12, 0.14]), k_m=0.8)
