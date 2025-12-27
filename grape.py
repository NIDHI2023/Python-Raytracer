from utils import *
from ray import *
from cli import render
from PIL import Image

magenta = Material(vec([0.4, 0.2, 0.5]), k_m=0.3)
green = Material(vec([0,0.8,0.1]))
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)



g1 = CSG(SphereTextured(vec([0.4,0,0]), 0.1, magenta))
g2 = CSG(SphereTextured(vec([0.5,0.1,0]), 0.1, magenta))
g3 = CSG(SphereTextured(vec([0.4,0.15,0]), 0.1, magenta))
g4 = CSG(SphereTextured(vec([0.4,0.25,0]), 0.1, magenta))
g5 = CSG(SphereTextured(vec([0.5,0.2,0]), 0.1, magenta))
g6 = CSG(SphereTextured(vec([0.6,0.15,0]), 0.1, magenta))
stem = Cylinder(vec([0.5,0.2,0]), 0.02, 0.02, vec([0.15,0.2,0]), green)

intersect_csg = CSG()
intersect_csg.add_children(g1,g2,"u")

test = CSG()
test.add_children(intersect_csg,g3,"u")

grape = CSG()
grape.add_children(test,g4,"u")

grape1 = CSG()
grape1.add_children(grape,g5,"u")

grape2 = CSG()
grape2.add_children(grape1,g6,"u")

final_grape = CSG()
final_grape.add_children(grape2, stem, "u")


scene = Scene([
    final_grape,
    Sphere(vec([0,-40,0]), 39.5, gray),
])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)


render(camera, scene, lights)