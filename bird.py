from utils import *
from ray import *
from cli import render
from materials import *
from PIL import Image


gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

head = Sphere(vec([0.5,0.7,0.4]), 0.2, bird_blue)
body = Ellipsoid(vec([0,0.1,0]),vec([.5,0.125,0.125]),vec([0.125,0.5,0.125]),vec([0.125,0.125,0.5]),bird_blue)
eye_left = Sphere(vec([0.55,0.8,0.55]), 0.04, eyes)
eye_right = Sphere(vec([0.67,0.77,0.4]), 0.04, eyes)
beak = Cylinder(vec([0.65, 0.7, 0.55]), 0.08, 0, vec([0.1, 0, 0.1]), material=beak_yellow)
wing = Cylinder(vec([0.2, 0.4, 0.6]), 0.2, 0, vec([-1.5, -1.0, -1.3]), material=bird_blue)
tail = Cylinder(vec([-0.4, -0.1, -0.4]), 0.3, 0, vec([-1.3, -1.0, -1.3]), material=bird_blue)
left_leg = Cylinder(vec([-0.2,-0.2,0]), 0.02, 0.02, vec([0,-0.5,0]), eyes)
right_leg = Cylinder(vec([0.2,-0.2,0]), 0.02, 0.02, vec([0,-0.4,0]), eyes)

bird1 = CSG()
bird1.add_children(head, eye_left, "u")

bird2 = CSG()
bird2.add_children(bird1,beak, "u")

birdface = CSG()
birdface.add_children(bird2, eye_right, "u")

bird_body_top = CSG()
bird_body_top.add_children(birdface, body, "u")

bird_body_wing = CSG()
bird_body_wing.add_children(bird_body_top, wing, "u")

bird_body = CSG()
bird_body.add_children(bird_body_wing, tail, "u")

bird_almost = CSG()
bird_almost.add_children(bird_body, left_leg, "u")

bird_final = CSG()
bird_final.add_children(bird_almost, right_leg, "u")

scene = Scene([
    bird_final,
    Sphere(vec([0,-40,0]), 37.5, gray),
])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)


render(camera, scene, lights)