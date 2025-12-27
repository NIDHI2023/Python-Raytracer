from utils import *
from ray import *
from cli import render
from PIL import Image

tan = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
magenta = Material(vec([0.4, 0.2, 0.5]), k_m=0.5)
green = Material(vec([0.0, 0.7, 0.2]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)
text = Material(vec([0.2, 0.2, 0.2]), k_m=0)

# You must normalize the images to (0,1) range before passing them in to scene
normal = Image.open("normal_map_test2.png")
normal = np.array(normal)
normal = normal / 255.0

texture = Image.open("kiran_texture.png")
texture = np.array(texture)
texture = texture / 255.0

p = Parallelepiped(vec([-0.7,0,0]),vec([0.5,-0.125,-0.125]),vec([-0.125,0.5,-0.125]),vec([-0.125,-0.125,0.5]),green)
csg1 = CSG(SphereTextured(vec([0.3,0,0]), 0.5, magenta, texture_map=texture))
csg2 = CSG(SphereTextured(vec([0.7,0,0]), 0.5, blue, normal_map=normal))
csg3 = CSG(SphereTextured(vec([0.5,0.3,0]), 0.3, green, normal_map=normal))

intersect_csg = CSG()
intersect_csg.add_children(csg1,csg2,"i")

test = CSG()
test.add_children(intersect_csg,csg3,"s")


scene = Scene([
    ParallelpipedTextured(vec([-0.7,0,0]),vec([0.5,-0.125,-0.125]),vec([-0.125,0.5,-0.125]),vec([-0.125,-0.125,0.5]),green, normal_map=normal),
    #test,
    Sphere(vec([0,-40,0]), 39.5, gray),
])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)


render(camera, scene, lights)