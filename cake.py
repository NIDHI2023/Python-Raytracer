from utils import *
from ray import *
from cli import render
from PIL import Image

cake = Material(vec([0.85, 0.77, 0.549]))
cake_strawberry = Material(vec([0.411, 0.078, 0.023]))
vanilla = Material(vec([0.96,0.94,0.85]))
strawberry = Material(vec([0.788, 0.27, 0.1882]))
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)



bottom_layer = CSG(Cylinder(vec([0,0,0]), 0.5, 0.5, vec([0,0.5,0]), vanilla))
middle_layer = CSG(Cylinder(vec([0,0.2,0]), 0.5, 0.5, vec([0,0.1,0]), strawberry))
top_layer = CSG(Cylinder(vec([0,0.3,0]), 0.49, 0.49, vec([0,0.2,0]), vanilla))

sliced_bottom = CSG(Parallelepiped(vec([0,0.2,0]), vec([0.7,0,0]), vec([0.2,0,0.7]), vec([0,-0.8,0]), cake))
sliced_middle = CSG(Parallelepiped(vec([0,0.3,0]), vec([0.7,0,0]), vec([0.2,0,0.7]), vec([0,-0.5,0]), cake_strawberry))
sliced_top = CSG(Parallelepiped(vec([0,0.5,0]), vec([0.7,0,0]), vec([0.2,0,0.7]), vec([0,-0.45,0]), cake))

cake1 = CSG()
cake1.add_children(middle_layer,bottom_layer,"u")

cake2 = CSG()
cake2.add_children(cake1,top_layer,"u")

sliced1 = CSG()
sliced1.add_children(sliced_bottom,sliced_middle,"u")

sliced = CSG()
sliced.add_children(sliced1, sliced_top, "u")


cake = CSG(bottom_layer)

final_cake = CSG()
final_cake.add_children(cake, sliced, "s")

#use sliced for piece on plate, add other parallelpipeds for icing.
#use final_cake for cut out cake

scene = Scene([
    final_cake,
    Sphere(vec([0,-40,0]), 39.5, gray),
])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)


render(camera, scene, lights)