
from utils import *
from ray import *
from cli import render

tan = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
magenta = Material(vec([0.4, 0.2, 0.5]), k_m=0.5)
green = Material(vec([0.0, 0.7, 0.2]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

sphere1 = Sphere(vec([-0.7,0,0]), 0.5, magenta)
ellipse_egg = Ellipsoid(vec([-0.7,0.2,0]),vec([.5,0.125,0.125]),vec([0.125,0.5,0.125]),vec([0.125,0.125,0.5]),magenta)
ellipse_long = Ellipsoid(vec([-0.7,0.2,0]),vec([1.3,0.125,0.125]),vec([0.125,0.5,0.125]),vec([0.125,0.125,1.3]),magenta)
ellipse_sphere = Ellipsoid(vec([-0.7,0.2,0]),vec([0.5,0,0]),vec([0,0.5,0]),vec([0,0,0.5]),magenta)

cylinder1 = Cylinder(vec([-0.7,0.2,0]),0.7,0.2,vec([-0.5,-0.4,-0.5]),magenta)
cylinder_works = Cylinder(vec([-0.7,0.2,0]),0.5,0.2,vec([0.5,0.4,0.5]),magenta)
cylinder_lamp = Cylinder(vec([-0.7,0.2,0]),0.5,0.2,vec([0.5,0.4,0.5]),magenta,True)
cylinder_cone = Cylinder(vec([-0.7,0.2,0]),0.5,0,vec([0.5,0.4,0.5]),magenta)
cylinder_c = Cylinder(vec([-0.5,0.6,0.2]),0.6,0.6,vec([-0.2,-0.4,-0.2]),magenta)
p3 = Parallelepiped(vec([-0.7,0.2,0]),vec([1,0,0]),vec([0,0.1,0]),vec([0,0,1]),magenta)

csg1 = CSG(Sphere(vec([0.3,0,0]), 0.5, magenta))
csg2 = CSG(Sphere(vec([0.7,0,0]), 0.5, green))
csg3 = CSG(Sphere(vec([0.5,0.3,0]), 0.3, green))
sphere4 = Sphere(vec([0,-40,0]), 39.5, gray)
p = Parallelepiped(vec([-0.7,0,0]),vec([0.5,0,0]),vec([0.2,0.5,0]),vec([0.2,0.2,0.5]),green)
p2 = Parallelepiped(vec([0.45,0.3,0]),vec([0.7,0,0]),vec([0,0.7,0]),vec([0,0,0.7]),green)
csg4 = CSG(p2)

intersect_csg = CSG()
intersect_csg.add_children(csg1,csg2,"i")

test = CSG()
test.add_children(intersect_csg,csg4,"s")

csg5 = CSG(Sphere(vec([.5,.6,-0.75]), 0.4, magenta))
csg6 = CSG(Parallelepiped(vec([-0.5,-0.4,-1.25]),vec([1,0,0]),vec([0,1,0]),vec([0,0,1]),blue))
test2 = CSG()
test2.add_children(csg6,csg5,"i")


scene = Scene([
    test,
    #intersect_csg,
    sphere4,
    #test2,
    #p,
    #ellipse_egg,
    #ellipse_long,
    #ellipse_sphere,
    #cylinder1
    #cylinder_works
    #cylinder_lamp
    #cylinder_cone
    #cylinder_c,
    #p3
    #sphere1,

])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)

render(camera, scene, lights)