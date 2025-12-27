
from utils import *
from ray import *
from cli import render

tan = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
magenta = Material(vec([0.4, 0.2, 0.5]), k_m=0.5)
green = Material(vec([0.0, 0.7, 0.2]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)
brown = Material(vec([0.6, 0.3, 0]), k_m=0.4)

#sphere1 = Sphere(vec([-0.7,0,0]), 0.5, magenta)
#ellipse_egg = Ellipsoid(vec([-0.7,0.2,0]),vec([.5,0.125,0.125]),vec([0.125,0.5,0.125]),vec([0.125,0.125,0.5]),magenta)
#ellipse_long = Ellipsoid(vec([-0.7,0.2,0]),vec([1.3,0.125,0.125]),vec([0.125,0.5,0.125]),vec([0.125,0.125,1.3]),magenta)
#ellipse_sphere = Ellipsoid(vec([-0.7,0.2,0]),vec([0.5,0,0]),vec([0,0.5,0]),vec([0,0,0.5]),magenta)

#cylinder1 = Cylinder(vec([-0.7,0.2,0]),0.7,0.2,vec([-0.5,-0.4,-0.5]),magenta)
#cylinder = Cylinder(vec([-0.7,-.1,-1]),0.5,0.5,vec([1,0,0]),brown)
#cylinder_lamp = Cylinder(vec([-0.7,0.2,0]),0.5,0.2,vec([0.5,0.4,0.5]),magenta,True)
#cylinder_cone = Cylinder(vec([-0.7,0.2,0]),0.5,0,vec([0.5,0.4,0.5]),magenta)
#cylinder_c = Cylinder(vec([-0.5,0.6,0.2]),0.6,0.6,vec([-0.2,-0.4,-0.2]),magenta)
#bread_base = Parallelepiped(vec([-0.7,-0.3,-0.5]),vec([1,0,0]),vec([0,0.5,0]),vec([0,0,1]),brown)
#bread_base = Parallelepiped(vec([-0.7,-0.4,-1.5]),vec([1,0,0]),vec([0,0.3,0]),vec([0,0,1]),brown)
#cut_cyl = Parallelepiped(vec([-0.7,-1.1,-1.5]),vec([2,0,0]),vec([0,1,0]),vec([0,0,2]),brown)

#csg1 = CSG(Sphere(vec([0.3,0,0]), 0.5, magenta))
#csg2 = CSG(Sphere(vec([0.7,0,0]), 0.5, green))
#csg3 = CSG(Sphere(vec([0.5,0.3,0]), 0.3, green))
sphere4 = Sphere(vec([0,-40,0]), 39.5, green)
#p = Parallelepiped(vec([-0.7,0,0]),vec([0.5,0,0]),vec([0.2,0.5,0]),vec([0.2,0.2,0.5]),green)
#p2 = Parallelepiped(vec([0.45,0.3,0]),vec([0.7,0,0]),vec([0,0.7,0]),vec([0,0,0.7]),green)
#csg4 = CSG(p2)

#csg1 = CSG(cylinder)
#csg2 = CSG(bread_base)
#csg3 = CSG(cut_cyl)
#csg4 = CSG()

#half_cylinder = CSG()
#half_cylinder.add_children(csg1,csg3,"s")

#bread = CSG()
#bread.add_children(half_cylinder,csg2,"u")

#basket
cup1_outside = Cylinder(vec([0.4,-.4,.2]),0.35,0.5,vec([0,0.8,0]),magenta)
cup2_outside = Cylinder(vec([0.8,-.4,-1]),0.35,0.5,vec([0,0.8,0]),green)
cup3_outside = Cylinder(vec([-0.7,-.4,-1]),0.35,0.5,vec([0,0.8,0]),blue)

cup1_inside = Cylinder(vec([0.4,-.35,.2]),0.3,0.45,vec([0,0.75,0]),magenta)
cup2_inside = Cylinder(vec([0.8,-.35,-1]),0.3,0.45,vec([0,0.75,0]),green)
cup3_inside = Cylinder(vec([-0.7,-.35,-1]),0.3,0.45,vec([0,0.75,0]),blue)

cup1 = CSG()
cup1.add_children(cup1_outside,cup1_inside,"s")
cup2 = CSG()
cup2.add_children(cup2_outside,cup2_inside,"s")
cup3 = CSG()
cup3.add_children(cup3_outside,cup3_inside,"s")



#cylinder_slice2 = Cylinder(vec([0.9,-.4,-1]),0.5,0.5,vec([0,0.1,0]),brown)
#bread_base_slice2 = Parallelepiped(vec([0.6,-0.4,-1.5]),vec([.3,0,0]),vec([0,0.1,0]),vec([0,0,1]),brown)
#cut_cyl_slice2 = Parallelepiped(vec([-.3,-.4,-1.5]),vec([1,0,0]),vec([0,2,0]),vec([0,0,2]),brown)

#half_cylinder_slice2 = CSG()
#half_cylinder_slice2.add_children(CSG(cylinder_slice2),CSG(cut_cyl_slice2),"s")
#bread_slice2 = CSG()
#bread_slice2.add_children(half_cylinder_slice2,CSG(bread_base_slice2),"u")

#test = CSG()
#test.add_children(intersect_csg,csg4,"i")

#csg5 = CSG(Sphere(vec([.5,.6,-0.75]), 0.4, magenta))
#csg6 = CSG(Parallelepiped(vec([-0.5,-0.4,-1.25]),vec([1,0,0]),vec([0,1,0]),vec([0,0,1]),blue))
#test2 = CSG()
#test2.add_children(csg6,csg5,"i")


scene = Scene([
    #test,
    #half_cylinder,
    #cylinder,
    sphere4,
    cup1,
    cup2,
    cup3
    #handle,
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
    #bread_base
    #sphere1,

])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)

render(camera, scene, lights)