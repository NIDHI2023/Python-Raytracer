from utils import *
from ray import *
from cli import render
from materials import *
from PIL import Image

red = Material(vec([0.5, 0, 0]), k_m=0)
grass_color = vec([0.1, 0.4, 0.2])
grass_color_night = vec([0.02, 0.10, 0.05])

# You must normalize the images to (0,1) range before passing them in to scene
picnic_normal = Image.open("normal_map.png")
picnic_normal = np.array(picnic_normal)
picnic_normal = picnic_normal / 255.0

picnic_texture = Image.open("picnic_texture.png")
picnic_texture = np.array(picnic_texture)
picnic_texture = picnic_texture / 255.0

basket_texture = Image.open("grass.png")
basket_texture = np.array(basket_texture)
basket_texture = basket_texture / 255.0



#---------BIRD------------------
head = Sphere(vec([-4.0,1.0,-4.2]), 0.3, bird_blue_night)
eye_left = Sphere(vec([-4.0,1.0,-3.9]), 0.04, eyes_night)
eye_right = Sphere(vec([-3.75,1.12,-4]), 0.04, eyes_night)
beak = Cylinder(vec([-3.85, 1.0, -3.95]), 0.08, 0, vec([0.1, 0, 0.1]), material=beak_yellow_night)
body = Ellipsoid(vec([-4.2,0.3,-4.6]),vec([.5,0.125,0.125]),vec([0.125,0.5,0.125]),vec([0.125,0.125,0.5]),bird_blue)
wing = Cylinder(vec([-4.4, 0.4, -4.5]), 0.2, 0, vec([-1.0, -1.0, -1.3]), material=bird_blue_night)
tail = Cylinder(vec([-4.4, 0.2, -4.5]), 0.2, 0, vec([-1.0, -1.4, -1.3]), material=bird_blue_night)
left_leg = Cylinder(vec([-3.9,0.5,-3.6]), 0.01, 0.01, vec([-0.1,-0.5,0]), beak_yellow_night)
right_leg = Cylinder(vec([-3.7,0.5,-3.6]), 0.01, 0.01, vec([-0.1,-0.5,0]), beak_yellow_night)

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


#----CAKE----
icing = CSG(Cylinder(vec([3.0,0,1.0]), 0.5, 0.5, vec([0,0.5,0]), vanilla_night))
sliced_bottom = CSG(Parallelepiped(vec([3,0.2,1]), vec([0.7,0,0]), vec([0.2,0,0.7]), vec([0,-0.8,0]), cake_night))
sliced_middle = CSG(Parallelepiped(vec([3,0.3,1]), vec([0.7,0,0]), vec([0.2,0,0.7]), vec([0,-0.5,0]), cake_strawberry_night))
sliced_top = CSG(Parallelepiped(vec([3,0.5,1]), vec([0.7,0,0]), vec([0.2,0,0.7]), vec([0,-0.45,0]), cake_night))
plate = CSG(Cylinder(vec([3.0,0,1.0]), 1, 1, vec([0,0.1,0]), plate_reflect_night))
strawberry1 = CSG(Cylinder(vec([2.8,0.5,1.1]), 0.08, 0.05, vec([0,0.15,0]), strawberry_night))
strawberry2 = CSG(Cylinder(vec([2.9,0.5,0.8]), 0.08, 0.05, vec([0,0.15,0]), strawberry_night))
strawberry3 = CSG(Cylinder(vec([3.2,0.5,0.85]), 0.08, 0.05, vec([0,0.15,0]), strawberry_night))

sliced1 = CSG()
sliced1.add_children(sliced_bottom,sliced_middle,"u")
sliced = CSG()
sliced.add_children(sliced1, sliced_top, "u")
icing_cake = CSG(icing)

final_cake = CSG()
final_cake.add_children(icing_cake, sliced, "s")

final_cake1 = CSG()
final_cake1.add_children(final_cake, strawberry1, "u")

final_cake2 = CSG()
final_cake2.add_children(final_cake1, strawberry2, "u")

final_cake3 = CSG()
final_cake3.add_children(final_cake2, strawberry3, "u")

plated_cake = CSG()
plated_cake.add_children(final_cake3, plate, "u")



#----Bread----

#Bread Loaf
cylinder = Cylinder(vec([-4,.3,-2.5]),0.5,0.5,vec([1,0,0]),brown_night)
bread_base = Parallelepiped(vec([-4,0,-3]),vec([1,0,0]),vec([0,0.3,0]),vec([0,0,1]),brown_night)
cut_cyl = Parallelepiped(vec([-4,-.7,-3]),vec([2,0,0]),vec([0,1,0]),vec([0,0,2]),brown_night)
csg1 = CSG(cylinder)
csg2 = CSG(bread_base)
csg3 = CSG(cut_cyl)
csg4 = CSG()
half_cylinder = CSG()
half_cylinder.add_children(csg1,csg3,"s")
bread = CSG()
bread.add_children(half_cylinder,csg2,"u")

#bread slice1
cylinder_slice = Cylinder(vec([-2.9,0.3,-2.5]),0.5,0.5,vec([.1,0,0]),brown_night)
bread_base_slice = Parallelepiped(vec([-2.9,0,-3]),vec([.1,0,0]),vec([0,0.3,0]),vec([0,0,1]),brown_night)
cut_cyl_slice = Parallelepiped(vec([-2.9,-.7,-3]),vec([2,0,0]),vec([0,1,0]),vec([0,0,2]),brown_night)

half_cylinder_slice = CSG()
half_cylinder_slice.add_children(CSG(cylinder_slice),CSG(cut_cyl_slice),"s")
bread_slice1 = CSG()
bread_slice1.add_children(half_cylinder_slice,CSG(bread_base_slice),"u")

#bread slice2 turned
cylinder_slice2 = Cylinder(vec([-2.4,0,-2.5]),0.5,0.5,vec([0,0.1,0]),brown_night)
bread_base_slice2 = Parallelepiped(vec([-2.7,0,-3]),vec([.3,0,0]),vec([0,0.1,0]),vec([0,0,1]),brown_night)
cut_cyl_slice2 = Parallelepiped(vec([-3.6,0,-3]),vec([1,0,0]),vec([0,2,0]),vec([0,0,2]),brown_night)

half_cylinder_slice2 = CSG()
half_cylinder_slice2.add_children(CSG(cylinder_slice2),CSG(cut_cyl_slice2),"s")
bread_slice2 = CSG()
bread_slice2.add_children(half_cylinder_slice2,CSG(bread_base_slice2),"u")


#----BASKET----
handle_outside = Cylinder(vec([2.2,1,-3.5]),1,1,vec([.1,0,0]),brown_night)
handle_inside = Cylinder(vec([2.2,1,-3.5]),0.9,0.9,vec([.1,0,0]),brown_night)
cut_handle = Parallelepiped(vec([2.2,-0.9,-4]),vec([2,0,0]),vec([0,1,0]),vec([0,0,2]),brown_night)

handle_circle = CSG()
handle_circle.add_children(CSG(handle_outside),CSG(handle_inside),"s")

handle = CSG()
handle.add_children(handle_circle,cut_handle,"s")

right_cylinder = Cylinder(vec([2.6,0,-3.5]),1,1,vec([0,1.2,0]),brown)
middle = Parallelepiped(vec([1.8,0,-4]),vec([.3,0,0]),vec([0,1.2,0]),vec([0,0,1]),brown_night)
left_cylinder = Cylinder(vec([1.8,0,-3.5]),1,1,vec([0,1.2,0]),brown_night)

basket_part = CSG()
basket_part.add_children(CSG(right_cylinder),CSG(middle),"u")
basket_part2 = CSG()
basket_part2.add_children(basket_part,CSG(left_cylinder),"u")

right_cylinder_inside = Cylinder(vec([2.5,0.05,-3.5]),0.8,0.8,vec([0,1.1,0]),brown_night)
middle_inside = Parallelepiped(vec([1.8,0.05,-4]),vec([.25,0,0]),vec([0,1.1,0]),vec([0,0,.95]),brown_night)
left_cylinder_inside = Cylinder(vec([1.9,0.05,-3.5]),0.8,0.8,vec([0,1.1,0]),brown_night)

basket_inside1 = CSG()
basket_inside1.add_children(CSG(right_cylinder_inside),CSG(left_cylinder_inside),"u")

basket_outside = CSG()
basket_outside.add_children(basket_part2,basket_inside1,"s")

basket = CSG()
basket.add_children(basket_outside,handle,"u")

#----FOOD PLATE RIGHT----
food_plate_right = CSG(Cylinder(vec([-2.2,0,1.0]), 0.6, 0.6, vec([0,0.1,0]), plate_reflect_night))

h = vec([0, 0.25, 0])

w = vec([0.3, 0, 0])

d = vec([-0.2, 0, 0.5])

sliced_bottom = CSG(Parallelepiped(vec([-2.1, 0.2, 1]),
    d,
    w,
    h, cake))
sliced_middle = CSG(Parallelepiped( vec([-2.1, 0.4, 1]),  
    d,
    w,
    h,
    cake_strawberry))
sliced_top = CSG(Parallelepiped( vec([-2.1, 0.6, 1]),
    d,
    w,
    h,
    cake))

subtract_slice = CSG(Parallelepiped( vec([-2.1, 0.85, 1]),
    vec([0.1, 0, 0.5]),
    w,
    vec([0, -1, 0]),
    brown_night))


sliced1 = CSG()
sliced1.add_children(sliced_bottom,sliced_middle,"u")
sliced = CSG()
sliced.add_children(sliced1, sliced_top, "u")

cake_slice = CSG()
cake_slice.add_children(sliced, subtract_slice, "s")

sliced_plate = CSG()
sliced_plate.add_children(food_plate_right, cake_slice, "u")


#----FOOD PLATE RIGHT----
food_plate_left = CSG(Cylinder(vec([-2.6,0,-0.5]), 0.6, 0.6, vec([0,0.1,0]), plate_reflect_night))

#----CUPS----
cup1_outside = Cylinder(vec([0.4,0,.2]),0.35,0.5,vec([0,0.8,0]),magenta_night)
cup2_outside = Cylinder(vec([0.8,0,-1]),0.35,0.5,vec([0,0.8,0]),green_night)
cup3_outside = Cylinder(vec([-0.3,0,-.7]),0.35,0.5,vec([0,0.8,0]),blue_night)

cup1_inside = Cylinder(vec([0.4,.05,.2]),0.3,0.45,vec([0,0.75,0]),magenta_night)
cup2_inside = Cylinder(vec([0.8,.05,-1]),0.3,0.45,vec([0,0.75,0]),green_night)
cup3_inside = Cylinder(vec([-0.3,.05,-.7]),0.3,0.45,vec([0,0.75,0]),blue_night)

cup1 = CSG()
cup1.add_children(cup1_outside,cup1_inside,"s")
cup2 = CSG()
cup2.add_children(cup2_outside,cup2_inside,"s")
cup3 = CSG()
cup3.add_children(cup3_outside,cup3_inside,"s")

g1 = CSG(SphereTextured(vec([-2.7,0.2,-0.4]), 0.1, magenta_night))
g2 = CSG(SphereTextured(vec([-2.8,0.25,-0.4]), 0.1, magenta_night))
g3 = CSG(SphereTextured(vec([-2.5,0.2,-0.4]), 0.1, magenta_night))
g4 = CSG(SphereTextured(vec([-2.6,0.3,-0.5]), 0.1, magenta_night))
g5 = CSG(SphereTextured(vec([-2.7,0.4,-0.5]), 0.1, magenta_night))
g6 = CSG(SphereTextured(vec([-2.8,0.2,-0.2]), 0.1, magenta_night))
stem = Cylinder(vec([-2.6,0.4,-0.5]), 0.02, 0.02, vec([0.15,0.2,-0.3]), green_night)

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

grape_plate = CSG()
grape_plate.add_children(final_grape, food_plate_left, "u")

#---PANCAKES----
pancake_color = Material(vec([0.631, 0.490, 0.310]))
x_axis = vec([ 0.98, 0.00,  0.20 ])   
y_axis = vec([ 0.00, 1.00,  0.00 ])   
z_axis = vec([-0.20, 0.00,  0.98 ])  

rx = x_axis * 0.7     
ry = y_axis * 0.18    
rz = z_axis * 0.7    

# Pancake centers stacked upward
p1 = vec([2.6, 0.00, -1.5])
p2 = vec([2.6, 0.20, -1.5])
p3 = vec([2.6, 0.40, -1.5])

bottom  = CSG(Ellipsoid(p1,rx,ry,rz, pancake_color))
middle  = CSG(Ellipsoid(p2, rx,ry,rz, pancake_color))
top     = CSG(Ellipsoid(p3, rx,ry,rz, pancake_color))
butter_color = Material(vec([0.95, 0.85, 0.35]))
b_pos = vec([2.6 + 0.10, 0.40 + 0.12, -1.5 + 0.05])

butter = Parallelepiped(b_pos,vec([0.12, 0.00, 0.01]), vec([0.00, 0.12, 0.00]),vec([-0.01, 0.00, 0.12]) ,butter_color )

pancake1 = CSG()
pancake1.add_children(bottom,middle, "u")

pancake2 =CSG()
pancake2.add_children(pancake1, top,"u")

pancake =CSG()
pancake.add_children(pancake2, butter,"u")



scene = Scene([
    basket,
    pancake,
    ParallelpipedPlaneTextured(vec([-7.0,0,-1]), vec([10,0,-8]), vec([5,0,10]), vec([0,-0.1,0]), red),
    
],
bg_color=grass_color_night
)

lights = [
    PointLight(vec([-4,10,2]), vec([300,300,300])),
    AmbientLight(vec([0.6196, 0.4902, 0.2471])),
]

camera = Camera(vec([0,5,5]), target=vec([0,0,-1]),  up=vec([0,1,0]), vfov=40, aspect=16/9)


render(camera, scene, lights)