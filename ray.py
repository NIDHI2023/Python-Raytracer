from typing import Tuple
import numpy as np

from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in 
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might 
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d


class Hit:

    def __init__(self, t, point=None, normal=None, material=None, surface=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
          surface : Sphere, SphereTextured, Parallelepiped, ParallelepipedTextured, CSG, etc. -- the surface itself. Used for determining when to texture map
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material
        self.surface = surface


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


def dist(point_a, point_b):
    return np.linalg.norm(point_a - point_b)


class Cylinder:
    def __init__(self, bot_center, act_bot_radius, act_top_radius, vec_height, material, hollow=False):
        """Create a cylinder or cone segment

        Parameters:
          bot_center : (3,) -- a 3D point specifying the center of the cylinder's bottom face
          act_bot_radius : float -- a Python float specifying the actual radius of the cylinder's bottom face
          act_top_radius : float -- a Python float specifying the actual radius of the cylinder's top face
          vec_height : (3,) -- a 3D vector specifying the cylinder's height and direction
          material : Material -- the material of the surface
        """
         
        self.center = bot_center + vec_height/2 
        self.height = np.linalg.norm(vec_height)
        self.bot_center = bot_center
        self.bot_radius = act_bot_radius/self.height #so it can be used in our coord system
        self.top_radius = act_top_radius/self.height
        self.vec_height = vec_height
        self.material = material
        self.hollow = hollow


        #user expected to not put in [0,0,0] vector for the height
        if not (vec_height[2] == 0):
            x_unnorm = [1,0,-vec_height[0]/vec_height[2]]
        elif not (vec_height[1] == 0):
            x_unnorm = [1,-vec_height[0]/vec_height[1],0]
        else:
            x_unnorm = [-vec_height[1]/vec_height[0],1,0]

        self.x_vec = (x_unnorm/np.linalg.norm(x_unnorm))*self.height
        self.y_vec = np.cross(self.vec_height,self.x_vec)/self.height

        #transforms from world to obj coords
        self.trans_obj_to_w = np.array([np.append(self.x_vec,0),np.append(self.y_vec,0),np.append(vec_height,0),np.append(self.bot_center,1)],np.float64).T
        self.trans_w_to_obj = np.linalg.inv(self.trans_obj_to_w)

        un_norm_norm_bot = np.matmul(self.trans_obj_to_w,[0,0,-1,0])
        self.normalbot = (un_norm_norm_bot[:3])/(np.linalg.norm(un_norm_norm_bot[:3]))

        un_norm_norm_top = np.matmul(self.trans_obj_to_w,[0,0,1,0])
        self.normaltop = (un_norm_norm_top[:3])/(np.linalg.norm(un_norm_norm_top[:3]))


    def inside(self, point):
        
        pt = np.append(point,1)
        trans_pt = np.matmul(self.trans_w_to_obj,pt)
        z = trans_pt[2]
        rad = np.linalg.norm(trans_pt[:2])
        if (rad <= (1.0001 * (((1-z)*self.bot_radius) + (z*self.top_radius))) and 
            z > -.0001 and z < 1.0001):
            return True
        return False
    
    def strict_inside(self,point):
        pt = np.append(point,1)
        trans_pt = np.matmul(self.trans_w_to_obj,pt)
        z = trans_pt[2]
        rad = np.linalg.norm(trans_pt[:2])
        if (rad <= (.999 * (((1-z)*self.bot_radius) + (z*self.top_radius))) and 
            z > .0001 and z < .999):
            return True
        return False
    
    def hit_list(self,ray):
        #intersect w/ sphere after transitioning to object coords

        #ray distance
        v = np.matmul(self.trans_w_to_obj,np.append(ray.direction,0)) #v
        #ray origin
        p = np.matmul(self.trans_w_to_obj,np.append(ray.origin,1)) #p

        r_b = self.bot_radius
        r_t = self.top_radius
        hit_list = []

        #could start with initial quadratic equation
        #a = (v[0])**2 + (v[1])**2 - (v[2]**2 * self.bot_radius**2) - (v[2]**2 * self.top_radius**2) + 2*(v[2]**2 * self.bot_radius * self.top_radius)
        #c = p[0]**2 + p[1]**2 - ((1-p[2])**2 * self.bot_radius**2) - (p[2]**2 * self.top_radius**2) - 2*(p[2]**2 * self.bot_radius * self.top_radius) + 2*(p[1]**2 * self.bot_radius * self.top_radius)
        #b = 2*(p[0] * v[0]) + 2*(p[1] * v[1]) - 2*((1-p[2]) * (-v[2]) * self.bot_radius**2) - 2*(p[2] * v[2] * self.top_radius**2) - (2*(v[2] - 2*p[2]*v[2]) * self.bot_radius * self.top_radius)
        
        c = np.float64(p[0]**2 + p[1]**2 - r_b**2 + (2*(r_b**2)*p[2]) - (2*r_t*r_b*p[2]) - ((r_b**2)*(p[2]**2)) - ((r_t**2)*(p[2]**2)) + (2*r_b*r_t*(p[2]**2)))
        b = np.float64((2*v[0]*p[0]) + (2*v[1]*p[1]) + (2*(r_b**2)*v[2]) - (2*r_t*r_b*v[2]) - (2*(r_b**2)*v[2]*p[2]) - (2*(r_t**2)*v[2]*p[2]) + (4*r_b*r_t*v[2]*p[2]))
        a = np.float64(v[0]**2 + v[1]**2 - ((r_b**2)*(v[2]**2)) - ((r_t**2)*(v[2]**2)) + (2*r_b*r_t*(v[2]**2)))


        discriminant = np.power(b,2) - (4*a*c)


        
        # Check if no sols
        if (discriminant < 0.0):
            return [no_hit]
        if (a == 0):
            t = -c/b
            q = p + (t*v)
            #transform to world
            if (q[2] > -0.0001 and q[2] < 1.0001):
              rad = np.linalg.norm(q[:2])

              un_norm_norm = [q[0]/rad,q[1]/rad,self.bot_radius-self.top_radius,0]
              un_norm_norm_world = np.matmul(self.trans_obj_to_w,un_norm_norm)
              
              normal = (un_norm_norm_world[:3])/(np.linalg.norm(un_norm_norm_world[:3]))
              q_trans_p = np.matmul(self.trans_obj_to_w,q)
              hit_list = [Hit(t, q_trans_p[:3], normal, self.material)]
            
        tplus = (- b + np.sqrt(discriminant))/(2*a)
        tminus = (- b - np.sqrt(discriminant))/(2*a)

        qplus = p + (tplus*v)
        qminus = p + (tminus*v)

        
        if (qminus[2] > -0.0001 and qminus[2] < 1.0001):
            rad = np.linalg.norm(qminus[:2])

            un_norm_norm_minus = [qminus[0]/rad,qminus[1]/rad,self.bot_radius-self.top_radius,0]
            un_norm_norm_world_minus = np.matmul(self.trans_obj_to_w,un_norm_norm_minus)

            normalminus = (un_norm_norm_world_minus[:3])/(np.linalg.norm(un_norm_norm_world_minus[:3]))

            qminus_trans_p = np.matmul(self.trans_obj_to_w,qminus)

            hit_list = np.append(hit_list, Hit(tminus,qminus_trans_p[:3],normalminus,self.material))


        if (qplus[2] > -0.0001 and qplus[2] < 1.0001):
            rad = np.linalg.norm(qplus[:2])

            un_norm_norm_plus = [qplus[0]/rad,qplus[1]/rad,self.bot_radius-self.top_radius,0]
            un_norm_norm_world_plus = np.matmul(self.trans_obj_to_w,un_norm_norm_plus)

            normalplus = (un_norm_norm_world_plus[:3])/(np.linalg.norm(un_norm_norm_world_plus[:3]))

            qplus_trans_p = np.matmul(self.trans_obj_to_w,qplus)

            hit_list = np.append(hit_list, Hit(tplus,qplus_trans_p[:3],normalplus,self.material))
   


        if not self.hollow:
            if v[2] == 0:
                return hit_list
            #z=0
            t_bot = (-p[2])/v[2]
            #z=1
            t_top = (1-p[2])/v[2]

            q_bot = p + (v*t_bot)
            q_top = p + (v*t_top)


            if np.linalg.norm(q_bot[:2]) <= (1.0001 * r_b):
                q_bot_trans = ray.origin + (ray.direction*t_bot)
                hit_list = np.append(hit_list, Hit(t_bot,q_bot_trans[:3],self.normalbot,self.material))
                #if np.abs(dist(q_bot[:3],self.bot_center) - self.bot_radius) < (0.05 * self.bot_radius):
                #    print("normal",self.normalbot)
            

            if np.linalg.norm(q_top[:2]) <= (1.0001 * r_t):
                q_top_trans = ray.origin + (ray.direction*t_top)
                hit_list = np.append(hit_list, Hit(t_top,q_top_trans[:3],self.normaltop,self.material))



        if len(hit_list) == 0:
            return [no_hit]
        
        return hit_list
    
    def intersect(self, ray):
        val = self.hit_list(ray)
        min_t = minimum_t(val, ray)
        #if not min_t == no_hit:
          #if not min_t.normal[0] == self.normalbot[0]:
            #print(val[0].t,val[1].t,min_t.t, (val[0].t < val[1].t))
            #print(min_t.point[1],(val[0].t < val[1].t))
        return (min_t)

class Ellipsoid:
    def __init__(self, center, x, y, z, material):
        """Create an ellipsoid

        Parameters:
          center : (3,) -- a 3D point specifying the ellipsoid's center
          x : (3,) -- a 3D vector specifying the ellipsoid's a coef
          y : (3,) -- a 3D vector specifying the ellipsoid's b coef
          z : (3,) -- a 3D vector specifying the ellipsoid's c coef
          material : Material -- the material of the surface
        """
        #honomegizing the point and vectors
        self.center = center
        self.x = x
        self.y = y
        self.z = z
        self.material = material

        #transforms from world to obj coords
        self.trans_obj_to_w = np.array([np.append(x,0),np.append(y,0),np.append(z,0),np.append(center,1)],np.float64).T
        self.trans_w_to_obj = np.linalg.inv(self.trans_obj_to_w)


    def inside(self, point):
        pt = np.append(point,1)
        trans_pt = np.matmul(self.trans_w_to_obj,pt)
        if (dist(trans_pt[:3],[0,0,0]) <= 1.001):
            return True
        return False
    
    def strict_inside(self,point):
        pt = np.append(point,1)
        trans_pt = np.matmul(self.trans_w_to_obj,pt)
        if (dist(trans_pt[:3],[0,0,0]) <= .999):
            return True
        return False
         

    def hit_list(self, ray):
        #intersect w/ sphere after transitioning to object coords

        #ray distance
        trans_ray_d = np.matmul(self.trans_w_to_obj,np.append(ray.direction,0)) #v
        #ray origin
        trans_ray_o = np.matmul(self.trans_w_to_obj,np.append(ray.origin,1)) #p

        # ray has vec3 of origin and direction
        c = np.dot(trans_ray_o[:3], trans_ray_o[:3]) - 1
        
        b = 2 * np.dot(trans_ray_o[:3], trans_ray_d[:3])
        a = np.dot(trans_ray_d[:3], trans_ray_d[:3])
        discriminant = np.power(b,2) - (4*a*c)

        # Check if no sols
        if (discriminant < 0.0):
            return [no_hit]
        if (a == 0):
            t = -c/b
            q = trans_ray_o + (t*trans_ray_d)
            #transform to world
            q_trans_p = np.matmul(self.trans_obj_to_w,q)
            #z = (-a)/c, solving equation where dot product is 0
            perp1 = vec([1,0,(-q[0])/(q[2])])
            perp2 = np.cross(q[:3],perp1)
            #TODO: fix variable names
            #unnormalized normal :)
            un_norm_normal = np.cross(np.matmul(self.trans_obj_to_w,np.append(perp1,0))[:3],
                                      np.matmul(self.trans_obj_to_w,np.append(perp2,0))[:3])
            normal = (un_norm_normal)/(np.linalg.norm(un_norm_normal))
            return [Hit(t, q_trans_p[:3], normal, self.material)]
        tplus = (- b + np.sqrt(discriminant))/(2*a)
        tminus = (- b - np.sqrt(discriminant))/(2*a)

        qplus = trans_ray_o + (tplus*trans_ray_d)
        qminus = trans_ray_o + (tminus*trans_ray_d)

        qplus_trans_p = np.matmul(self.trans_obj_to_w,qplus)
        qminus_trans_p = np.matmul(self.trans_obj_to_w,qminus)

        perp1_plus = vec([1,0,(-qplus[0])/(qplus[2])])
        perp2_plus = np.cross(qplus[:3],perp1_plus)
        un_norm_normal_plus = np.cross(np.matmul(self.trans_obj_to_w,np.append(perp1_plus,0))[:3],
                                       np.matmul(self.trans_obj_to_w,np.append(perp2_plus,0))[:3])
        normalplus = (un_norm_normal_plus)/(np.linalg.norm(un_norm_normal_plus))

        perp1_minus = vec([1,0,(-qminus[0])/(qminus[2])])
        perp2_minus = np.cross(qminus[:3],perp1_minus)
        un_norm_normal_minus = np.cross(np.matmul(self.trans_obj_to_w,np.append(perp1_minus,0))[:3],
                                        np.matmul(self.trans_obj_to_w,np.append(perp2_minus,0))[:3])
        normalminus = (un_norm_normal_minus)/(np.linalg.norm(un_norm_normal_minus))
        #print("qplus",qplus,"qplus_trans_p",qplus_trans_p,'normalplus',normalplus)

        return [Hit(tminus,qminus_trans_p[:3],normalminus,self.material), Hit(tplus,qplus_trans_p[:3],normalplus,self.material)]
        
        
    def intersect(self, ray):
        return (minimum_t(self.hit_list(ray), ray))

class Parallelepiped:
    def __init__(self, origin, x, y, z, material):
        """Create a parallelepiped with the given parameters. These are created using the Right Handed Coord System

        Parameters:
          origin : (3,) -- a 3D point specifying the parallelepiped's corner from which x, y, and z are defined
          length : (3,) -- a 3D vector specifying the parallelepiped's x
          width : (3,) -- a 3D vector specifying the parallelepiped's y
          height : (3,) -- a 3D vector specifying the parallelepiped's z
          material : Material -- the material of the surface
        """
        self.origin = origin
        self.vec1 = x # using picnic blanket as referece, assume length is larger side. x in WORLD coords
        self.vec2 = y # using picnic blanket as referece, assume width is smaller side. z in WORLD coords
        self.vec3 = z # y in WORLD coords
        self.material = material

        self.center = origin + x/2 + y/2 + z/2
        self.radius = max(np.linalg.norm(origin-self.center),np.linalg.norm(origin+x-self.center),np.linalg.norm(origin+y-self.center),np.linalg.norm(origin+z-self.center))

        #transforms from world to obj coords
        self.trans_obj_to_w = np.array([np.append(self.vec1,0),np.append(self.vec2,0),np.append(self.vec3,0),np.append(self.origin,1)],np.float64).T
        self.trans_w_to_obj = np.linalg.inv(self.trans_obj_to_w)
        
        # NOTE: these are NOT normalized values
        self.xnormplus = np.cross(self.vec2,self.vec3)/np.linalg.norm(np.cross(self.vec2,self.vec3))
        self.ynormplus = np.cross(self.vec3,self.vec1)/np.linalg.norm(np.cross(self.vec3,self.vec1))
        self.znormplus = np.cross(self.vec1,self.vec2)/np.linalg.norm(np.cross(self.vec1,self.vec2))

        self.xnormplus = self.xnormplus / np.linalg.norm(self.xnormplus)
        self.ynormplus = self.ynormplus / np.linalg.norm(self.ynormplus)
        self.znormplus = self.znormplus / np.linalg.norm(self.znormplus)

    def inside(self,point):
        pt = np.append(point,1)
        trans_pt = np.matmul(self.trans_w_to_obj,pt)
        if trans_pt[0] <= 1.0001 and trans_pt[0] >= -.0001:
            if trans_pt[1] <= 1.0001 and trans_pt[1] >= -.0001:
                if trans_pt[2] <= 1.0001 and trans_pt[2] >= -.0001:
                    return True
        return False
    
    def strict_inside(self,point):
        pt = np.append(point,1)
        trans_pt = np.matmul(self.trans_w_to_obj,pt)
        if trans_pt[0] <= .999 and trans_pt[0] >= .0001:
            if trans_pt[1] <= .999 and trans_pt[1] >= .0001:
                if trans_pt[2] <= .999 and trans_pt[2] >= .0001:
                    return True
        return False
    

    def plane_intersection(self,axis,translation,v,p):
        #ray = p + vt

        #ax+by+cz=d for plane 

        t = -np.inf

        if axis == 0:
            if np.dot(v,[1,0,0]) == 0:
                return [[np.inf,np.inf,np.inf],-np.inf] #ray is parallel to plane 
            t = (translation-p[0])/v[0]
            q = p + v*t
        if axis == 1:
            if np.dot(v,[0,1,0]) == 0:
                return [[np.inf,np.inf,np.inf],-np.inf] #ray is parallel to plane 
            t = (translation-p[1])/v[1]
            q = p + v*t
        if axis == 2:
            if np.dot(v,[0,0,1]) == 0:
                return [[np.inf,np.inf,np.inf],-np.inf] #ray is parallel to plane 
            t = (translation-p[2])/v[2]
            q = p + v*t

        return [q[:3],t]
    
    def point_in_bounds(self, point, axis):
        if axis == 0:
            if point[1] > -0.0001 and point[1] < 1.0001 and point[2] > -0.0001 and point[2] < 1.0001:
                return True
        elif axis == 1:
            if point[0] > -0.0001 and point[0] < 1.0001 and point[2] > -0.0001 and point[2] < 1.0001:
                return True
        elif axis == 2:
            if point[1] > -0.0001 and point[1] < 1.0001 and point[0] > -0.0001 and point[0] < 1.0001:
                return True
        return False

    
    def hit_list(self,ray):
        hit_list = []

        #Sphere test for faster time

        origin = ray.origin-self.center

        c = np.dot(origin, origin) - np.power(self.radius,2)
        
        b = 2 * np.dot(origin, ray.direction)
        a = np.dot(ray.direction, ray.direction)
        discriminant = np.power(b,2) - (4*a*c)
        if discriminant < 0.0001:
            return hit_list

        #ray distance
        trans_ray_d = np.matmul(self.trans_w_to_obj,np.append(ray.direction,0)) #v
        #ray origin
        trans_ray_o = np.matmul(self.trans_w_to_obj,np.append(ray.origin,1)) #p

        #vec2 and vec3
        q1,t1 = self.plane_intersection(0,0,trans_ray_d[:3],trans_ray_o[:3])
        if self.point_in_bounds(q1,0):
            q1_world = np.matmul(self.trans_obj_to_w,np.append(q1,1))
            hit_list.append(Hit(t1,q1_world[:3],-self.xnormplus,self.material))

        q2,t2 = self.plane_intersection(0,1,trans_ray_d[:3],trans_ray_o[:3])
        if self.point_in_bounds(q2,0):
            q2_world = np.matmul(self.trans_obj_to_w,np.append(q2,1))
            hit_list.append(Hit(t2,q2_world[:3],self.xnormplus,self.material))

        #vec3 and vec1
        q3,t3 = self.plane_intersection(1,0,trans_ray_d[:3],trans_ray_o[:3])
        if self.point_in_bounds(q3,1):
            q3_world = np.matmul(self.trans_obj_to_w,np.append(q3,1))
            hit_list.append(Hit(t3,q3_world[:3],-self.ynormplus,self.material))

        q4,t4 = self.plane_intersection(1,1,trans_ray_d[:3],trans_ray_o[:3])
        if self.point_in_bounds(q4,1):
            q4_world = np.matmul(self.trans_obj_to_w,np.append(q4,1))
            hit_list.append(Hit(t4,q4_world[:3],self.ynormplus,self.material))

        #vec1 and vec2
        q5,t5 = self.plane_intersection(2,0,trans_ray_d[:3],trans_ray_o[:3])
        if self.point_in_bounds(q5,2):
            q5_world = np.matmul(self.trans_obj_to_w,np.append(q5,1))
            hit_list.append(Hit(t5,q5_world[:3],-self.znormplus,self.material))

        q6,t6 = self.plane_intersection(2,1,trans_ray_d[:3],trans_ray_o[:3])
        if self.point_in_bounds(q6,2):
            q6_world = np.matmul(self.trans_obj_to_w,np.append(q6,1))
            hit_list.append(Hit(t6,q6_world[:3],self.znormplus,self.material))
            
        return hit_list
    
    def intersect(self,ray):
        return (minimum_t(self.hit_list(ray), ray))

class ParallelpipedTextured(Parallelepiped):
    """
          origin : (3,) -- a 3D point specifying the parallelepiped's corner from which length, width, and height are defined
          length : (3,) -- a 3D vector specifying the parallelepiped's length
          width : (3,) -- a 3D vector specifying the parallelepiped's width
          height : (3,) -- a 3D vector specifying the parallelepiped's height
          material : Material -- the material of the surface
    """
    def __init__(self, origin, length, width, height, material, texture_map=None, normal_map=None):
        super().__init__(origin, length, width, height, material)
        self.texture_map = texture_map 
        self.normal_map = normal_map 
    def sample_texture(self, u, v):
        if self.texture_map is not None and len(self.texture_map) > 0 and self.texture_map.shape[0] > 0 and self.texture_map.shape[1] > 0:
          U_REPEAT = 4.0
          V_REPEAT = 4.0

          u = (u * U_REPEAT) % 1.0
          v = (v * V_REPEAT) % 1.0
          x = u * (self.texture_map.shape[1] - 1)
          y = v * (self.texture_map.shape[0] - 1)
          

          x0 = int(np.floor(x))
          x1 = min(x0 + 1, self.texture_map.shape[1] - 1)
          y0 = int(np.floor(y))
          y1 = min(y0 + 1, self.texture_map.shape[0] - 1)

          alpha = x - x0
          beta = y - y0

          # Fetch four neighbors
          c00 = self.texture_map[y0, x0]
          c10 = self.texture_map[y0, x1]
          c01 = self.texture_map[y1, x0]
          c11 = self.texture_map[y1, x1]

          texture_color = (1 - alpha)*(1 - beta)*c00 + alpha*(1 - beta)*c10 + (1 - alpha)*beta*c01 + alpha*beta*c11
          #return texture_color/255.0
          return texture_color
        else:
          return []

    def sample_normal(self, u, v):
        if self.normal_map is not None and len(self.normal_map > 0) and self.normal_map.shape[0] > 0 and self.normal_map.shape[1] > 0:
          normal_width = self.normal_map.shape[1]
          normal_height = self.normal_map.shape[0]
          x = u * (normal_width - 1)
          y = v * (normal_height - 1)

          x0 = int(np.floor(x))
          x1 = min(x0 + 1, normal_width - 1)
          y0 = int(np.floor(y))
          y1 = min(y0 + 1, normal_height - 1)

          alpha = x - x0
          beta = y - y0

          # Fetch four neighbors
          c00 = self.normal_map[y0, x0]
          c10 = self.normal_map[y0, x1]
          c01 = self.normal_map[y1, x0]
          c11 = self.normal_map[y1, x1]

          normal_color = (1 - alpha)*(1 - beta)*c00 + alpha*(1 - beta)*c10 + (1 - alpha)*beta*c01 + alpha*beta*c11

          #change texture from (0,255) -> (0,1)
          #normal_color = normal_color / 255.0
          # map (0,1) -> (-1,1) to allow for negative normals
          normal_color = (2.0*normal_color) - 1.0
          return normal_color
        else:
         return []
       

    #source: https://medium.com/@Ksatese/advanced-ray-tracer-part-4-87d1c98eecff
    def sphere_uv_calculations(self,hit:Hit)->Tuple[int, int]:
      point = hit.point - self.center
      theta = np.arccos(point[1]/self.radius)
      phi = np.arctan2(point[2], point[0])
      u = (-phi + np.pi) / (2*np.pi)
      v = theta/np.pi
      return (u,v)
    
    def sphere_normal_calculations(self,hit:Hit):
        point = hit.point - self.center
        x=point[0]
        y=point[1]
        z=point[2]
        theta = np.arccos(y/self.radius)
        phi = np.arctan2(z,x)
        T=np.array([2*np.pi*z, 0 , -2*np.pi*x])
        B=np.array([np.pi*y*np.cos(phi), -self.radius*np.pi*np.sin(theta), np.pi*y*np.sin(phi)])
        T = T / np.linalg.norm(T)
        B = B / np.linalg.norm(B)
        N = hit.normal
        return np.array([np.array([T[0], B[0], N[0]]),
                        np.array([T[1], B[1], N[1]]),
                        np.array([T[2], B[2], N[2]])])
    


class ParallelpipedPlaneTextured():
    """
          origin : (3,) -- a 3D point specifying the parallelepiped's corner from which length, width, and height are defined
          length : (3,) -- a 3D vector specifying the parallelepiped's length (vec1)
          width : (3,) -- a 3D vector specifying the parallelepiped's width (vec2)
          height : (3,) -- a 3D vector specifying the parallelepiped's height (vec3)
          material : Material -- the material of the surface
    """
    def __init__(self, origin, length, width, height, material, texture_map=None, normal_map=None):
        """Create a parallelepiped with the given parameters. These are created using the Right Handed Coord System

        Parameters:
          origin : (3,) -- a 3D point specifying the parallelepiped's corner from which length, width, and height are defined
          length : (3,) -- a 3D vector specifying the parallelepiped's length
          width : (3,) -- a 3D vector specifying the parallelepiped's width
          height : (3,) -- a 3D vector specifying the parallelepiped's height
          material : Material -- the material of the surface
        """
        self.origin = origin
        self.vec1 = length
        self.vec2 = width
        self.vec3 = height
        self.material = material

        self.texture_map = texture_map 
        self.normal_map = normal_map 

        self.center = origin + length/2 + width/2 + height/2
        self.radius = max(np.linalg.norm(origin-self.center),np.linalg.norm(origin+length-self.center),np.linalg.norm(origin+width-self.center),np.linalg.norm(origin+height-self.center))

        self.v23 = np.cross(self.vec2,self.vec3)
        self.v31 = np.cross(self.vec3,self.vec1)
        self.v12 = np.cross(self.vec1,self.vec2)

    def inside(self,point):
        vec = point - self.origin
        mat = np.array([self.vec1, self.vec2, self.vec3]).T #matrix to get from parallelepiped's coord system to world coords
        mat_inv = np.linalg.inv(mat) #from world to parallelepiped

        new_v = np.matmul(mat_inv,vec)

        if new_v[0] <= (np.linalg.norm(self.vec1))*1.0001 and new_v[0] >= -.0001:
            if new_v[1] <= (np.linalg.norm(self.vec2))*1.0001 and new_v[1] >= -.0001:
                if new_v[2] <= (np.linalg.norm(self.vec3))*1.0001 and new_v[2] >= -.0001:
                    return True
        return False
    

    def plane_intersection(self,origin,vec1,vec2,ray):
        #ray = p + vt
        v = np.append(ray.direction,0) #v
        p = np.append(ray.origin,1) #p

        #ax+by+cz=d for plane 

        #take cross product of vectors
        norm = np.cross(vec1,vec2) #normal vector to plane

        #ax+by+cz=d for plane 
        a = norm[0]
        b = norm[1]
        c = norm[2]
        #to find d, use a,b,c and plug in origin point

        d = -((a*origin[0]) + (b*origin[1]) + (c*origin[2]))

        #let coord vector be alpha
        A = np.array([a,b,c,d]).T
        if (np.dot(A,v) == 0):
            return [[np.inf,np.inf,np.inf],-np.inf, norm] #ray is parallel to plane
        t = (-(np.dot(A,p)/(np.dot(A,v))))
        q = p + v * t
        #normal = (q-self.origin)/(np.linalg.norm(q-self.origin))

        return [q[:3],t, norm]
    
    def point_in_bounds(self, point, origin, vec1, vec2, norm):
        A = np.array([vec1,vec2,norm]).T
        vec = np.linalg.solve(A, point-origin)

        if vec[0] > -0.0001 and vec[1] > -0.0001 and vec[0] < 1.0001 and vec[1] < 1.0001:
            return True
        return False

    
    def hit_list(self,ray):
        hit_list = []

        origin = ray.origin-self.center

        c = np.dot(origin, origin) - np.power(self.radius,2)
        
        b = 2 * np.dot(origin, ray.direction)
        a = np.dot(ray.direction, ray.direction)
        discriminant = np.power(b,2) - (4*a*c)
        if discriminant < 0.0001:
            return hit_list

        #vec1 and vec2
        q1,t1,norm1 = self.plane_intersection(self.origin,self.vec1,self.vec2,ray)
        if self.point_in_bounds(q1,self.origin,self.vec1,self.vec2, norm1):
            v21 = -self.v12
            new_material:Material = self.material
            #ASSUMING THIS PLANE IS X-Z (LENGTH X WIDTH)
            repeat = 4.0  # world units per repeat

            u = (q1[0] % repeat) / repeat
            v = (q1[2] % repeat ) / repeat

            k_d = self.sample_texture(u, v)
            if len(k_d) > 0:
                new_material.k_d = k_d

            T = self.vec1 / np.linalg.norm(self.vec1)
            B = self.vec2 / np.linalg.norm(self.vec2)
            N = v21 / np.linalg.norm(v21)
            TBN = np.column_stack((T, B, N))

            # sample normal
            new_norm = self.sample_normal(u, v)
            if len(new_norm) > 0:
                n = TBN @ new_norm
                n = n / np.linalg.norm(n)
                v21 = n

            hit_list.append(Hit(t1,q1,v21/np.linalg.norm(v21),new_material, self))

        q2,t2,norm2 = self.plane_intersection(self.origin + self.vec3,self.vec1,self.vec2,ray)
        if self.point_in_bounds(q2,self.origin + self.vec3,self.vec1,self.vec2,norm2):
            v12 = self.v12
            hit_list.append(Hit(t2,q2,v12/np.linalg.norm(v12),self.material, self))

        #vec2 and vec3
        q3,t3,norm3 = self.plane_intersection(self.origin,self.vec2,self.vec3,ray)
        if self.point_in_bounds(q3,self.origin,self.vec2,self.vec3,norm3):
            v32 = -self.v23
            hit_list.append(Hit(t3,q3,v32/np.linalg.norm(v32),self.material,self))

        q4,t4,norm4 = self.plane_intersection(self.origin + self.vec1,self.vec2,self.vec3,ray)
        if self.point_in_bounds(q4,self.origin + self.vec1,self.vec2,self.vec3, norm4):
            v23 = self.v23
            hit_list.append(Hit(t4,q4,v23/np.linalg.norm(v23),self.material,self))

        #vec3 and vec1
        q5,t5,norm5 = self.plane_intersection(self.origin,self.vec3,self.vec1,ray)
        if self.point_in_bounds(q5,self.origin,self.vec3,self.vec1,norm5):
            v13 = -self.v31
            hit_list.append(Hit(t5,q5,v13/np.linalg.norm(v13),self.material,self))

        q6,t6,norm6 = self.plane_intersection(self.origin + self.vec2,self.vec3,self.vec1,ray)
        if self.point_in_bounds(q6,self.origin + self.vec2,self.vec3,self.vec1,norm6):
            v31 = self.v31
            hit_list.append(Hit(t6,q6,v31/np.linalg.norm(v31),self.material,self))
            
        return hit_list
    
    def intersect(self,ray):
        return (minimum_t(self.hit_list(ray), ray))


    def sample_texture(self, u, v):
        if self.texture_map is not None and len(self.texture_map) > 0 and self.texture_map.shape[0] > 0 and self.texture_map.shape[1] > 0:
          x = u * (self.texture_map.shape[1] - 1)
          y = v * (self.texture_map.shape[0] - 1)
          

          x0 = int(np.floor(x))
          x1 = min(x0 + 1, self.texture_map.shape[1] - 1)
          y0 = int(np.floor(y))
          y1 = min(y0 + 1, self.texture_map.shape[0] - 1)

          alpha = x - x0
          beta = y - y0

          # Fetch four neighbors
          c00 = self.texture_map[y0, x0]
          c10 = self.texture_map[y0, x1]
          c01 = self.texture_map[y1, x0]
          c11 = self.texture_map[y1, x1]

          texture_color = (1 - alpha)*(1 - beta)*c00 + alpha*(1 - beta)*c10 + (1 - alpha)*beta*c01 + alpha*beta*c11
          #return texture_color/255.0
          return texture_color
        else:
          return []

    def sample_normal(self, u, v):
        if self.normal_map is not None and len(self.normal_map > 0) and self.normal_map.shape[0] > 0 and self.normal_map.shape[1] > 0:
          normal_width = self.normal_map.shape[1]
          normal_height = self.normal_map.shape[0]
          x = u * (normal_width - 1)
          y = v * (normal_height - 1)

          x0 = int(np.floor(x))
          x1 = min(x0 + 1, normal_width - 1)
          y0 = int(np.floor(y))
          y1 = min(y0 + 1, normal_height - 1)

          alpha = x - x0
          beta = y - y0

          # Fetch four neighbors
          c00 = self.normal_map[y0, x0]
          c10 = self.normal_map[y0, x1]
          c01 = self.normal_map[y1, x0]
          c11 = self.normal_map[y1, x1]

          normal_color = (1 - alpha)*(1 - beta)*c00 + alpha*(1 - beta)*c10 + (1 - alpha)*beta*c01 + alpha*beta*c11

          #change texture from (0,255) -> (0,1)
          #normal_color = normal_color / 255.0
          # map (0,1) -> (-1,1) to allow for negative normals
          normal_color = (2.0*normal_color) - 1.0
          return normal_color
        else:
         return []
       




class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material
        
    def inside(self, point):
        return (dist(point,self.center) <= (1.0001 * self.radius))
    
    def strict_inside(self,point):
        return (dist(point,self.center) <= (.999 * self.radius))
    
    def hit_list(self, ray):
        direction = ray.direction# v
        origin = ray.origin - self.center#p

        # ray has vec3 of origin and direction
        c = np.dot(origin, origin) - np.power(self.radius,2)
        
        b = 2 * np.dot(origin, direction)
        a = np.dot(direction, direction)
        discriminant = np.power(b,2) - (4*a*c)

        # Check if no sols
        if (discriminant < 0.0):
            return [no_hit]
        if (a == 0):
            if(b ==0):
                print('somethings wrong')
                return [no_hit]
            t = -c/b
            q = ray.origin + (t*ray.direction)
            normal = (q-self.center)/(np.linalg.norm(q-self.center))
            return [Hit(t, q, normal, self.material,self)]
        tplus = (- b + np.sqrt(discriminant))/(2*a)
        tminus = (- b - np.sqrt(discriminant))/(2*a)

        qplus = ray.origin + (tplus*ray.direction)
        qminus = ray.origin + (tminus*ray.direction)
        normalplus = (qplus-self.center)/(np.linalg.norm(qplus-self.center))
        normalminus = (qminus-self.center)/(np.linalg.norm(qminus-self.center))
        return [Hit(tminus,qminus,normalminus,self.material,self), Hit(tplus,qplus,normalplus,self.material,self)]
        
        
    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """

        """
        The t value along the ray that corresponds to the first valid intersection
        The location of that intersection
        The material value at the point of intersection (used for shading later)
        the normal at the point of intersection (used for shading later)

        """

        direction = ray.direction# v
        origin = ray.origin - self.center#p

        # ray has vec3 of origin and direction
        c = np.dot(origin, origin) - np.power(self.radius,2)
        
        b = 2 * np.dot(origin, direction)
        a = np.dot(direction, direction)
        discriminant = np.power(b,2) - (4*a*c)
        t=0
        # Check if no sols
        if (discriminant < 0.0):
            return no_hit
        if (a == 0):
            if(b ==0):
                print('somethings wrong')
                return no_hit
            t=-c/b
        # get earliest time for first hit, these are the intersections between the ray and shape
        tplus = (- b + np.sqrt(discriminant))/(2*a)
        tminus = (- b - np.sqrt(discriminant))/(2*a)
        t = tminus
        if (tplus < ray.start or tminus > ray.end):
            return no_hit
        elif (tminus < ray.start and tplus >= ray.start):
            t = tplus
        elif (tminus >= ray.start and tplus >= ray.start):
            t = min(tminus, tplus)
        
        if t >= ray.start and t <= ray.end:
          q = ray.origin + (t*ray.direction)
          # print(f"sphere intersection point is {q}")
          # print(f"q-c is {q-c}")
          normal = (q-self.center)/(np.linalg.norm(q-self.center))
          #print(f"normal is {normal}")
          return Hit(t,q,normal,self.material,surface=self)
        else:
          return no_hit
        
class SphereTextured(Sphere):
    def __init__(self, center, radius, material, texture_map=None, normal_map=None):
        super().__init__(center, radius, material)
        self.texture_map = texture_map 
        self.normal_map = normal_map 
    def sample_texture(self, u, v):
        if self.texture_map is not None and len(self.texture_map) > 0 and self.texture_map.shape[0] > 0 and self.texture_map.shape[1] > 0:
          U_REPEAT = 10.0
          V_REPEAT = 10.0

          u = (u * U_REPEAT) % 1.0
          v = (v * V_REPEAT) % 1.0
          x = u * (self.texture_map.shape[1] - 1)
          y = v * (self.texture_map.shape[0] - 1)
          

          x0 = int(np.floor(x))
          x1 = min(x0 + 1, self.texture_map.shape[1] - 1)
          y0 = int(np.floor(y))
          y1 = min(y0 + 1, self.texture_map.shape[0] - 1)

          alpha = x - x0
          beta = y - y0

          # Fetch four neighbors
          c00 = self.texture_map[y0, x0]
          c10 = self.texture_map[y0, x1]
          c01 = self.texture_map[y1, x0]
          c11 = self.texture_map[y1, x1]

          texture_color = (1 - alpha)*(1 - beta)*c00 + alpha*(1 - beta)*c10 + (1 - alpha)*beta*c01 + alpha*beta*c11
          #return texture_color/255.0
          return texture_color
        else:
          return []

    def sample_normal(self, u, v):
        if self.normal_map is not None and len(self.normal_map > 0) and self.normal_map.shape[0] > 0 and self.normal_map.shape[1] > 0:
          normal_width = self.normal_map.shape[1]
          normal_height = self.normal_map.shape[0]
          x = u * (normal_width - 1)
          y = v * (normal_height - 1)

          x0 = int(np.floor(x))
          x1 = min(x0 + 1, normal_width - 1)
          y0 = int(np.floor(y))
          y1 = min(y0 + 1, normal_height - 1)

          alpha = x - x0
          beta = y - y0

          # Fetch four neighbors
          c00 = self.normal_map[y0, x0]
          c10 = self.normal_map[y0, x1]
          c01 = self.normal_map[y1, x0]
          c11 = self.normal_map[y1, x1]

          normal_color = (1 - alpha)*(1 - beta)*c00 + alpha*(1 - beta)*c10 + (1 - alpha)*beta*c01 + alpha*beta*c11

          #change texture from (0,255) -> (0,1)
          #normal_color = normal_color / 255.0
          # map (0,1) -> (-1,1) to allow for negative normals
          normal_color = (2.0*normal_color) - 1.0
          return normal_color
        else:
         return []
       

    #source: https://medium.com/@Ksatese/advanced-ray-tracer-part-4-87d1c98eecff
    def sphere_uv_calculations(self,hit:Hit)->Tuple[int, int]:
      point = hit.point - self.center
      theta = np.arccos(point[1]/self.radius)
      phi = np.arctan2(point[2], point[0])
      u = (-phi + np.pi) / (2*np.pi)
      v = theta/np.pi
      return (u,v)
    
    def sphere_normal_calculations(self,hit:Hit):
        point = hit.point - self.center
        x=point[0]
        y=point[1]
        z=point[2]
        theta = np.arccos(y/self.radius)
        phi = np.arctan2(z,x)
        T=np.array([2*np.pi*z, 0 , -2*np.pi*x])
        B=np.array([np.pi*y*np.cos(phi), -self.radius*np.pi*np.sin(theta), np.pi*y*np.sin(phi)])
        T = T / np.linalg.norm(T)
        B = B / np.linalg.norm(B)
        N = hit.normal
        return np.array([np.array([T[0], B[0], N[0]]),
                        np.array([T[1], B[1], N[1]]),
                        np.array([T[2], B[2], N[2]])])
    


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material
        

    #TODO: make intersect_list

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # need to use barycentric coords
        #within time and inside triangle
        a = self.vs[0]
        b = self.vs[1]
        c = self.vs[2]
        d = ray.direction 
        p = ray.origin
        A = np.array([(a-b), (a-c), d], np.float64).T
        solved = np.linalg.solve(A,a-p)
        beta = solved[0]
        gamma = solved[1]
        t = solved[2]
        if t >= ray.start and t <= ray.end and beta > 0 and gamma > 0 and beta + gamma < 1:
          p = p + (t*d)
          normal = np.cross((b-a), (c-a))
          normal = normal / np.linalg.norm(normal)
          return Hit(t,p,normal,self.material)
        else:
          return no_hit
        
class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]), 
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        self.physical_vertical_size_of_image_plane = 2.0
        self.physical_horizontal_size_of_image_plane = self.physical_vertical_size_of_image_plane * aspect
        self.f = (self.physical_vertical_size_of_image_plane/2) / np.tan(np.deg2rad(vfov/2.0)) # assuming vertical size of image plane is 1
        
        
        # you should set this to the distance from your center of projection to the image plane
        x = np.cross((target - self.eye), up)
        x /= np.linalg.norm(x)
        
        z = target - self.eye
        z /= np.linalg.norm(z)

        y = np.cross(x,z)
        y /= np.linalg.norm(y)
        self.x = x
        self.y = y
        self.z = z

        self.M = np.array([
            np.append(x, 0),
            np.append(y, 0),
            np.append(-z, 0),
            np.append(self.eye, 1)
        ]).T;  # set this to the matrix that transforms your camera's coordinate system to world coordinates

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
 
        x_p = (-self.physical_horizontal_size_of_image_plane/2.0) + (img_point[0]*self.physical_horizontal_size_of_image_plane)
        y_p = (self.physical_vertical_size_of_image_plane/2.0) - (img_point[1]*self.physical_vertical_size_of_image_plane)
        z_p = -self.f
        direction = self.M @ vec([x_p,y_p, z_p, 0])
        origin = self.eye #already in world coords
        return Ray(origin, direction[:3])


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
          texture : (3,) -- the texture offset to potentially add at this point
        Return:
          (3,) -- the light reflected from the surface
        """
        #first one
        
        #Own Light: position,intensity
        #Ray:origin,direction,start,end  NOTE: assuming ray is vector from camera to surface.
        #Material:k_d(diffuse coefficient),k_s(specular coefficient),p(specular exponent),
        # k_m(mirror reflection coefficient),k_a(ambient coefficient)
        #Hit:t,point,normal,material
        #Scene:surfs (list of surfaces in scene),bg_color,
        # intersect function(computes first smallest t intersection with ray and scene)

        #Component 1: Nonreflection
        shading = vec([0,0,0])

        d = self.position - hit.point
        norm_d = d/(np.linalg.norm(d))

        k_d = hit.material.k_d
        k_s = hit.material.k_s
        k_m = hit.material.k_m
        p = hit.material.p
        n = hit.normal

        #ray from hit point to light source intersecting scene, 0 might have floating point errors, 
        #Note: Intersect functions need to take into account end point of ray
        light_ray = Ray(hit.point + 1e-4 * hit.normal, d, 0.0001, 1)

        #Ray(hit.point, d, 0.0001, 1) 

        v = -ray.direction/np.linalg.norm(ray.direction)

        h = (v+norm_d)/(np.linalg.norm(v+norm_d))
        

        #intersect this ray with the scene. If hits, then nonreflected component should be (0,0,0)
        if (scene.intersect(light_ray)  == no_hit):
            shading = self.intensity * ((k_d + (k_s*(pow(np.dot(n,h),p)))) * max(0,np.dot(norm_d,n)))/(np.dot(d,d))
        return shading


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        #Use: intensity and ray
        #Idea: consistant "ambient color" to shading
        if (np.any(hit.material.k_a == None)):
            return vec([0,0,0])
        else:
            return hit.material.k_a * self.intensity


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """

        best_t = np.inf
        best_hit = no_hit
        surfs = self.surfs 

        for surf in surfs:
            h = surf.intersect(ray)
            t = h.t
            if t < best_t:
                best_t = t
                best_hit = h

        return best_hit
       


MAX_DEPTH = 4


def shade(ray:Ray, hit:Hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # loop thru all ambient and point lights and sum it up
    if (depth > MAX_DEPTH):
        return vec([0,0,0])
    
    ill_tot = vec([0,0,0]) #illumination total variable
    k_d = hit.material.k_d
    k_m = hit.material.k_m

    #RN texture is uint8 0,255. we want it to just be float to be 0->1. then we want to map to -1,1 (2*texture) - 1. This is the NEW normal (could be offset or also replace entirely)

    if hit.surface.__class__ == SphereTextured or hit.surface.__class__ == ParallelpipedTextured or hit.surface.__class__ == CSG:
      u, v = hit.surface.sphere_uv_calculations(hit)
      texture_color = hit.surface.sample_texture(u, v)
      normal_color = hit.surface.sample_normal(u, v)
      if len(normal_color) > 0:
        #TBN transforms the normal from sphere normal to global normal
        TBN = hit.surface.sphere_normal_calculations(hit)
        normal_color = TBN @ normal_color
        hit.normal += normal_color 
        hit.normal /= np.linalg.norm(hit.normal)
      if len(texture_color) > 0:
        # If there is an alpha channel
        if len(texture_color) == 4:
          alpha = texture_color[3]
          hit.material.k_d = alpha * texture_color[:3] + (1-alpha)*k_d
        else:
          hit.material.k_d = texture_color         
    

    for i in lights:
        ill_tot += i.illuminate(ray,hit,scene)
    
    v = -ray.direction/np.linalg.norm(ray.direction)
    n = hit.normal
    nv_dot = np.dot(v, n)

    #PASS IN THE NEW NORMAL HERE, NEVER USE OLD NORMAL AGIN
    reflect_comp = vec([0,0,0])
    if nv_dot > 0:
          rv = (nv_dot * 2 * n) - v
          reflect_ray = Ray(hit.point,rv, 0.001, np.inf)
          reflect_intersection = scene.intersect(reflect_ray)
          if (reflect_intersection != no_hit):
               reflect_comp = k_m * shade(reflect_ray, reflect_intersection, scene, lights, depth + 1)
          else:
              reflect_comp = k_m * scene.bg_color
    
    #return ill_tot + reflect_comp
    return ill_tot

def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """

    """ from the assignment suggestion integration steps"""
    output_image = np.zeros((ny,nx,3), np.float32)
    


    for i in range(ny):
        for j in range(nx):
            # Use NORMALIZED texture coords from 0 to 1.
            ray = camera.generate_ray(vec([j / nx, i / ny]))# Generate Ray---we recommend just generating an orthographic ray to start with
            intersection = scene.intersect(ray)
          

            # set the output pixel color if an intersection is found
            # ...
            if intersection.t != np.inf:
                output_image[i][j] = shade(ray, intersection, scene, lights)
            else:
                output_image[i][j] = scene.bg_color
            #intersections.append(intersection) # might not need to keep track of this?

    return output_image

    #return np.zeros((ny,nx,3), np.float32)


class CSG:
    def __init__(self, shape=None):
        
        #added
        self.left_child = None
        self.right_child = None
        self.is_leaf = True
        #Possible opps: "u", "i", "s"
        self.opp = None 
        #sphere or triangle shape or None if combination of shapes
        self.shape = shape
        
    def add_children(self, child_1, child_2, opp):
        if self.is_leaf:
          self.left_child = child_1
          self.right_child = child_2
          self.is_leaf = False
          self.opp = opp
    
    def inside(self, point):
        if self.shape == None:
            if self.opp == "u":
                return self.left_child.inside(point) or self.right_child.inside(point)
            if self.opp == "i":
                return self.left_child.inside(point) and self.right_child.inside(point)
            #NOTE: For subtraction, assuming that left_child is the object we're keeping
            if self.opp == "s": 
                return self.left_child.inside(point) and not self.right_child.inside(point)
        else:
            return self.shape.inside(point)
        
    def strict_inside(self,point):
        if self.shape == None:
            if self.opp == "u":
                return self.left_child.strict_inside(point) or self.right_child.strict_inside(point)
            if self.opp == "i":
                return self.left_child.strict_inside(point) and self.right_child.strict_inside(point)
            #NOTE: For subtraction, assuming that left_child is the object we're keeping
            if self.opp == "s": 
                return self.left_child.strict_inside(point) and not self.right_child.strict_inside(point)
        else:
            return self.shape.strict_inside(point)
    
    def hit_list(self, ray):
        if self.is_leaf:
            return self.shape.hit_list(ray)
        else:
            #lst = self.left_child.hit_list(ray) + self.right_child.hit_list(ray)
            final_lst = []
            for i in self.left_child.hit_list(ray):
              if not i == no_hit:
                  a = self.left_child.inside(i.point)
                  a_s = self.left_child.strict_inside(i.point)
                  b = self.right_child.inside(i.point)   
                  b_s = self.right_child.strict_inside(i.point)         
                  if self.opp == "u":
                      if not a_s and not b_s:
                          final_lst.append(i)
                  elif self.opp == "i":
                      if (a and b):
                          final_lst.append(i)
                  elif self.opp == "s":
                      if (not b):
                          final_lst.append(i)

            for i in self.right_child.hit_list(ray):
              if not i == no_hit:
                  a = self.left_child.inside(i.point)
                  a_s = self.left_child.strict_inside(i.point)
                  b = self.right_child.inside(i.point)  
                  b_s = self.right_child.strict_inside(i.point)           
                  if self.opp == "u":
                      if not a_s and not b_s:
                          final_lst.append(i)
                  elif self.opp == "i":
                      if (a and b):
                          final_lst.append(i)
                  elif self.opp == "s":
                      if (a_s):
                          new_hit = Hit(i.t,i.point,(-1) * i.normal,i.material, i.surface)
                          final_lst.append(new_hit)
                          
            return final_lst
        
    def intersect(self,ray):
        return (minimum_t(self.hit_list(ray), ray))
        

def minimum_t(hit_list, ray):
    best_t = ray.end
    best_hit = no_hit
    for i in hit_list:
        if i.t > ray.start and i.t < best_t:
            best_t = i.t
            best_hit = i
    return best_hit