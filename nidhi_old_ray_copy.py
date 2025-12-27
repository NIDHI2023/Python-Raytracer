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
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material
        self.surface =surface

# Value to represent absence of an intersection
no_hit = Hit(np.inf)


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
        # get earliest time for first hit
        tplus = (- b + np.sqrt(discriminant))/(2*a)
        tminus = (- b - np.sqrt(discriminant))/(2*a)
        t = tminus
        if (tplus < ray.start or tminus > ray.end):
            return no_hit
        elif (tminus < ray.start and tplus >= ray.start):
            t = tplus
        elif (tminus >= ray.start and tplus >= ray.start):
            t = min(tminus, tplus)
        
        # TODO SHAPE- A4 implement this function
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
        #N = np.cross(T,B)
        return np.array([np.array([T[0], B[0], N[0]]),
                        np.array([T[1], B[1], N[1]]),
                        np.array([T[2], B[2], N[2]])])

    
    def sphere_normal_calculations1(self,hit):
      normal = hit.normal
    # normalize norm in this case
      normal = normal / np.linalg.norm(normal)
      up = np.array([0.0, 1.0, 0.0]) #THIS CHANGES THE MAPPING A LOT
      if abs(normal[1]) > 0.999:
          up = np.array([1.0, 0.0, 0.0])
      tangent = np.cross(up, normal)
      tangent /= np.linalg.norm(tangent)
      bitangent = np.cross(normal, tangent)
      bitangent /= np.linalg.norm(bitangent)
      return tangent, bitangent


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # TODO SHAPE- A4 implement this function
        # need to use barycentric coords
        #within time and inside triangle
        a = self.vs[0]
        b = self.vs[1]
        c = self.vs[2]
        d = ray.direction # TODO: DOES THIS NEED TO BE NORMALIZED??
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
        

class Plane:

  def __init__(self, n, p, material, width=None,height=None):
      """Create a triangle from the given vertices.

      Parameters:
        n -- the normal vector of the plane (assuming positive means facing "up")
        p (3,) -- any point (x,y,z) on the plane. if width and height are given, this is the center of the plane. used to solve ray equation
        material : Material -- the material of the surface
        width: float -- should be > 0. width of the plane (x coords)
        height: float -- should be >0. height of plane (y coords)


        Finite plane currently only works in x-z plane..
        TODO: find what two vectors cross product would make the normal vector
      """
      self.norm = n
      self.point = p
      self.material = material
      self.w = width
      self.h = height

  def intersect(self, ray):
      """Computes the intersection between a ray and this triangle, if it exists.

      Parameters:
        ray : Ray -- the ray to intersect with the triangle
      Return:
        Hit -- the hit data
      """
      # TODO SHAPE- A4 implement this function
      # need to use barycentric coords
      #within time and inside triangle
      a = self.norm[0]
      b = self.norm[1]
      c = self.norm[2]
      d = -(a*self.point[0] + b*self.point[1] + c*self.point[2])
      v = np.append(ray.direction,1)
      p = np.append(ray.origin,1)
      A = np.array([a,b,c,d])

      t = - np.dot(A,p) / np.dot(A,v)
      p = p + (t*v)
      if (self.w is not None and (p[0] > self.point[0] + self.w or p[0] < self.point[0] - self.w)):
          return no_hit
      
      if (self.h is not None and (p[2] > self.point[2] + self.h or p[2] < self.point[2] - self.h)):
          return no_hit
          
      if t >= ray.start and t <= ray.end:
        return Hit(t,p[:3],self.norm/np.linalg.norm(self.norm),self.material, surface=self)
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
        # TODO SHAPE - A4 implement this function
            #   # TEMPORARY HARDCODE DIRECTION FOR ORTHONORAML - THIS SEEMS TO WORK?
        # # origin_coord = vec([(img_point[0] + self.eye[0])-0.5*w, (img_point[1]+ self.eye[1])-0.5*h,self.eye[2]])
        # # direction = vec([0,0,-1])

        # origin_coord = self.M @ vec([self.eye[0], self.eye[1],self.eye[2],1])
        # direction = self.M @ vec([(img_point[0] + self.eye[0])-0.5*w, (img_point[1]+ self.eye[1])-0.5*h,-self.f, 1])
        # # normalize?
        # direction /= np.linalg.norm(direction)
 
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
        # TODO LIGHT- A4 implement this function
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
        # TODO LIGHT - A4 implement this function
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
        # TODO SHAPE - A4 implement this function
        hits = []

        for i in self.surfs:
            hits.append(i.intersect(ray))
        
        best_t = np.inf
        min_h = no_hit
        for i in hits:
            if (i.t < best_t):
                best_t = i.t
                min_h = i
        return min_h


MAX_DEPTH = 4

def shade(ray:Ray, hit:Hit, scene, lights, normal, texture, depth=0):
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
    # TODO LIGHT - A4 implement this function
    # loop thru all ambient and point lights and sum it up
    if (depth > MAX_DEPTH):
        return vec([0,0,0])
    
    ill_tot = vec([0,0,0]) #illumination total variable
    n = hit.normal

    #RN texture is uint8 0,255. we want it to just be float to be 0->1. then we want to map to -1,1 (2*texture) - 1. This is the NEW normal (could be offset or also replace entirely)

    if hit.surface.__class__ == SphereTextured:
      normal_color, texture_color = calculate_sphere_texture_coords(hit, normal, texture)
      if len(normal_color) > 0:
        #change texture from (0,255) -> (0,1)
        normal_color = normal_color / 255.0
        # map (0,1) -> (-1,1) to allow for negative normals
        normal_color = (2.0*normal_color) - 1.0
        #TBN transforms the normal from sphere normal to global normal
        TBN = hit.surface.sphere_normal_calculations(hit)
        #this part is for sphere_normal_calculations1
        #TBN = np.column_stack((tangent, bitangent, n/np.linalg.norm(n)))
        #Normalizing before matrix doesnt change much
        #texture = texture/ np.linalg.norm(texture)
        normal_color = TBN @ normal_color
        normal_color /= np.linalg.norm(normal_color)
        hit.normal += normal_color # TRY REPLACEING OR ADDING - Replacing is much harsher so adding seems better
        hit.normal /= np.linalg.norm(hit.normal)
      if len(texture_color) > 0:
        texture_color = texture_color / 255.0
        # If there is an alpha channel
        if len(texture_color) == 4:
          alpha = texture_color[3]
          hit.material.k_d = alpha * texture_color[:3] + (1-alpha)*hit.material.k_d
        else:
          hit.material.k_d = texture_color
    for i in lights:
        ill_tot += i.illuminate(ray,hit,scene)


    
    k_m = hit.material.k_m
    
    v = -ray.direction/np.linalg.norm(ray.direction)

    #PASS IN THE NEW NORMAL HERE, NEVER USE OLD NORMAL AGIN
    rv = (np.dot(v,hit.normal) * 2 * hit.normal) - v
    reflect_comp = vec([0,0,0])
    if np.dot(v,hit.normal) > 0:
          reflect_ray = Ray(hit.point,rv, 0.001, np.inf)
          if (scene.intersect(reflect_ray) != no_hit):
               reflect_comp = k_m * shade(reflect_ray, scene.intersect(reflect_ray), scene, lights, normal, texture, depth + 1)
          else:
              reflect_comp = k_m * scene.bg_color
    
    return ill_tot + reflect_comp


def calculate_sphere_texture_coords(intersection:Hit, normal:np.ndarray, texture:np.ndarray):
    
  #Spherical UV mapping
  (u,v) = intersection.surface.sphere_uv_calculations(intersection)

  #Bilinear interpolation to map uv->ij
  if len(normal > 0) and normal.shape[0] > 0 and normal.shape[1] > 0:
    normal_width = normal.shape[1]
    normal_height = normal.shape[0]
    x = u * (normal_width - 1)
    y = v * (normal_height - 1)

    x0 = int(np.floor(x))
    x1 = min(x0 + 1, normal_width - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, normal_height - 1)

    alpha = x - x0
    beta = y - y0

    # Fetch four neighbors
    c00 = normal[y0, x0]
    c10 = normal[y0, x1]
    c01 = normal[y1, x0]
    c11 = normal[y1, x1]

    norm_color = (1 - alpha)*(1 - beta)*c00 + alpha*(1 - beta)*c10 + (1 - alpha)*beta*c01 + alpha*beta*c11
  else:
    norm_color = []

  #Bilinear interpolation for texture coords to map uv->ij
  if len(texture) > 0 and texture.shape[0] > 0 and texture.shape[1] > 0:
    U_REPEAT = 4.0
    V_REPEAT = 4.0

    u = (u * U_REPEAT) % 1.0
    v = (v * V_REPEAT) % 1.0
    x = u * (texture.shape[1] - 1)
    y = v * (texture.shape[0] - 1)
    

    x0 = int(np.floor(x))
    x1 = min(x0 + 1, texture.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, texture.shape[0] - 1)

    alpha = x - x0
    beta = y - y0

    # Fetch four neighbors
    c00 = texture[y0, x0]
    c10 = texture[y0, x1]
    c01 = texture[y1, x0]
    c11 = texture[y1, x1]

    texture_color = (1 - alpha)*(1 - beta)*c00 + alpha*(1 - beta)*c10 + (1 - alpha)*beta*c01 + alpha*beta*c11
  else:
    texture_color = []   

  return (norm_color, texture_color) 


def render_image(camera, scene, lights, nx, ny, normal,texture):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    # TODO LIGHT- A4 implement this function

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
                # Separating functionality
                if intersection.surface.__class__ == SphereTextured:

                  output_image[i][j] = shade(ray, intersection, scene, lights, normal, texture)
                else:
                  output_image[i][j] = shade(ray, intersection, scene, lights, np.array([]),([]))
            else:
                output_image[i][j] = scene.bg_color
            #intersections.append(intersection) # might not need to keep track of this?



    return output_image

    #return np.zeros((ny,nx,3), np.float32)
