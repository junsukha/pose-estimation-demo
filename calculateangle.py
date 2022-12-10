import math
import numpy as np
def calculate_angle(a,b,c):
    # a = np.array(a) # Frist
    # b = np.array(b) # Mid
    # c = np.array(c) # End
    
    # ba = a - b
    # bc = c - b
    
    # v1mag = math.sqrt(ba[0]**2 + ba[1]**2+ba[2]**2)
    # v1norm = [ba[0]/v1mag, ba[1]/v1mag, ba[2]/v1mag]
    
    # v2mag = math.sqrt(bc[0]**2 + bc[1]**2+bc[2]**2)
    # v2norm = [bc[0]/v2mag, bc[1]/v2mag, bc[2]/v2mag]
    
    # res = v1norm[0]*v2norm[0] + v1norm[1]*v2norm[1] + v1norm[2]*v2norm[2]
    # angle = np.arccos(res)
    
    

    # cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # angle = np.arccos(cosine_angle)
    
    # angle = np.abs(angle * 180.0/np.pi)
    
    # if angle > 180.0:
    #     angle = 360 - angle
        
    ## Using only x,y coordniates
    a = np.array(a) # Frist
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1] - b[1], c[0] -b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

