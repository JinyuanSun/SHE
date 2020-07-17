import numpy as np
import math

AAdata = {
    "SER": {
        "l_CB_OG": 1.417,
        "a_CA_CB_OG": 110.773
    },
    "GLU": {
        "l_CB_CG": 1.52,
        "a_CA_CB_CG": 113.82,
        "l_CG_CD": 1.52,
        "a_CB_CG_CD": 113.31,
        "l_CD_OE1" : 1.25,
        "a_CG_CD_OE1" : 119.02,
        "l_CD_OE2" : 1.25,
        "a_CG_CD_OE2" : 118.08
    },
    "HIS": {
        "l_CB_CG" : 1.49,
        "a_CA_CB_CG" : 113.74,
        "l_CG_ND1" : 1.38,
        "a_CB_CG_ND1" : 122.85,
        "l_CG_CD2" : 1.35,
        "a_CB_CG_CD2" : 130.61,
        "l_ND1_CE1" : 1.32,
        "a_CG_ND1_CE1" : 108.5,
        "l_CD2_NE2" : 1.35,
        "a_CG_CD2_NE2" : 108.5
    }
}

SERCHI = [65,-64,179]

def rotamer2pdb(A,B,C,chi,l,ang):

    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    #normlaized 
    BA = np.array(A-B)
    BC = np.array(C-B)
    v = np.cross(BA,BC)
    axis = np.array(BC)
    theta = chi/180*math.pi

    #print(np.dot(rotation_matrix(axis, theta), v)) 
    beta = np.dot(rotation_matrix(axis, theta), v)
    CD = np.dot(rotation_matrix(beta, ang/180*math.pi),np.array(B-C))
    CD = CD/np.linalg.norm(CD)*l
    D = CD+C
    return D


def SER(N, CA, CB, chi1):
    # CB
    # OG
    l = AAdata["SER"]["l_CB_OG"]
    ang = AAdata["SER"]["a_CA_CB_OG"]
    OG = rotamer2pdb(N, CA, CB, chi1, l, ang)
    # return coord
    return {"OG":OG}


def GLU(N, CA, CB, chi1,chi2,rot3):
    # CB
    # CG
    # CD
    # OE1
    # OE2
    
    l_CB_CG = AAdata["GLU"]["l_CB_CG"]
    a_CA_CB_CG = AAdata["GLU"]["a_CA_CB_CG"]
    CG = rotamer2pdb(N, CA, CB, chi1, l_CB_CG, a_CA_CB_CG)
    
    l_CG_CD = AAdata["GLU"]["l_CG_CD"]
    a_CB_CG_CD = AAdata["GLU"]["a_CB_CG_CD"]
    CD = rotamer2pdb(CA, CB, CG, chi2, l_CG_CD, a_CB_CG_CD)
    
    l_CD_OE1 = AAdata["GLU"]["l_CD_OE1"]
    a_CG_CD_OE1 = AAdata["GLU"]["a_CG_CD_OE1"]
    OE1 = rotamer2pdb(CB, CG, CD, rot3, l_CD_OE1, a_CG_CD_OE1)
    
    l_CD_OE2 = AAdata["GLU"]["l_CD_OE2"]
    a_CG_CD_OE2 = AAdata["GLU"]["a_CG_CD_OE2"]
    OE2 = rotamer2pdb(CB, CG, CD, 180+rot3, l_CD_OE2, a_CG_CD_OE2)
    # return coord
    return {"CG":CG,"CD":CD,"OE1":OE1,"OE2":OE2}

def HIS(N, CA, CB, chi1, rot2):
    # N
    # CA
    # C
    # O
    # CB
    # CG
    # ND1
    # CD2
    # CE1
    # NE2
    l_CB_CG = AAdata["HIS"]["l_CB_CG"]
    a_CA_CB_CG = AAdata["HIS"]["a_CA_CB_CG"]
    CG = rotamer2pdb(N, CA, CB, chi1, l_CB_CG, a_CA_CB_CG)

    l_CG_ND1 = AAdata["HIS"]["l_CG_ND1"]
    a_CB_CG_ND1 = AAdata["HIS"]["a_CB_CG_ND1"]
    ND1 = rotamer2pdb(CA, CB, CG, rot2, l_CG_ND1, a_CB_CG_ND1)

    l_CG_CD2 = AAdata["HIS"]["l_CG_CD2"]
    a_CB_CG_CD2 = AAdata["HIS"]["a_CB_CG_CD2"]
    CD2 = rotamer2pdb(CA, CB, CG, 180 + rot2, l_CG_CD2, a_CB_CG_CD2)

    l_ND1_CE1 = AAdata["HIS"]["l_ND1_CE1"]
    a_CG_ND1_CE1 = AAdata["HIS"]["a_CG_ND1_CE1"]
    CE1 = rotamer2pdb(CB, CG, ND1, 180, l_ND1_CE1, a_CG_ND1_CE1)

    l_CD2_NE2 = AAdata["HIS"]["l_CD2_NE2"]
    a_CG_CD2_NE2 = AAdata["HIS"]["a_CG_CD2_NE2"]
    NE2 = rotamer2pdb(CB, CG, CD2, 180, l_CD2_NE2, a_CG_CD2_NE2)
    return {"CG": CG, "ND1": ND1, "CD2": CD2, "CE1": CE1, "NE2": NE2}


def cal_dihedral(p):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def pdb2dict(PDBfile,mainchain=0):
    pdb = open(PDBfile)
    pdbdict = {}
    for line in pdb:
        if line.startswith("ATOM"):
            if mainchain == 0:
        #for line in pdb:
            #if line.startswith("ATOM"):
                record_name = line[0:6].replace(" ","")
                atom_num = line[6:11].replace(" ","")
                atom = line[12:16].replace(" ","")
                altLoc = line[16].replace(" ","")
                res = line[17:20].replace(" ","")
                chainid = line[21].replace(" ","")
                resseq = int(line[22:26].replace(" ",""))
                icode = line[26].replace(" ","")
                x = float(line[30:38].replace(" ",""))
                y = float(line[38:46].replace(" ",""))
                z = float(line[46:54].replace(" ",""))
                occ = line[54:60].replace(" ","")
                tfactor = line[60:66].replace(" ","")
                element = line[76:78].replace(" ","")
                charge = line[78:80].replace(" ","")

                try:

                    pdbdict[chainid][resseq][atom] = np.array([x,y,z])
                except KeyError:
                    try:
                        pdbdict[chainid][resseq] = {}
                        pdbdict[chainid][resseq]["res"] = res
                        pdbdict[chainid][resseq][atom] = np.array([x,y,z])
                    except KeyError:
                        pdbdict[chainid] = {resseq:{atom:np.array([x,y,z])}}
                        #pdbdict[chainid][resseq] = {}
                        
                        
    #return pdbdict
            if mainchain == 1:
                if line.startswith("ATOM") and line[12:16].replace(" ","") in ["N", "O", "C", "CA" ,"CB"]:
                    record_name = line[0:6].replace(" ","")
                    atom_num = line[6:11].replace(" ","")
                    atom = line[12:16].replace(" ","")
                    altLoc = line[16].replace(" ","")
                    res = line[17:20].replace(" ","")
                    chainid = line[21].replace(" ","")
                    resseq = int(line[22:26].replace(" ",""))
                    icode = line[26].replace(" ","")
                    x = float(line[30:38].replace(" ",""))
                    y = float(line[38:46].replace(" ",""))
                    z = float(line[46:54].replace(" ",""))
                    occ = line[54:60].replace(" ","")
                    tfactor = line[60:66].replace(" ","")
                    element = line[76:78].replace(" ","")
                    charge = line[78:80].replace(" ","")

                    try:

                        pdbdict[chainid][resseq][atom] = np.array([x,y,z])
                    except KeyError:
                        try:
                            pdbdict[chainid][resseq] = {}
                            pdbdict[chainid][resseq]["res"] = res
                            pdbdict[chainid][resseq][atom] = np.array([x,y,z])
                        except KeyError:
                            pdbdict[chainid] = {resseq:{atom:np.array([x,y,z])}}
    return pdbdict