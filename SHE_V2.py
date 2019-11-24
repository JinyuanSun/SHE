import itertools
import os
import sys
import Bio
import numpy as np
import time
from PeptideBuilder import *
start = time.time()

def _3_2_1(x):
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    y = d[x]
    return y


# calculate diangle psi and phi
def dihedral3(p):
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array([np.cross(v, b[1]) for v in [b[0], b[2]]])
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
    return np.degrees(np.arccos(v[0].dot(v[1])))


# get CA pairs within a custom range
def _pair_map(atom_list, min, max):
    l = len(atom_list)
    pair_dict = {}
    i = 0
    n = 0
    for i in range(l):
        n = i + 1
        lst = [i + 1]
        for n in range(i, l):
            d = atom_list[i] - atom_list[n]

            if min < d < max:
                lst.append(n + 1)
            n = n + 1
        if len(lst) > 2:
            pair_dict[i + 1] = lst
        i = i + 1
    return pair_dict


# phi_psi_dict
def _main_diangle_dict(res_list):
    n = 0
    di_dic = {}
    for x in res_list:
        try:
            c1 = res_list[n]["C"].get_vector()
            n2 = res_list[n + 1]["N"].get_vector()
            ca2 = res_list[n + 1]["CA"].get_vector()
            c2 = res_list[n + 1]["C"].get_vector()
            n3 = res_list[n + 2]["N"].get_vector()
            c3 = res_list[n + 2]["C"].get_vector()
            n = n + 1
            phi = np.array(
                [[c1[0], c1[1], c1[2]], [n2[0], n2[1], n2[2]], [ca2[0], ca2[1], ca2[2]], [c2[0], c2[1], c2[2]]])
            psi = np.array(
                [[n2[0], n2[1], n2[2]], [ca2[0], ca2[1], ca2[2]], [c2[0], c2[1], c2[2]], [n3[0], n3[1], n3[2]]])
            id = str(n) + ":" + str(n + 1) + ":" + str(n + 2)
            _phi = int(round((-dihedral3(phi)), -1))
            _psi = int(round((-dihedral3(psi)), -1))
            di_dic[id] = (_phi, _psi)
        except IndexError:
            break
        except KeyError:
            break
    return di_dic


def GLU_backbone_dic():
    lib = open("GLU.simple.lib")
    # print(lib)
    GLU_lib_dic = {}
    for line in lib:
        lst = line.replace("\n", "").split()
        # print(lst[1])
        try:
            psi_phi = (int(lst[1]), int(lst[2]))
            pob = float(lst[8])
            if pob > 0.0001:
                key = (psi_phi, pob)
                rot_list = [float(lst[9]), float(lst[10]), float(lst[11])]
                GLU_lib_dic[key] = rot_list
        except IndexError:
            print(lst)
        except ValueError:
            continue

    return GLU_lib_dic


def HIS_backbone_dic():
    lib = open("HIS.simple.lib")
    HIS_lib_dic = {}
    for line in lib:
        lst = line.replace("\n", "").split()
        try:
            psi_phi = (int(lst[1]), int(lst[2]))
            pob = float(lst[8])
            if pob > 0.0001:
                key = (psi_phi, pob)
                rot_list = [float(lst[9]), float(lst[10])]
                HIS_lib_dic[key] = rot_list
        except IndexError:
            print(lst)
        except ValueError:
            continue

    return HIS_lib_dic


def SER_backbone_dic():
    lib = open("SER.simple.lib")
    SER_lib_dic = {}
    for line in lib:
        lst = line.replace("\n", "").split()
        try:
            psi_phi = (int(lst[1]), int(lst[2]))
            pob = float(lst[8])
            if pob > 0.0001:
                key = (psi_phi, pob)
                rot_list = [float(lst[9])]
                SER_lib_dic[key] = rot_list
        except IndexError:
            print(lst)
        except ValueError:
            print(lst)

    return SER_lib_dic


def backbone_sub_dic(position, lib_dic):
    sub_dic = {}
    p = int(position)
    key = str(p - 1) + ":" + str(p) + ":" + str(p + 1)
    diangle = di_dic.get(key)
    phi = diangle[0]
    psi_im1 = diangle[1]
    # return diangle
    for k in lib_dic:
        if k[0] == diangle:
            sub_dic[k[1]] = lib_dic[k]

    return sub_dic, phi, psi_im1


def mut(position, mut_AA):
    res = res_list[position - 1]
    out_P_S_lst = []
    ca = res["CA"]
    C = res["C"]
    N = res["N"]
    CA_coord = ca.get_coord()
    C_coord = C.get_coord()
    N_coord = N.get_coord()
    # print(ca_coor) check
    if mut_AA == "SER":
        # lib_name = mut_AA+"_lib_dic"
        sub_dic = backbone_sub_dic(position, SER_lib_dic)[0]
        phi = backbone_sub_dic(position, SER_lib_dic)[1]
        psi_im1 = backbone_sub_dic(position, SER_lib_dic)[2]
    if mut_AA == "HIS":
        # lib_name = mut_AA+"_lib_dic"
        sub_dic = backbone_sub_dic(position, HIS_lib_dic)[0]
        phi = backbone_sub_dic(position, HIS_lib_dic)[1]
        psi_im1 = backbone_sub_dic(position, HIS_lib_dic)[2]
    if mut_AA == "GLU":
        # lib_name = mut_AA+"_lib_dic"
        sub_dic = backbone_sub_dic(position, GLU_lib_dic)[0]
        phi = backbone_sub_dic(position, GLU_lib_dic)[1]
        psi_im1 = backbone_sub_dic(position, GLU_lib_dic)[2]
    # print(sub_dic)
    for k in sub_dic:
        rot = sub_dic[k]
        geo = Geometry.geometry(_3_2_1(mut_AA))
        geo.phi = phi
        geo.psi_im1 = psi_im1
        geo.inputRotamers(rot)
        structure = PeptideBuilder.initialize_res(geo, CA_coord, C_coord, N_coord)
        out = Bio.PDB.PDBIO()
        out.set_structure(structure)
        out_P_S_lst.append((out, structure))
    return out_P_S_lst


# print(mut(2,"SER","OG"))

k = 0.3


def SER_HIS_GLU_pair(pSER, pHIS, pGLU, k):
    try:
        n = 0
        SER_HIS_GLU_dict = {}
        sSER = mut(pSER, "SER")
        sHIS = mut(pHIS, "HIS")
        sGLU = mut(pGLU, "GLU")
        # get atoms
        for SER in sSER:
            OG = SER[1][0]["A"][1]["OG"]
            OG_v = OG.get_vector()
            SER_out = SER[0]
            for HIS in sHIS:
                HIS_out = HIS[0]
                NE2 = HIS[1][0]["A"][1]["NE2"]
                NE2_v = NE2.get_vector()
                OG_NE2 = OG - NE2
                if abs(1 - OG_NE2 / 3.1) < k:
                    # print(OG_NE2)
                    CB = SER[1][0]["A"][1]["CB"]
                    CB_v = CB.get_vector()
                    OG_CB_NE2 = calc_angle(OG_v, CB_v, NE2_v)
                    if abs(1 - 57.3 * OG_CB_NE2 / 76.7) < k:
                        # print(OG_CB_NE2)
                        CE1 = HIS[1][0]["A"][1]["CE1"]
                        CE1_v = CE1.get_vector()
                        ND1 = HIS[1][0]["A"][1]["ND1"]
                        ND1_v = ND1.get_vector()
                        CB_NE2_CE1 = calc_angle(CB_v, NE2_v, CE1_v)
                        if abs(1 - 57.3 * CB_NE2_CE1 / 102.5) < k:
                            # OG_CB_NE2_CE1 = calc_dihedral(OG_v, CB_v, NE2_v, CE1_v)
                            CB_NE2_CE1_ND1 = calc_dihedral(CB_v, NE2_v, CE1_v, ND1_v)
                            if abs(1 - 57.3 * CB_NE2_CE1_ND1 / 151.1) < k:
                                # print(CB_NE2_CE1)
                                for GLU in sGLU:
                                    GLU_out = GLU[0]
                                    OE1 = GLU[1][0]["A"][1]["OE1"]
                                    OE1_v = OE1.get_vector()
                                    OE1_ND1 = OE1 - ND1
                                    if abs(1 - OE1_ND1 / 2.7) < k:
                                        CE1_ND1_OE1 = calc_angle(CE1_v, ND1_v, OE1_v)
                                        if abs(1 - 57.3 * CE1_ND1_OE1 / 126.8) < k:
                                            NE2_CE1_ND1_OE1 = calc_dihedral(NE2_v, CE1_v, ND1_v, OE1_v)
                                            if abs(1 - 57.3 * NE2_CE1_ND1_OE1 / -173.3) < k:
                                                # print(1,OE1_ND1)
                                                CD = GLU[1][0]["A"][1]["CD"]
                                                CD_v = CD.get_vector()
                                                OE2 = GLU[1][0]["A"][1]["OE2"]
                                                OE2_v = OE2.get_vector()
                                                CE1_ND1_OE1_CD = calc_dihedral(CE1_v, ND1_v, OE1_v, CD_v)
                                                if abs(1 - 57.3 * CE1_ND1_OE1_CD / 49.3) < k:
                                                    # ND1_OE1_CD_OE2 = calc_dihedral(ND1_v, OE1_v, CD_v, OE2_v)
                                                    OG_CE1_OE1 = calc_angle(OG_v, CE1_v, OE1_v)
                                                    if abs(1 - 57.3 * OG_CE1_OE1 / 160) < k:
                                                        # print(57.3 * CE1_ND1_OE1_CD)
                                                        n = n + 1
                                                        SER_out_name = "SER" + "_" + str(pSER)
                                                        HIS_out_name = "HIS" + "_" + str(pHIS)
                                                        GLU_out_name = "GLU" + "_" + str(pGLU)
                                                        SER_out.save(SER_out_name + "_" + str(n))
                                                        HIS_out.save(HIS_out_name + "_" + str(n))
                                                        GLU_out.save(GLU_out_name + "_" + str(n))
                                                        # print("SER,HIS,GLU", ab, ac, bc)
                                                        # new_lst.append(("SER",SER,"HIS",HIS,"GLU",GLU))
                                                        os.system(
                                                            "cat " + pdb_filename + " " + SER_out_name + "_" + str(
                                                                n) + " " + HIS_out_name + "_" + str(n) +
                                                            " " + GLU_out_name + "_" + str(
                                                                n) + " > " + pdb_filename.replace(".pdb", "_") +
                                                            SER_out_name +"_"+ HIS_out_name +"_"+ GLU_out_name + "_" + str(
                                                                n) + ".pdb")
                                                        SER_HIS_GLU_dict[
                                                            SER_out_name + "_" + HIS_out_name + "_" + GLU_out_name] = n
    except TypeError:
        print(pSER, pHIS, pGLU)

    return SER_HIS_GLU_dict

    # condition:


def _run(pairs):
    res_cob_list = list(itertools.combinations(pairs, 3))
    pairs_out_lst = []
    for x in res_cob_list:
        CA1 = atom_list[x[1] - 1]
        CA2 = atom_list[x[2] - 1]
        d = CA1 - CA2
        # print(d)
        if 3 < d < 10:
            pSER = x[0]
            pHIS = x[1]
            pGLU = x[2]
            SER_HIS_GLU_lst = SER_HIS_GLU_pair(pSER, pHIS, pGLU, k)
            pairs_out_lst.append(SER_HIS_GLU_lst)

            pSER = x[0]
            pHIS = x[2]
            pGLU = x[1]

            SER_HIS_GLU_lst = SER_HIS_GLU_pair(pSER, pHIS, pGLU, k)
            pairs_out_lst.append(SER_HIS_GLU_lst)

            pSER = x[1]
            pHIS = x[2]
            pGLU = x[0]

            SER_HIS_GLU_lst = SER_HIS_GLU_pair(pSER, pHIS, pGLU, k)
            pairs_out_lst.append(SER_HIS_GLU_lst)

            pSER = x[1]
            pHIS = x[0]
            pGLU = x[2]

            SER_HIS_GLU_lst = SER_HIS_GLU_pair(pSER, pHIS, pGLU, k)
            pairs_out_lst.append(SER_HIS_GLU_lst)

            pSER = x[2]
            pHIS = x[1]
            pGLU = x[0]

            SER_HIS_GLU_lst = SER_HIS_GLU_pair(pSER, pHIS, pGLU, k)
            pairs_out_lst.append(SER_HIS_GLU_lst)

            pSER = x[2]
            pHIS = x[0]
            pGLU = x[1]

            SER_HIS_GLU_lst = SER_HIS_GLU_pair(pSER, pHIS, pGLU, k)
            pairs_out_lst.append(SER_HIS_GLU_lst)
    return pairs_out_lst


# scan_lst = ["SER_OG","HIS_NE2","HIS_ND1","GLU_OE1"]
# constraint:
# S102_OG_H242_NE2 = 3.1190376
# H242_ND1_E126_OE1 = 2.7223976
# S102_OG_E126_OE1 = 7.308651
# interface
if len(sys.argv) < 2:
    print("Error, not enough arguments!\nUsage:python3 SHE.py you_pdbfile\n\n\ne.g.: python3 SHE.py 3wzl.pdb\n\n"
          "Before the first trail, you may like to run python3 SHE.py demo, this is fast and can let you know what "
          "is the outcome looks like\n\nPlease notice that prefrom a complete scan will take hours on a 4 GHz CPU!")
try:
    if sys.argv[1] == "demo":
        pdb_filename = "3wzl.pdb"
        p = PDBParser(PERMISSIVE=1)
        s = p.get_structure("X", pdb_filename)

        model = s[0]
        chain = model["A"]
        residues = model.get_residues()
        atom_list = []
        res_list = []
        for residue in residues:
            if is_aa(residue) == True:
                CA = residue['CA']
                atom_list.append(CA)
                res_list.append(residue)

        pair_dict = _pair_map(atom_list, 3, 10)

        di_dic = _main_diangle_dict(res_list)

        # Build backbone dependent dict of each amino acid
        SER_lib_dic = SER_backbone_dic()
        # print(SER_lib_dic) check
        HIS_lib_dic = HIS_backbone_dic()
        GLU_lib_dic = GLU_backbone_dic()
        test = (102, 242, 126)
        # for xlst in pair_dict:
        pairs_out_lst = _run(test)
        for x in pairs_out_lst:
            if x != {}:
                print(x)
        end = time.time()
        print("This job took "+str(end-start)+" seconds!")
    if sys.argv[1] != "demo":
        pdb_filename = sys.argv[1]
        p = PDBParser(PERMISSIVE=1)
        s = p.get_structure("X", pdb_filename)

        model = s[0]
        chain = model["A"]
        residues = model.get_residues()
        atom_list = []
        res_list = []
        for residue in residues:
            if is_aa(residue) == True:
                CA = residue['CA']
                atom_list.append(CA)
                res_list.append(residue)

        pair_dict = _pair_map(atom_list, 3, 10)

        di_dic = _main_diangle_dict(res_list)

        # Build backbone dependent dict of each amino acid
        SER_lib_dic = SER_backbone_dic()
        # print(SER_lib_dic) check
        HIS_lib_dic = HIS_backbone_dic()
        GLU_lib_dic = GLU_backbone_dic()

        # print(pair_dict[2])
        # test = (102,242,126)

        for xlst in pair_dict:
            # pair_dict[xlst]
            ofile = open(pdb_filename + "_SHE.out", "a+")
            pairs_out_lst = _run(pair_dict[xlst])
            for x in pairs_out_lst:
                if x != {}:
                    print(x, file=ofile)
        end = time.time()
        print("This job took "+str(end-start)+" seconds!")


except IndexError:
    print("Error, not enough arguments!\nUsage:python3 SHE.py you_pdbfile\n\n\ne.g.: python3 SHE.py 3wzl.pdb\n\n"
          "Before the first trail, you may like to run python3 SHE.py demo, this is fast and can let you know what "
          "is the outcome looks like.\n\nPlease notice that prefrom a complete scan will take hours on a 4 GHz CPU!")
