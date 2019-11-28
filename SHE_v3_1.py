import itertools
import math
import sys
import os
import time
from Bio.PDB import *
from Bio.PDB.Atom import *
from Bio.PDB.DSSP import DSSP
from tqdm import tqdm

start = time.time()


def _phi_psi_dic(model, pdb_filename):
    dssp = DSSP(model, pdb_filename, dssp='mkdssp')
    phi_psi_dic = {}
    for x in dssp.property_dict:
        phi_psi_dic[x[1][1]] = str(round(dssp.property_dict[x][4], -1))[:-2] + "_" + str(
            round(dssp.property_dict[x][5], -1))[:-2]

    return phi_psi_dic


def calculateCoordinates(refA, refB, refC, L, ang, di):
    AV = refA.get_vector()
    BV = refB.get_vector()
    CV = refC.get_vector()

    CA = AV - CV
    CB = BV - CV

    ##CA vector
    AX = CA[0]
    AY = CA[1]
    AZ = CA[2]

    ##CB vector
    BX = CB[0]
    BY = CB[1]
    BZ = CB[2]

    ##Plane Parameters
    A = (AY * BZ) - (AZ * BY)
    B = (AZ * BX) - (AX * BZ)
    G = (AX * BY) - (AY * BX)

    ##Dot Product Constant
    F = math.sqrt(BX * BX + BY * BY + BZ * BZ) * L * math.cos(ang * (math.pi / 180.0))

    ##Constants
    const = math.sqrt(math.pow((B * BZ - BY * G), 2) * (-(F * F) * (A * A + B * B + G * G) + (
            B * B * (BX * BX + BZ * BZ) + A * A * (BY * BY + BZ * BZ) - (2 * A * BX * BZ * G) + (
            BX * BX + BY * BY) * G * G - (2 * B * BY) * (A * BX + BZ * G)) * L * L))
    denom = (B * B) * (BX * BX + BZ * BZ) + (A * A) * (BY * BY + BZ * BZ) - (2 * A * BX * BZ * G) + (
            BX * BX + BY * BY) * (G * G) - (2 * B * BY) * (A * BX + BZ * G)

    X = ((B * B * BX * F) - (A * B * BY * F) + (F * G) * (-A * BZ + BX * G) + const) / denom

    if ((B == 0 or BZ == 0) and (BY == 0 or G == 0)):
        const1 = math.sqrt(G * G * (-A * A * X * X + (B * B + G * G) * (L - X) * (L + X)))
        Y = ((-A * B * X) + const1) / (B * B + G * G)
        Z = -(A * G * G * X + B * const1) / (G * (B * B + G * G))
    else:
        Y = ((A * A * BY * F) * (B * BZ - BY * G) + G * (-F * math.pow(B * BZ - BY * G, 2) + BX * const) - A * (
                B * B * BX * BZ * F - B * BX * BY * F * G + BZ * const)) / ((B * BZ - BY * G) * denom)
        Z = ((A * A * BZ * F) * (B * BZ - BY * G) + (B * F) * math.pow(B * BZ - BY * G, 2) + (A * BX * F * G) * (
                -B * BZ + BY * G) - B * BX * const + A * BY * const) / ((B * BZ - BY * G) * denom)

    # GET THE NEW VECTOR from the orgin
    D = Vector(X, Y, Z) + CV
    with warnings.catch_warnings():
        # ignore inconsequential warning
        warnings.simplefilter("ignore")
        temp = calc_dihedral(AV, BV, CV, D) * (180.0 / math.pi)

    di = di - temp
    rot = rotaxis(math.pi * (di / 180.0), CV - BV)
    D = (D - BV).left_multiply(rot) + BV

    return D.get_array()


def makeSer(N, CA, CB, chi1):
    _dict = {}
    CB_OG_length = 1.417
    CA_CB_OG_angle = 110.773
    N_CA_CB_OG_diangle = chi1
    oxygen_g = calculateCoordinates(N, CA, CB, CB_OG_length, CA_CB_OG_angle, N_CA_CB_OG_diangle)
    OG = Atom("OG", oxygen_g, 0.0, 1.0, " ", " OG", 0, "O")
    _dict['og'] = (oxygen_g, OG)
    return _dict


def makeGlu(N, CA, CB, chi1, chi2, chi3):
    CB_CG_length = 1.52
    CA_CB_CG_angle = 113.82
    N_CA_CB_CG_diangle = chi1

    CG_CD_length = 1.52
    CB_CG_CD_angle = 113.31
    CA_CB_CG_CD_diangle = chi2

    CD_OE1_length = 1.25
    CG_CD_OE1_angle = 119.02
    CB_CG_CD_OE1_diangle = chi3

    CD_OE2_length = 1.25
    CG_CD_OE2_angle = 118.08
    CB_CG_CD_OE2_diangle = 180.0 + CB_CG_CD_OE1_diangle

    carbon_g = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG = Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d = calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, CA_CB_CG_CD_diangle)
    CD = Atom("CD", carbon_d, 0.0, 1.0, " ", " CD", 0, "C")
    oxygen_e1 = calculateCoordinates(CB, CG, CD, CD_OE1_length, CG_CD_OE1_angle, CB_CG_CD_OE1_diangle)
    OE1 = Atom("OE1", oxygen_e1, 0.0, 1.0, " ", " OE1", 0, "O")
    oxygen_e2 = calculateCoordinates(CB, CG, CD, CD_OE2_length, CG_CD_OE2_angle, CB_CG_CD_OE2_diangle)
    OE2 = Atom("OE1", oxygen_e2, 0.0, 1.0, " ", " OE1", 0, "O")
    _dict = {'cg': (carbon_g, CG), 'cd': (carbon_d, CD), 'oe1': (oxygen_e1, OE1), 'oe2': (oxygen_e2, OE2)}
    return _dict


def makeHis(N, CA, CB, chi1, chi2):
    '''Creates a Histidine residue'''
    CB_CG_length = 1.49
    CA_CB_CG_angle = 113.74
    N_CA_CB_CG_diangle = chi1

    CG_ND1_length = 1.38
    CB_CG_ND1_angle = 122.85
    CA_CB_CG_ND1_diangle = chi2

    CG_CD2_length = 1.35
    CB_CG_CD2_angle = 130.61
    CA_CB_CG_CD2_diangle = 180.0 + CA_CB_CG_ND1_diangle

    ND1_CE1_length = 1.32
    CG_ND1_CE1_angle = 108.5
    CB_CG_ND1_CE1_diangle = 180.0

    CD2_NE2_length = 1.35
    CG_CD2_NE2_angle = 108.5
    CB_CG_CD2_NE2_diangle = 180.0

    carbon_g = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG = Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    nitrogen_d1 = calculateCoordinates(CA, CB, CG, CG_ND1_length, CB_CG_ND1_angle, CA_CB_CG_ND1_diangle)
    ND1 = Atom("ND1", nitrogen_d1, 0.0, 1.0, " ", " ND1", 0, "N")
    carbon_d2 = calculateCoordinates(CA, CB, CG, CG_CD2_length, CB_CG_CD2_angle, CA_CB_CG_CD2_diangle)
    CD2 = Atom("CD2", carbon_d2, 0.0, 1.0, " ", " CD2", 0, "C")
    carbon_e1 = calculateCoordinates(CB, CG, ND1, ND1_CE1_length, CG_ND1_CE1_angle, CB_CG_ND1_CE1_diangle)
    CE1 = Atom("CE1", carbon_e1, 0.0, 1.0, " ", " CE1", 0, "C")
    nitrogen_e2 = calculateCoordinates(CB, CG, CD2, CD2_NE2_length, CG_CD2_NE2_angle, CB_CG_CD2_NE2_diangle)
    NE2 = Atom("NE2", nitrogen_e2, 0.0, 1.0, " ", " OE1", 0, "N")
    _dict = {'cg': (carbon_g, CG), 'nd1': (nitrogen_d1, ND1), 'cd2': (carbon_d2, CD2), 'ce1': (carbon_e1, CE1),
             'ne2': (nitrogen_e2, NE2)}

    return _dict


def _pair_map(atom_dict, min, max):
    plst = []
    for x in atom_dict:
        plst.append(x)
    l = len(plst)
    pair_dict = {}
    full_list = []
    i = 0
    n = 0
    for i in range(l):
        n = i + 1
        lst = [plst[i]]
        for n in range(i, l):
            d = atom_dict[plst[i]] - atom_dict[plst[n]]

            if min < d < max:
                lst.append(plst[n])
            n = n + 1

        if len(lst) > 2:
            pair_dict[i + 1] = lst
            res_cob_list = list(itertools.combinations(lst, 3))
            for res_cob in res_cob_list:
                if abs(res_cob[0] - res_cob[1]) > 10 and abs(res_cob[0] - res_cob[2]) > 10 and abs(
                        res_cob[1] - res_cob[2]) > 10:
                    full_list.append(res_cob)

        i = i + 1

    return full_list


def _lst2pdb(lst, n):
    atom = "ATOM"
    serial = str("      " + str(n))[-7:]
    # n = serial
    atomname = "  " + str(lst[2])
    nl = atom + serial + atomname + "     "
    nl = nl[0:17]
    nl = nl + lst[3] + " " + lst[4] + " "
    respos = str("    " + lst[5])[-3:]
    nl = nl + respos + "    "
    x = ("        " + str(lst[6]).split(".")[0] + "." + str(lst[6]).split(".")[1][:3])[-8:]
    y = ("        " + str(lst[7]).split(".")[0] + "." + str(lst[7]).split(".")[1][:3])[-8:]
    z = ("        " + str(lst[8]).split(".")[0] + "." + str(lst[8]).split(".")[1][:3])[-8:]

    nl = nl + x + y + z

    return nl


def _BBB(pdb):
    n = 0
    ofile = open(pdb.split(".")[0] + "bbb.pdb", "w+")
    ofile = open(pdb.split(".")[0] + "bbb.pdb", "a+")
    bbb_list = ["N", "C", "CB", "O", "CA"]
    aa_list = ["GLY", "PRO"]
    pdb_file = open(pdb)
    for line in pdb_file:
        if line.startswith("ATOM"):
            l = line.strip("\n").split()
            if l[2] in bbb_list:
                n = n + 1
                print(_lst2pdb(l, n), file=ofile)
                # print(line)


def GLU_backbone_dic():
    lib = open("GLU.simple.lib")
    # print(lib)
    GLU_lib_dic = {}
    for line in lib:
        lst = line.replace("\n", "").split()
        # print(lst[1])
        try:
            psi_phi = (lst[1] + "_" + lst[2])
            pob = float(lst[8])
            if pob > 0.001:
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
            psi_phi = (lst[1] + "_" + lst[2])
            pob = float(lst[8])
            if pob > 0.001:
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
            psi_phi = (lst[1] + "_" + lst[2])
            pob = float(lst[8])
            if pob > 0.05:
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
    diangle = phi_psi_dic.get(position)
    for k in lib_dic:
        if k[0] == diangle:
            sub_dic[k[1]] = lib_dic[k]
    return sub_dic


def mut(position, mut_AA):
    try:
        res = res_dict[position]
        CA = res["CA"]
        CB = res["CB"]
        N = res["N"]
        outd = {}
        if mut_AA == "SER":
            # lib_name = mut_AA+"_lib_dic"
            sub_dic = backbone_sub_dic(position, SER_lib_dic)
            for k in sub_dic:
                outd[k] = makeSer(N, CA, CB, sub_dic[k][0])
        if mut_AA == "HIS":
            # lib_name = mut_AA+"_lib_dic"
            sub_dic = backbone_sub_dic(position, HIS_lib_dic)
            for k in sub_dic:
                outd[k] = makeHis(N, CA, CB, sub_dic[k][0], sub_dic[k][1])
        if mut_AA == "GLU":
            sub_dic = backbone_sub_dic(position, GLU_lib_dic)
            for k in sub_dic:
                outd[k] = makeGlu(N, CA, CB, sub_dic[k][0], sub_dic[k][1], sub_dic[k][2])
        return outd
    except KeyError:
        print(position)


def SER_HIS_GLU_pair(pSER, pHIS, pGLU, k):
    try:
        n = 1
        SER_HIS_GLU_dict = {}
        SHE_dict = {}
        sSER = mut(pSER, "SER")
        sHIS = mut(pHIS, "HIS")
        sGLU = mut(pGLU, "GLU")
        # get atoms
        ser_ca = res_dict[pSER]["CA"]
        his_ca = res_dict[pHIS]["CA"]
        glu_ca = res_dict[pGLU]["CA"]
        for pobhis in sHIS:
            his = sHIS[pobhis]
            ne2 = his['ne2'][1]
            ne2_v = ne2.get_vector()
            nd1 = his["nd1"][1]
            nd1_v = nd1.get_vector()
            ce1 = his['ce1'][1]
            ce1_v = ce1.get_vector()
            for pobser in sSER:
                ser = sSER[pobser]
                og = ser['og'][1]
                his_ca_og = his_ca - og
                ser_ca_ce1 = ser_ca - ce1
                # cb1 = ser_cb-his_cb
                if his_ca_og > 3.5 and ser_ca_ce1 > 3.5:
                    og_v = og.get_vector()
                    cb_v = cb.get_vector()
                    og_ne2 = og - ne2
                    if abs(1 - og_ne2 / 2.8) < k:
                        #print(type(og_ne2))
                        ne2_og_cb = calc_angle(ne2_v, og_v, cb_v)
                        if abs(1 - 57.3 * ne2_og_cb / 90) < k:
                            ce1_ne2_og = calc_angle(ce1_v, ne2_v, og_v)
                            if abs(1 - 57.3 * ce1_ne2_og / 100) < k:
                                nd1_ce1_ne2_og = calc_dihedral(nd1_v, ce1_v, ne2_v, og_v)
                                if 1 - abs(57.3 * nd1_ce1_ne2_og / 140) < k:
                                    # print(nd1_ce1_ne2_og)
                                    for pobglu in sGLU:
                                        glu = sGLU[pobglu]
                                        glu_ca_og = glu_ca - og
                                        glu_ca_ce1 = glu_ca - ce1
                                        # cb2 = his_cb - glu_cb
                                        # cb3 = ser_cb - glu_cb
                                        if glu_ca_ce1 > 3.5 and glu_ca_og > 3.5:
                                            oe1 = glu['oe1'][1]
                                            his_ca_oe1 = his_ca - oe1
                                            ser_ca_oe1 = ser_ca - oe1
                                            if his_ca_oe1 > 3.5 and ser_ca_oe1 > 3.5:
                                                oe1_nd1 = oe1 - nd1
                                                # print(oe1_nd1)
                                                if abs(1 - oe1_nd1 / 2.7) < k:
                                                    oe1_v = oe1.get_vector()
                                                    oe1_nd1_ce1 = calc_angle(oe1_v, nd1_v, ce1_v)
                                                    og_ce1_oe1 = calc_angle(og_v, ce1_v, oe1_v)
                                                    if abs(1 - 57.3 * oe1_nd1_ce1 / 120) < k and abs(
                                                            1 - 57.3 * og_ce1_oe1 / 160) < k:
                                                        oe1_nd1_ce1_ne2 = calc_dihedral(oe1_v, nd1_v, ce1_v, ne2_v)
                                                        # print(oe1_nd1_ce1_ne2)
                                                        if 1 - abs(57.3 * oe1_nd1_ce1_ne2 / 140) < k:
                                                            # print(57.3*oe1_nd1_ce1_ne2)
                                                            oe2 = glu['oe2'][1]
                                                            oe2_nd1 = abs(oe2 - nd1)
                                                            if oe2_nd1 > oe1_nd1:
                                                                n = n + 1
                                                                SHE_dict[pSER] = [
                                                                    ['ATOM', 1, 'OG', 'SER', "A", str(pSER)] +
                                                                    ser['og'][0].tolist()]

                                                                SHE_dict[pHIS] = [
                                                                    ['ATOM', 1, 'CG', 'HIS', "A", str(pHIS)] +
                                                                    his['cg'][0].tolist(),
                                                                    ['ATOM', 1, 'ND1', 'HIS', "A", str(pHIS)] +
                                                                    his['nd1'][0].tolist(),
                                                                    ['ATOM', 1, 'CD2', 'HIS', "A", str(pHIS)] +
                                                                    his['cd2'][0].tolist(),
                                                                    ['ATOM', 1, 'CE1', 'HIS', "A", str(pHIS)] +
                                                                    his['ce1'][0].tolist(),
                                                                    ['ATOM', 1, 'NE2', 'HIS', "A", str(pHIS)] +
                                                                    his['ne2'][0].tolist()]

                                                                SHE_dict[pGLU] = [
                                                                    ['ATOM', 1, 'CG', 'GLU', "A", str(pGLU)] +
                                                                    glu['cg'][0].tolist(),
                                                                    ['ATOM', 1, 'CD', 'GLU', "A", str(pGLU)] +
                                                                    glu['cd'][0].tolist(),
                                                                    ['ATOM', 1, 'OE1', 'GLU', "A", str(pGLU)] +
                                                                    glu['oe1'][0].tolist(),
                                                                    ['ATOM', 1, 'OE2', 'GLU', "A", str(pGLU)] +
                                                                    glu['oe2'][0].tolist()]
                                                                ifile = open(pdb_filename)
                                                                # print(SHE_dict)
                                                                # for x in SHE_dict.keys():
                                                                # print(x)
                                                                ofilename = pdb_filename.split(".")[0] + "_" + str(
                                                                    pHIS) + "_HIS" + "_" + str(
                                                                    pSER) + "_SER" + "_" + str(
                                                                    pGLU) + "_GLU" + "_" + str(n) + ".pdb"
                                                                ofile = open(ofilename, "w+")
                                                                ofile = open(ofilename, "a+")
                                                                s = 1
                                                                bbb_list = ["N", "C", "CB", "O", "CA"]
                                                                seri = 0
                                                                hisi = 0
                                                                glui = 0
                                                                for line in ifile:
                                                                    ol = line.split()
                                                                    if ol[0] == "ATOM":
                                                                        if int(ol[5]) in SHE_dict.keys():
                                                                            if ol[2] in bbb_list:
                                                                                lsts = SHE_dict[int(ol[5])]
                                                                                for nl in lsts:
                                                                                    resname = nl[3]
                                                                                    ol[3] = resname
                                                                                print(_lst2pdb(ol, s), file=ofile)
                                                                                s = s + 1
                                                                            if ol[2] not in bbb_list:
                                                                                lsts = SHE_dict[int(ol[5])]
                                                                                len(lsts)
                                                                                if len(lsts) == 1 and seri < 1:
                                                                                    print(_lst2pdb(lsts[seri], s), file=ofile)
                                                                                    seri = seri + 1
                                                                                    s = s + 1
                                                                                if len(lsts) == 5 and hisi < 5:
                                                                                    print(_lst2pdb(lsts[hisi], s), file=ofile)
                                                                                    hisi = hisi + 1
                                                                                    s = s + 1
                                                                                if len(lsts) == 4 and glui < 4:
                                                                                    print(_lst2pdb(lsts[glui], s), file=ofile)
                                                                                    glui = glui + 1
                                                                                    s = s + 1

                                                                        else:
                                                                            print(_lst2pdb(ol, s), file=ofile)
                                                                            s = s + 1


                                                                ser_out_name = "SER" + "_" + str(pSER)
                                                                his_out_name = "HIS" + "_" + str(pHIS)
                                                                glu_out_name = "GLU" + "_" + str(pGLU)

                                                                SER_HIS_GLU_dict[
                                                                    ser_out_name + "_" + his_out_name + "_" + glu_out_name] = n



    except KeyError:
        print(TypeError)

    return SER_HIS_GLU_dict




if len(sys.argv) < 2:
    print("Error, not enough arguments!\nUsage:python3 SHE.py you_pdbfile\n\n\ne.g.: python3 SHE.py 3wzl.pdb\n\n"
          "Before the first trail, you may like to run python3 SHE.py demo, this is fast and can let you know what "
          "is the outcome looks like\n\nPlease notice that prefrom a complete scan will take hours on a 4 GHz CPU!")
try:
    if sys.argv[1] == "demo":
        # input
        pdb_filename = "3wzl_A.pdb"
        p = PDBParser(PERMISSIVE=True, QUIET=True)
        s = p.get_structure("X", pdb_filename)
        model = s[0]
        chain = model["A"]
        residues = model.get_residues()
        atom_dict = {}
        res_dict = {}
        n = 1
        for residue in residues:
            if is_aa(residue):
                try:
                    cb = residue['CB']
                    atom_dict[n] = cb
                    res_dict[n] = residue
                    n = n + 1
                except KeyError:
                    n = n + 1

        phi_psi_dic = _phi_psi_dic(model, pdb_filename)


        SER_lib_dic = SER_backbone_dic()
        HIS_lib_dic = HIS_backbone_dic()
        GLU_lib_dic = GLU_backbone_dic()

        k = 0.25
        min = 4
        max = 12

        fulllst = [[102,242,126]]
        pbar = tqdm(6)
        #print(len(fulllst))
        for xlst in fulllst:

            ofile = open(pdb_filename + "_SHE.out", "a+")
            od = SER_HIS_GLU_pair(xlst[0], xlst[1], xlst[2], k)
            pbar.update(1)
            if od != {}:
                for key in od:
                    print(key,od[key])

            od = SER_HIS_GLU_pair(xlst[0], xlst[2], xlst[1], k)
            pbar.update(1)
            if od != {}:
                for key in od:
                    print(key,od[key])

            od = SER_HIS_GLU_pair(xlst[1], xlst[0], xlst[2], k)
            pbar.update(1)
            if od != {}:
                for key in od:
                    print(key,od[key])

            od = SER_HIS_GLU_pair(xlst[1], xlst[2], xlst[0], k)
            pbar.update(1)
            if od != {}:
                for key in od:
                    print(key,od[key])

            od = SER_HIS_GLU_pair(xlst[2], xlst[1], xlst[0], k)
            pbar.update(1)
            if od != {}:
                for key in od:
                    print(key,od[key])

            od = SER_HIS_GLU_pair(xlst[2], xlst[0], xlst[1], k)
            pbar.update(1)
            if od != {}:
                for key in od:
                    print(key,od[key])

        pbar.close()

        end = time.time()
        print("This job took " + str(end - start) + " seconds!")


    if sys.argv[1] != "demo" and sys.argv[2] != "mt" and sys.argv[2] != "nt":
        pdb_filename = sys.argv[1]

        p = PDBParser(PERMISSIVE=True, QUIET=True)
        s = p.get_structure("X", pdb_filename)
        model = s[0]
        chain = model["A"]
        residues = model.get_residues()
        atom_dict = {}
        res_dict = {}
        n = 1
        for residue in residues:
            if is_aa(residue):
                try:
                    cb = residue['CB']
                    atom_dict[n] = cb
                    res_dict[n] = residue
                    n = n + 1
                except KeyError:
                    n = n + 1

        phi_psi_dic = _phi_psi_dic(model, pdb_filename)


        SER_lib_dic = SER_backbone_dic()
        HIS_lib_dic = HIS_backbone_dic()
        GLU_lib_dic = GLU_backbone_dic()

        k = 0.25
        min = 4
        max = 12
        fulllst = _pair_map(atom_dict,min,max)
        pbar = tqdm(len(fulllst))
        #print(len(fulllst))
        for xlst in fulllst:
            pbar.update(1)
            ofile = open(pdb_filename + "_SHE.out", "a+")
            od = SER_HIS_GLU_pair(xlst[0], xlst[1], xlst[2], k)
            if od != {}:
                for key in od:
                    print(key,od[key],file=ofile)
            od = SER_HIS_GLU_pair(xlst[0], xlst[2], xlst[1], k)
            if od != {}:
                for key in od:
                    print(key,od[key],file=ofile)
            od = SER_HIS_GLU_pair(xlst[1], xlst[0], xlst[2], k)
            if od != {}:
                for key in od:
                    print(key,od[key],file=ofile)
            od = SER_HIS_GLU_pair(xlst[1], xlst[2], xlst[0], k)
            if od != {}:
                for key in od:
                    print(key,od[key],file=ofile)
            od = SER_HIS_GLU_pair(xlst[2], xlst[1], xlst[0], k)
            if od != {}:
                for key in od:
                    print(key,od[key],file=ofile)
            od = SER_HIS_GLU_pair(xlst[2], xlst[0], xlst[1], k)
            if od != {}:
                for key in od:
                    print(key,od[key],file=ofile)
        pbar.close()


        end = time.time()
        print("This job took " + str(end - start) + " seconds!")


    if sys.argv[2] == "nt":
        print("yes")
        pdb_filename = sys.argv[1]
        p = PDBParser(PERMISSIVE=True, QUIET=True)
        s = p.get_structure("X", pdb_filename)
        model = s[0]
        chain = model["A"]
        residues = model.get_residues()
        atom_dict = {}
        res_dict = {}
        n = 1
        for residue in residues:
            if is_aa(residue):
                try:
                    cb = residue['CB']
                    atom_dict[n] = cb
                    res_dict[n] = residue
                    n = n + 1
                except KeyError:
                    n = n + 1

        phi_psi_dic = _phi_psi_dic(model, pdb_filename)


        SER_lib_dic = SER_backbone_dic()
        HIS_lib_dic = HIS_backbone_dic()
        GLU_lib_dic = GLU_backbone_dic()

        k = 0.25
        min = 4
        max = 12
        fulllst = _pair_map(atom_dict,min,max)
        pbar = tqdm(len(fulllst))
        #print(len(fulllst))
        for xlst in fulllst:
            pbar.update(1)
            ofile = open(pdb_filename + "_SHE.out", "a+")
            p3 = str(xlst[2])
            p1 = str(xlst[0])
            p2 = str(xlst[1])
            os.system("nohup python3.6 SHE_v3_1.py "+pdb_filename+" mt "+p1+" "+p2+" "+p3+" &")
            os.system("nohup python3.6 SHE_v3_1.py "+pdb_filename+" mt "+p1+" "+p3+" "+p2+" &")
            os.system("nohup python3.6 SHE_v3_1.py "+pdb_filename+" mt "+p3+" "+p1+" "+p2+" &")
            os.system("nohup python3.6 SHE_v3_1.py "+pdb_filename+" mt "+p3+" "+p2+" "+p1+" &")
            os.system("nohup python3.6 SHE_v3_1.py "+pdb_filename+" mt "+p2+" "+p1+" "+p3+" &")
            os.system("nohup python3.6 SHE_v3_1.py "+pdb_filename+" mt "+p2+" "+p3+" "+p1+" &")
            time.sleep(0.1)

        pbar.close()


        end = time.time()
        print("This job took " + str(end - start) + " seconds!")


    if sys.argv[2] == "mt" and sys.argv[3] != "":
        pdb_filename = sys.argv[1]
        p = PDBParser(PERMISSIVE=True, QUIET=True)
        s = p.get_structure("X", pdb_filename)
        model = s[0]
        chain = model["A"]
        residues = model.get_residues()
        atom_dict = {}
        res_dict = {}
        n = 1
        for residue in residues:
            if is_aa(residue):
                try:
                    cb = residue['CB']
                    atom_dict[n] = cb
                    res_dict[n] = residue
                    n = n + 1
                except KeyError:
                    n = n + 1

        phi_psi_dic = _phi_psi_dic(model, pdb_filename)


        SER_lib_dic = SER_backbone_dic()
        HIS_lib_dic = HIS_backbone_dic()
        GLU_lib_dic = GLU_backbone_dic()

        k = 0.25
        min = 4
        max = 12
        lst = [int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]

        ofile = open(pdb_filename + "_SHE.out", "a+")

        od = SER_HIS_GLU_pair(lst[2], lst[0], lst[1], k)
        if od != {}:
            for key in od:
                print(key,od[key],file=ofile)



        end = time.time()
        print("This job took " + str(end - start) + " seconds!")


except IndexError:
    print("Error, not enough arguments!\nUsage:python3 SHE.py you_pdbfile\n\n\ne.g.: python3 SHE.py 3wzl.pdb\n\n"
          "Before the first trail, you may like to run python3 SHE.py demo, this is fast and can let you know what "
          "is the outcome looks like.\n\nPlease notice that preform a complete scan will take hours on a 4 GHz CPU!")
