#!/usr/bin/env python3
"""
================================================================================
OPTIMALITY GAP -- GNN vs MILP (VERSIONE v2: repair corretta + timing isolato
                                            + salvataggio incrementale)
================================================================================
Differenze rispetto alla versione precedente:
  1. repair_predictions CORRETTA: dc[d,i] viene derivato da y_pred dopo
     l'assegnazione (dc[d,i]=1 sse esiste almeno un cliente assegnato a quel
     DC), invece di essere preso direttamente dalla sigmoid della testa dc.
     Questo garantisce per costruzione i vincoli di linking dc<=sum(y) e
     y<=dc, eliminando la classe di infeasibility "predizioni incoerenti"
     osservata nella run precedente (~89% di fallimenti).
  2. Timing isolato: per ogni istanza si misura separatamente il tempo di
     (a) SOLO forward pass GNN, (b) SOLO repair, (c) SOLO solve_fixed (LP
     residuo con binarie fissate) -- oltre al tempo totale che include anche
     Z_true (CG+MILP esatto). Prima veniva misurato solo il totale, dominato
     da Z_true e quindi inutile per stimare la velocita' della GNN.
  3. Salvataggio incrementale: il file dei risultati viene riscritto dopo
     OGNI istanza, non solo alla fine. Se il processo si interrompe (sleep,
     crash, chiusura terminale), tutto il lavoro fatto fino a quel punto e'
     comunque salvato su disco.
================================================================================
"""

import json, glob, os, time
import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB
from scipy.special import pdtr
from sklearn.model_selection import train_test_split

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# CONFIG -- ADATTA I PATH
# ============================================================================
DATASET_DIR = "gnn_dataset"
MODEL_PATH  = "gnn_model.pt"
REFERENCE_CASE_JSON = "reference_case_results.json"
OUTPUT_FILE = "optimality_gap_results_v2.json"
THREADS_PER_SOLVE = 5
SEED = 42
SEED_BASE = 42

def get_time_limit(nI):
    if nI <= 10: return 300
    if nI <= 15: return 600
    if nI <= 25: return 900
    if nI <= 50: return 3600
    return 7200

MIP_GAP_TARGET = 0.02
N_SCENARIOS = 20
N_PWL = 12
H_D_VAL, CAP_3DP = 0.25, 5000.0
O_RAW, C_RAW, DEN_RAW, UNIT_RAW = 30.0, 50.0, 1200.0, 0.5
L_RAW, SL_RAW, O_VAL = 1.0/52, 0.85, 50.0
HIDDEN_DIM, NUM_LAYERS, DROPOUT = 128, 2, 0.2


# ============================================================================
# FUNZIONI ANALITICHE
# ============================================================================
def expected_backorder(mu_L, beta):
    if mu_L < 1e-9 or beta <= 0: return mu_L
    return max(mu_L*(1 - pdtr(beta-1, mu_L)) - beta*(1 - pdtr(beta, mu_L)), 0.0)

def expected_on_hand(mu_L, beta):
    return max(beta - mu_L + expected_backorder(mu_L, beta), 0.0)

def optimal_beta(mu_L, c_hold, b_val):
    if mu_L < 1e-9: return 0, 0.0
    best_c, best_b = float('inf'), 0
    for b in range(0, int(mu_L*4)+20):
        t = c_hold*b + b_val*expected_backorder(mu_L, b)
        if t < best_c: best_c, best_b = t, b
        elif t > best_c*1.5 and b > mu_L: break
    return best_b, best_c

def beta_poisson(mu_val, L_val, SL_target):
    lam = mu_val*L_val
    if lam <= 1e-9: return 0
    for s in range(100000):
        if pdtr(s, lam) >= SL_target: return int(s)
    return 99999


# ============================================================================
# RICOSTRUZIONE ISTANZA
# ============================================================================
def inst_from_json(d):
    nI, nD, nC = d['nI'], d['nD'], d['nC']
    c_CM = np.array(d['c_CM'], dtype=np.float64)
    ratio = np.array(d['ratio'], dtype=np.float64)
    L_CM = np.array(d['L_CM'], dtype=np.float64)
    L_AM = np.array(d['L_AM'], dtype=np.float64)
    b_i = np.array(d['b_i'], dtype=np.float64)
    SL = np.array(d['SL'], dtype=np.float64)
    mu_ic = np.array(d['mu_ic'], dtype=np.float64)
    coord_DC = np.array(d['coord_DC'], dtype=np.float64)
    coord_C = np.array(d['coord_C'], dtype=np.float64)
    dist_dc_c = np.array(d['dist_dc_c'], dtype=np.float64)
    Leas = d['Leas']; costo_km = d['costo_km']
    c_AM = c_CM * ratio; n_i = ratio
    h_d = np.full(nD, H_D_VAL)
    t_dic = np.stack([dist_dc_c * costo_km] * nI, axis=1)
    L_AM_mat = np.column_stack([L_AM]*nD)
    return {'nI':nI,'nD':nD,'nC':nC,'c_CM':c_CM,'ratio':ratio,'c_AM':c_AM,
            'n_i':n_i,'L_CM':L_CM,'L_AM':L_AM,'L_AM_mat':L_AM_mat,'b_i':b_i,
            'SL':SL,'mu_ic':mu_ic,'Leas':Leas,'costo_km':costo_km,
            'coord_DC':coord_DC,'coord_C':coord_C,'dist_dc_c':dist_dc_c,
            't_dic':t_dic,'h_d':h_d}


# ============================================================================
# COLUMN GENERATION (warm-start per Z_true) -- identica all'originale
# ============================================================================
def column_generation(inst, n_threads):
    nI, nD, nC = inst['nI'], inst['nD'], inst['nC']
    p = inst
    table = {}
    for i in range(nI):
        table[i] = {}
        for d in range(nD):
            table[i][d] = {}
            for tech in ['CM', 'AM']:
                table[i][d][tech] = {}
                for mask in range(1, 2**nC):
                    clients = [c for c in range(nC) if mask & (1<<c)]
                    mu_id = sum(p['mu_ic'][i,c] for c in clients)
                    if mu_id < 1e-9: continue
                    trans = sum(p['t_dic'][d,i,c]*p['mu_ic'][i,c] for c in clients)
                    if tech == 'CM':
                        mu_L = mu_id*p['L_CM'][i]
                        ch = p['c_CM'][i]*(1+p['h_d'][d])
                        beta, bsc = optimal_beta(mu_L, ch, p['b_i'][i])
                        cost = (bsc + np.sqrt(2*O_VAL*p['h_d'][d]*p['c_CM'][i]*mu_id)
                                + p['c_CM'][i]*mu_id + trans
                                + p['h_d'][d]*p['c_CM'][i]*expected_on_hand(mu_L, beta))
                        ph = 0
                    else:
                        L_am = p['L_AM'][i]
                        mu_L = mu_id*L_am
                        ch = p['c_AM'][i]*(1+p['h_d'][d])
                        beta, bsc = optimal_beta(mu_L, ch, p['b_i'][i])
                        prod = p['c_AM'][i]*mu_id
                        vol = p['c_AM'][i]/(1.30*1e6)
                        qr = vol*DEN_RAW/UNIT_RAW
                        cost = (bsc + prod
                                + np.sqrt(2*O_RAW*p['h_d'][d]*C_RAW*max(qr*mu_id,1e-9))
                                + p['h_d'][d]*C_RAW*qr*mu_id*0.5 + trans
                                + p['h_d'][d]*p['c_AM'][i]*expected_on_hand(mu_L, beta))
                        ph = p['c_AM'][i]/(1.30*0.00525*3600)*mu_id
                    table[i][d][tech][mask] = (cost, mu_id, beta, ph)

    columns = []
    fm = (1<<nC)-1
    for i in range(nI):
        for d in range(nD):
            for tech in ['CM','AM']:
                if fm in table[i][d][tech]:
                    columns.append((d,i,tech,fm)+table[i][d][tech][fm])
    existing = set((c[0],c[1],c[2],c[3]) for c in columns)

    Leas = p['Leas']
    mdl = gp.Model("CG"); mdl.Params.OutputFlag=0; mdl.Params.Method=1; mdl.Params.Threads=n_threads
    n3dp = {d: mdl.addVar(lb=0) for d in range(nD)}
    cover = {(i,c): mdl.addConstr(gp.LinExpr()==1) for i in range(nI) for c in range(nC)}
    cap_c = {d: mdl.addConstr(-CAP_3DP*n3dp[d]<=0) for d in range(nD)}
    mdl.setObjective(gp.quicksum(Leas*n3dp[d] for d in range(nD)), GRB.MINIMIZE)
    mdl.update()
    lv = []
    for col in columns:
        dc_r,sk,te,mk,co,mu,be,ph = col
        gc = gp.Column()
        for c in range(nC):
            if mk & (1<<c): gc.addTerms(1.0, cover[sk,c])
        if te=='AM' and ph>0: gc.addTerms(ph, cap_c[dc_r])
        lv.append(mdl.addVar(lb=0, ub=1, obj=co, column=gc)); mdl.update()

    for it in range(80):
        mdl.optimize()
        if mdl.Status != 2: break
        duals = np.zeros((nI,nC))
        for i in range(nI):
            for c in range(nC): duals[i,c]=cover[i,c].Pi
        wv = np.array([n3dp[d].X for d in range(nD)])
        new = []
        for i in range(nI):
            for d in range(nD):
                for tech in ['CM','AM']:
                    brc,bm,be = -1e12,0,None
                    for mask,entry in table[i][d][tech].items():
                        ds = sum(duals[i,c] for c in range(nC) if mask&(1<<c))
                        rc = ds-entry[0]
                        if rc>brc: brc,bm,be=rc,mask,entry
                    if be is None: continue
                    ra = brc
                    if tech=='AM' and wv[d]<0.5: ra -= Leas*(1-wv[d])
                    if ra > 1e-4:
                        key=(d,i,tech,bm)
                        if key not in existing: new.append((d,i,tech,bm)+be); existing.add(key)
        for nc in new:
            dc_r,sk,te,mk,co,mu,be,ph = nc; columns.append(nc); gc=gp.Column()
            for c in range(nC):
                if mk&(1<<c): gc.addTerms(1.0, cover[sk,c])
            if te=='AM' and ph>0: gc.addTerms(ph, cap_c[dc_r])
            lv.append(mdl.addVar(lb=0, ub=1, obj=co, column=gc)); mdl.update()
        if len(new)==0: break
    mdl.dispose()

    m2 = gp.Model("MILP_CG"); m2.Params.OutputFlag=0; m2.Params.TimeLimit=300; m2.Params.Threads=n_threads
    l2 = m2.addVars(len(columns), vtype=GRB.BINARY); n2 = m2.addVars(nD, vtype=GRB.INTEGER, lb=0)
    for i in range(nI):
        for c in range(nC):
            m2.addConstr(gp.quicksum(l2[r] for r in range(len(columns))
                        if columns[r][1]==i and columns[r][3]&(1<<c))==1)
    for d in range(nD):
        m2.addConstr(gp.quicksum(columns[r][7]*l2[r] for r in range(len(columns))
                     if columns[r][0]==d and columns[r][2]=='AM' and columns[r][7]>0)
                     <= CAP_3DP*n2[d])
    m2.setObjective(gp.quicksum(columns[r][4]*l2[r] for r in range(len(columns)))
                    + gp.quicksum(Leas*n2[d] for d in range(nD)), GRB.MINIMIZE)
    m2.optimize()
    sel = []; n3 = [0]*nD
    if m2.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        sel = [r for r in range(len(columns)) if l2[r].X>0.5]
        n3 = [int(round(n2[d].X)) for d in range(nD)]
    m2.dispose()

    am_a = np.zeros(nI, dtype=int); dc_s = np.zeros((nD,nI), dtype=int)
    y_s = np.zeros((nD,nI,nC), dtype=int); B_s = np.zeros((nI,nD))
    for r in sel:
        d,i,te,mk,co,mu,be,ph = columns[r]
        if te=='AM': am_a[i]=1
        dc_s[d,i]=1; B_s[i,d]=be
        for c in range(nC):
            if mk&(1<<c): y_s[d,i,c]=1
    return {'am':am_a,'dc':dc_s,'y':y_s,'B':B_s,'n3dp':n3}


# ============================================================================
# COSTRUZIONE MILP -- identica all'originale
# ============================================================================
def build_milp(inst, n_threads, time_limit, mip_gap):
    nI, nD, nC = inst['nI'], inst['nD'], inst['nC']
    nS = N_SCENARIOS
    I,D,C,S = range(nI),range(nD),range(nC),range(nS)
    pi_s = 1.0/nS
    c_CM,SL_i,b_i_arr = inst['c_CM'],inst['SL'],inst['b_i']
    L_CM,L_AM_mat = inst['L_CM'],inst['L_AM_mat']
    mu_ic,t_dic,h_d = inst['mu_ic'],inst['t_dic'],inst['h_d']
    Leas = inst['Leas']
    c_AM = inst['c_AM']; prod_i = c_AM
    a_i = c_AM/(1.30*0.00525*3600)
    vol_i = (prod_i/1.30)*1e-6
    qbar_i = (vol_i*DEN_RAW)/UNIT_RAW
    ptime_i = prod_i/(1.30*0.00525*3600)
    mu_max_i = np.array([mu_ic[i,:].sum() for i in I])

    rng = np.random.default_rng(SEED_BASE)
    d_ics = rng.poisson(mu_ic[:,:,np.newaxis], size=(nI,nC,nS)).astype(float)
    xi = 2.0
    Q_bar_CM = np.zeros((nI,nD)); F_bar = np.zeros((nI,nD))
    for i in I:
        for d in D:
            Q_bar_CM[i,d] = d_ics[i,:,:].sum(axis=0).max()*xi
            F_bar[i,d] = d_ics[i,:,:].max()*xi
    Q_bar_AM = np.full((nI,nD), CAP_3DP*xi)

    mdl = gp.Model("gap_val"); mdl.setParam("OutputFlag",0)
    mdl.setParam("MIPGap", mip_gap); mdl.setParam("TimeLimit", time_limit)
    mdl.setParam("Threads", n_threads); mdl.setParam("MIPFocus", 1)

    am = mdl.addVars(nI, vtype=GRB.BINARY); cm = mdl.addVars(nI, vtype=GRB.BINARY)
    y = mdl.addVars(nD,nI,nC, vtype=GRB.BINARY); dc = mdl.addVars(nD,nI, vtype=GRB.BINARY)
    v = mdl.addVars(nI,nD, vtype=GRB.BINARY); w = mdl.addVars(nI,nD, vtype=GRB.BINARY)
    B = mdl.addVars(nI,nD, lb=0.0); n3DP = mdl.addVars(nD, vtype=GRB.INTEGER, lb=0)
    mu_var = mdl.addVars(nD,nI, lb=0.0); C_EOQ_var = mdl.addVars(nD,nI, lb=0.0)
    dcAM = mdl.addVars(nD, vtype=GRB.BINARY)
    C_EOQ_raw_var = mdl.addVars(nD, lb=0.0); B_raw = mdl.addVars(nD, lb=0.0)
    mu_raw_contrib = mdl.addVars(nD,nI, lb=0.0); mu_raw_total = mdl.addVars(nD, lb=0.0)
    beta_cm_aux = mdl.addVars(nI,nD, lb=0.0); beta_am_aux = mdl.addVars(nI,nD, lb=0.0)
    bo_cm_aux = mdl.addVars(nI,nD, lb=0.0); bo_am_aux = mdl.addVars(nI,nD, lb=0.0)
    q_CM = mdl.addVars(nI,nD,nS, lb=0.0); q_AM = mdl.addVars(nI,nD,nS, lb=0.0)
    f = mdl.addVars(nI,nD,nC,nS, lb=0.0); u = mdl.addVars(nI,nC,nS, lb=0.0)
    I_inv = mdl.addVars(nI,nD,nS, lb=0.0)

    mdl.addConstrs((am[i]+cm[i]==1 for i in I))
    mdl.addConstrs((y[d,i,c]<=dc[d,i] for d in D for i in I for c in C))
    mdl.addConstrs((dc[d,i]<=gp.quicksum(y[d,i,c] for c in C) for d in D for i in I))
    mdl.addConstrs((gp.quicksum(y[d,i,c] for d in D)==1 for i in I for c in C))
    mdl.addConstrs((v[i,d]<=am[i] for i in I for d in D))
    mdl.addConstrs((v[i,d]<=dc[d,i] for i in I for d in D))
    mdl.addConstrs((v[i,d]>=am[i]+dc[d,i]-1 for i in I for d in D))
    mdl.addConstrs((w[i,d]<=cm[i] for i in I for d in D))
    mdl.addConstrs((w[i,d]<=dc[d,i] for i in I for d in D))
    mdl.addConstrs((w[i,d]>=cm[i]+dc[d,i]-1 for i in I for d in D))
    mdl.addConstrs((mu_var[d,i]==gp.quicksum(mu_ic[i,c]*y[d,i,c] for c in C) for d in D for i in I))
    mdl.addConstrs((mu_var[d,i]<=mu_max_i[i]*dc[d,i] for d in D for i in I))

    B_bar = np.zeros((nI,nD))
    for i in I:
        pts = np.linspace(0.0, max(mu_max_i[i],0.01), N_PWL).tolist()
        for d in D:
            bcm = [float(beta_poisson(mu, L_CM[i], SL_i[i])) for mu in pts]
            bam = [float(beta_poisson(mu, L_AM_mat[i,d], SL_i[i])) for mu in pts]
            eoq = [float(np.sqrt(2*O_VAL*h_d[d]*c_CM[i]*mu)) if mu>0 else 0.0 for mu in pts]
            mdl.addGenConstrPWL(mu_var[d,i], beta_cm_aux[i,d], pts, bcm)
            mdl.addGenConstrPWL(mu_var[d,i], beta_am_aux[i,d], pts, bam)
            mdl.addGenConstrPWL(mu_var[d,i], C_EOQ_var[d,i], pts, eoq)
            bo_cm_pts = [b_i_arr[i]*expected_backorder(pts[m]*L_CM[i], bcm[m]) for m in range(len(pts))]
            bo_am_pts = [b_i_arr[i]*expected_backorder(pts[m]*L_AM_mat[i,d], bam[m]) for m in range(len(pts))]
            mdl.addGenConstrPWL(mu_var[d,i], bo_cm_aux[i,d], pts, bo_cm_pts)
            mdl.addGenConstrPWL(mu_var[d,i], bo_am_aux[i,d], pts, bo_am_pts)
            B_bar[i,d] = bcm[-1]

    mdl.addConstrs((B[i,d]>=beta_cm_aux[i,d]-B_bar[i,d]*(1-cm[i]) for i in I for d in D))
    mdl.addConstrs((B[i,d]>=beta_am_aux[i,d]-B_bar[i,d]*(1-am[i]) for i in I for d in D))
    mdl.addConstrs((B[i,d]<=B_bar[i,d]*dc[d,i] for i in I for d in D))

    M_BO = max(max(b_i_arr[i]*mu_max_i[i] for i in I), 1.0)*2
    BO_act = mdl.addVars(nD,nI, lb=0.0)
    for d in D:
        for i in I:
            mdl.addConstr(BO_act[d,i]>=bo_cm_aux[i,d]-M_BO*(1-cm[i]))
            mdl.addConstr(BO_act[d,i]>=bo_am_aux[i,d]-M_BO*(1-am[i]))
            mdl.addConstr(BO_act[d,i]<=bo_cm_aux[i,d]+M_BO*(1-cm[i]))
            mdl.addConstr(BO_act[d,i]<=bo_am_aux[i,d]+M_BO*(1-am[i]))
            mdl.addConstr(BO_act[d,i]<=M_BO*dc[d,i])

    B_cm_lin = mdl.addVars(nI,nD, lb=0.0); B_am_lin = mdl.addVars(nI,nD, lb=0.0)
    for i in I:
        for d in D:
            Bub = float(B_bar[i,d])
            mdl.addConstr(B_cm_lin[i,d]<=Bub*w[i,d])
            mdl.addConstr(B_cm_lin[i,d]<=B[i,d])
            mdl.addConstr(B_cm_lin[i,d]>=B[i,d]-Bub*(1-w[i,d]))
            mdl.addConstr(B_am_lin[i,d]<=Bub*v[i,d])
            mdl.addConstr(B_am_lin[i,d]<=B[i,d])
            mdl.addConstr(B_am_lin[i,d]>=B[i,d]-Bub*(1-v[i,d]))

    for d in D:
        for i in I: mdl.addConstr(dcAM[d]>=v[i,d])
        mdl.addConstr(dcAM[d]<=gp.quicksum(v[i,d] for i in I))
    M_raw_big = float(max(qbar_i)*sum(mu_ic[i,:].sum() for i in I))*2
    for d in D:
        for i in I:
            mdl.addConstr(mu_raw_contrib[d,i]<=M_raw_big*v[i,d])
            mdl.addConstr(mu_raw_contrib[d,i]<=qbar_i[i]*mu_var[d,i])
            mdl.addConstr(mu_raw_contrib[d,i]>=qbar_i[i]*mu_var[d,i]-M_raw_big*(1-v[i,d]))
        mdl.addConstr(mu_raw_total[d]==gp.quicksum(mu_raw_contrib[d,i] for i in I))
        mdl.addConstr(mu_raw_total[d]<=M_raw_big*dcAM[d])
    mu_raw_max = float(sum(qbar_i[i]*mu_ic[i,:].sum() for i in I))
    mu_raw_pts = np.linspace(0.0, max(mu_raw_max,0.01), N_PWL).tolist()
    for d in D:
        brp = [float(beta_poisson(mu, L_RAW, SL_RAW)) for mu in mu_raw_pts]
        erp = [float(np.sqrt(2*O_RAW*h_d[d]*C_RAW*mu)) if mu>0 else 0.0 for mu in mu_raw_pts]
        bra = mdl.addVar(lb=0.0)
        mdl.addGenConstrPWL(mu_raw_total[d], bra, mu_raw_pts, brp)
        mdl.addConstr(B_raw[d]>=bra); mdl.addConstr(B_raw[d]<=max(brp)*dcAM[d])
        mdl.addGenConstrPWL(mu_raw_total[d], C_EOQ_raw_var[d], mu_raw_pts, erp)
    M_EOQ_big = float(max(np.sqrt(2*O_VAL*H_D_VAL*c_CM.max()*mu_max_i.max()),1))*2
    C_EOQ_act = mdl.addVars(nD,nI, lb=0.0)
    for d in D:
        for i in I:
            mdl.addConstr(C_EOQ_act[d,i]<=M_EOQ_big*w[i,d])
            mdl.addConstr(C_EOQ_act[d,i]<=C_EOQ_var[d,i])
            mdl.addConstr(C_EOQ_act[d,i]>=C_EOQ_var[d,i]-M_EOQ_big*(1-w[i,d]))
    ratio_pt = np.where(qbar_i>1e-9, ptime_i/qbar_i, 0.0)
    mdl.addConstrs((n3DP[d]>=gp.quicksum(ratio_pt[i]*mu_raw_contrib[d,i] for i in I)/CAP_3DP for d in D))

    cost_BS = gp.quicksum(c_CM[i]*(1+h_d[d])*B_cm_lin[i,d]
                          + prod_i[i]*(1+h_d[d])*B_am_lin[i,d] for i in I for d in D)
    cost_EOQ = gp.quicksum(C_EOQ_act[d,i] for i in I for d in D)
    cost_raw_dc = gp.quicksum(C_EOQ_raw_var[d]+h_d[d]*C_RAW*B_raw[d] for d in D)
    cost_leas = gp.quicksum(Leas*n3DP[d] for d in D)
    cost_bo_lt = gp.quicksum(BO_act[d,i] for i in I for d in D)

    alpha_ic = np.column_stack([SL_i]*nC)
    mdl.addConstrs((gp.quicksum(a_i[i]*q_AM[i,d,s] for i in I)<=CAP_3DP*n3DP[d] for d in D for s in S))
    mdl.addConstrs((B[i,d]+q_CM[i,d,s]+q_AM[i,d,s]-gp.quicksum(f[i,d,c,s] for c in C)==I_inv[i,d,s]
                    for i in I for d in D for s in S))
    mdl.addConstrs((gp.quicksum(f[i,d,c,s] for d in D)+u[i,c,s]==d_ics[i,c,s]
                    for i in I for c in C for s in S))
    mdl.addConstrs((f[i,d,c,s]<=F_bar[i,d]*dc[d,i] for i in I for d in D for c in C for s in S))
    mdl.addConstrs((q_CM[i,d,s]<=Q_bar_CM[i,d]*w[i,d] for i in I for d in D for s in S))
    mdl.addConstrs((q_AM[i,d,s]<=Q_bar_AM[i,d]*v[i,d] for i in I for d in D for s in S))
    mdl.addConstrs((gp.quicksum(pi_s*gp.quicksum(f[i,d,c,s] for d in D) for s in S)
                    >=alpha_ic[i,c]*gp.quicksum(pi_s*d_ics[i,c,s] for s in S) for i in I for c in C))
    cost_2nd = gp.quicksum(pi_s*(
        gp.quicksum(c_CM[i]*q_CM[i,d,s] for i in I for d in D) +
        gp.quicksum(prod_i[i]*q_AM[i,d,s] for i in I for d in D) +
        gp.quicksum(t_dic[d,i,c]*f[i,d,c,s] for i in I for d in D for c in C) +
        gp.quicksum(h_d[d]*c_CM[i]*I_inv[i,d,s] for i in I for d in D) +
        gp.quicksum(b_i_arr[i]*u[i,c,s] for i in I for c in C)) for s in S)
    mdl.setObjective(cost_BS+cost_EOQ+cost_raw_dc+cost_leas+cost_bo_lt+cost_2nd, GRB.MINIMIZE)

    varset = {'am':am,'cm':cm,'y':y,'dc':dc,'v':v,'w':w,'B':B,'n3DP':n3DP}
    return mdl, varset, {'d_ics': d_ics}


def solve_free(inst, warm_start, n_threads, time_limit, mip_gap):
    mdl, v, _ = build_milp(inst, n_threads, time_limit, mip_gap)
    ws = warm_start
    nI, nD, nC = inst['nI'], inst['nD'], inst['nC']
    for i in range(nI):
        v['am'][i].Start = float(ws['am'][i]); v['cm'][i].Start = float(1-ws['am'][i])
    for d in range(nD):
        for i in range(nI):
            v['dc'][d,i].Start = float(ws['dc'][d,i])
            v['v'][i,d].Start = float(ws['am'][i] and ws['dc'][d,i])
            v['w'][i,d].Start = float((not ws['am'][i]) and ws['dc'][d,i])
            for c in range(nC): v['y'][d,i,c].Start = float(ws['y'][d,i,c]>0.5)
        v['n3DP'][d].Start = float(ws['n3dp'][d])
    mdl.update()
    mdl.optimize()
    if mdl.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] or mdl.SolCount == 0:
        return None
    return {'obj': mdl.ObjVal, 'gap': mdl.MIPGap*100, 'am': np.array([v['am'][i].X>0.5 for i in range(nI)])}


def solve_fixed(inst, am_pred, dc_pred, y_pred, n_threads, time_limit=600):
    mdl, v, _ = build_milp(inst, n_threads, time_limit, mip_gap=0.001)
    nI, nD, nC = inst['nI'], inst['nD'], inst['nC']
    for i in range(nI):
        v['am'][i].lb = v['am'][i].ub = float(am_pred[i])
        v['cm'][i].lb = v['cm'][i].ub = float(1 - am_pred[i])
    for d in range(nD):
        for i in range(nI):
            v['dc'][d,i].lb = v['dc'][d,i].ub = float(dc_pred[d,i])
            v['v'][i,d].lb = v['v'][i,d].ub = float(am_pred[i] and dc_pred[d,i])
            v['w'][i,d].lb = v['w'][i,d].ub = float((1-am_pred[i]) and dc_pred[d,i])
            for c in range(nC):
                v['y'][d,i,c].lb = v['y'][d,i,c].ub = float(y_pred[d,i,c])
    mdl.update()
    mdl.optimize()
    if mdl.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] or mdl.SolCount == 0:
        return None
    return {'obj': mdl.ObjVal, 'gap': mdl.MIPGap*100}


# ============================================================================
# REPAIR -- VERSIONE CORRETTA (dc derivato da y)
# ============================================================================
def repair_predictions(am_prob, dc_prob, y_logits, nI, nD, nC):
    am_pred = (am_prob > 0.5).astype(int)
    dc_pred_raw = (dc_prob > 0.5).astype(int)

    for i in range(nI):
        if dc_pred_raw[:, i].sum() == 0:
            d_best = int(np.argmax(dc_prob[:, i]))
            dc_pred_raw[d_best, i] = 1

    y_pred = np.zeros((nD, nI, nC), dtype=int)
    for i in range(nI):
        open_dcs = [d for d in range(nD) if dc_pred_raw[d, i] == 1]
        for c in range(nC):
            scores = y_logits[:, i, c]
            masked = np.array([scores[d] if d in open_dcs else -1e9 for d in range(nD)])
            d_star = int(np.argmax(masked))
            y_pred[d_star, i, c] = 1

    # FIX: dc = f(y), garantisce dc<=sum(y) per costruzione
    dc_pred = (y_pred.sum(axis=2) > 0).astype(int)
    return am_pred, dc_pred, y_pred


# ============================================================================
# GNN v4 -- identica all'originale
# ============================================================================
def load_instance_graph(d):
    nI, nD, nC = d['nI'], d['nD'], d['nC']
    ratio = np.array(d['ratio'], dtype=np.float32)
    c_CM = np.array(d['c_CM'], dtype=np.float32)
    L_CM = np.array(d['L_CM'], dtype=np.float32)
    L_AM = np.array(d['L_AM'], dtype=np.float32)
    b_i = np.array(d['b_i'], dtype=np.float32)
    SL = np.array(d['SL'], dtype=np.float32)
    netto = np.array(d.get('netto', np.zeros(nI)), dtype=np.float32)
    mu_ic = np.array(d['mu_ic'], dtype=np.float32)
    mu_tot = mu_ic.sum(axis=1)
    log_c_CM = np.log10(np.clip(c_CM,1,None)); log_b_i = np.log10(np.clip(b_i,1,None))
    netto_norm = netto/(np.abs(netto).max()+1e-8)
    coord_DC_raw = np.array(d['coord_DC'], dtype=np.float32)
    coord_C_raw = np.array(d['coord_C'], dtype=np.float32)
    mu_sum = mu_tot[:,None]+1e-8
    baricentro = (mu_ic @ coord_C_raw)/mu_sum
    diff = coord_C_raw[None,:,:]-baricentro[:,None,:]
    dist2 = (diff**2).sum(axis=2)
    dispersione = (mu_ic*dist2).sum(axis=1)/mu_sum[:,0]
    diff_dc = coord_DC_raw[None,:,:]-baricentro[:,None,:]
    dist_dc = np.sqrt((diff_dc**2).sum(axis=2))
    DIAG_MAX = np.sqrt(200.0)
    baricentro_norm = baricentro/10.0
    dispersione_norm = np.sqrt(dispersione)/DIAG_MAX
    sku_features = np.stack([ratio/6.0, log_c_CM/4.0, L_CM*52/35.0, L_AM*52/4.0, log_b_i/5.5, SL,
        netto_norm, mu_tot/(mu_tot.max()+1e-8),
        np.full(nI, d['Leas']/30000.0, dtype=np.float32),
        np.full(nI, d['costo_km']/200.0, dtype=np.float32),
        baricentro_norm[:,0], baricentro_norm[:,1], dispersione_norm,
        dist_dc.min(axis=1)/DIAG_MAX, dist_dc.max(axis=1)/DIAG_MAX,
        dist_dc.mean(axis=1)/DIAG_MAX, dist_dc.std(axis=1)/DIAG_MAX], axis=1)
    data = HeteroData()
    data['sku'].x = torch.tensor(sku_features, dtype=torch.float32)
    data['dc'].x = torch.tensor(coord_DC_raw/10.0, dtype=torch.float32)
    data['cli'].x = torch.tensor(coord_C_raw/10.0, dtype=torch.float32)
    src_d, dst_d = [], []
    for i in range(nI):
        for c in range(nC):
            if mu_ic[i,c] > 1e-6: src_d.append(c); dst_d.append(i)
    if src_d:
        data['cli','demands','sku'].edge_index = torch.tensor([src_d,dst_d], dtype=torch.long)
        data['sku','demanded_by','cli'].edge_index = torch.tensor([dst_d,src_d], dtype=torch.long)
    src_p, dst_p = [], []
    for dd in range(nD):
        for c in range(nC): src_p.append(dd); dst_p.append(c)
    data['dc','near','cli'].edge_index = torch.tensor([src_p,dst_p], dtype=torch.long)
    data['cli','near_rev','dc'].edge_index = torch.tensor([dst_p,src_p], dtype=torch.long)
    src_po, dst_po = [], []
    for i in range(nI):
        for j in range(i+1,nI): src_po.extend([i,j]); dst_po.extend([j,i])
    if src_po:
        data['sku','portfolio','sku'].edge_index = torch.tensor([src_po,dst_po], dtype=torch.long)
    data.nI, data.nD, data.nC = nI, nD, nC
    return data


class HeteroGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers; self.dropout = dropout
        self.proj_sku = Linear(17,hidden_dim); self.proj_dc = Linear(2,hidden_dim); self.proj_cli = Linear(2,hidden_dim)
        self.convs = nn.ModuleList(); self.norms_sku = nn.ModuleList(); self.norms_dc = nn.ModuleList(); self.norms_cli = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('cli','demands','sku'): SAGEConv(hidden_dim,hidden_dim),
                ('sku','demanded_by','cli'): SAGEConv(hidden_dim,hidden_dim),
                ('dc','near','cli'): SAGEConv(hidden_dim,hidden_dim),
                ('cli','near_rev','dc'): SAGEConv(hidden_dim,hidden_dim),
                ('sku','portfolio','sku'): SAGEConv(hidden_dim,hidden_dim)}, aggr='sum')
            self.convs.append(conv)
            self.norms_sku.append(nn.LayerNorm(hidden_dim)); self.norms_dc.append(nn.LayerNorm(hidden_dim)); self.norms_cli.append(nn.LayerNorm(hidden_dim))
        self.head_am = nn.Sequential(Linear(hidden_dim,64), nn.ReLU(), nn.Dropout(dropout), Linear(64,1))
        self.head_dc = nn.Sequential(Linear(hidden_dim*2,64), nn.ReLU(), nn.Dropout(dropout), Linear(64,1))
        self.head_y = nn.Sequential(Linear(hidden_dim*3+1,64), nn.ReLU(), nn.Dropout(dropout), Linear(64,1))

    def forward(self, data):
        x_dict = {'sku': self.proj_sku(data['sku'].x), 'dc': self.proj_dc(data['dc'].x), 'cli': self.proj_cli(data['cli'].x)}
        for i, conv in enumerate(self.convs):
            x_new = conv(x_dict, data.edge_index_dict)
            for ntype in x_dict:
                if ntype in x_new:
                    h = x_new[ntype]
                    if ntype=='sku': h = self.norms_sku[i](h)
                    elif ntype=='dc': h = self.norms_dc[i](h)
                    elif ntype=='cli': h = self.norms_cli[i](h)
                    h = F.relu(h); h = F.dropout(h, p=self.dropout, training=self.training)
                    x_dict[ntype] = x_dict[ntype] + h
        return x_dict

    def predict_am(self, x_dict): return self.head_am(x_dict['sku']).squeeze(-1)
    def predict_dc(self, x_dict, nI, nD):
        h_s, h_d = x_dict['sku'], x_dict['dc']
        return self.head_dc(torch.cat([h_s.unsqueeze(0).expand(nD,nI,-1), h_d.unsqueeze(1).expand(nD,nI,-1)], dim=-1)).squeeze(-1)
    def predict_y(self, x_dict, data, nI, nD, nC):
        h_s, h_d, h_c = x_dict['sku'], x_dict['dc'], x_dict['cli']
        dist_dc_cli = torch.cdist(data['dc'].x, data['cli'].x)
        scores = torch.zeros(nD, nI, nC, device=h_s.device)
        for d in range(nD):
            scores[d] = self.head_y(torch.cat([h_s.unsqueeze(1).expand(nI,nC,-1),
                h_d[d].unsqueeze(0).unsqueeze(0).expand(nI,nC,-1),
                h_c.unsqueeze(0).expand(nI,nC,-1),
                dist_dc_cli[d].unsqueeze(0).expand(nI,nC).unsqueeze(-1)], dim=-1)).squeeze(-1)
        return scores


# ============================================================================
# VALIDATE INSTANCE -- con timing isolato
# ============================================================================
def validate_instance(fpath, model):
    with open(fpath) as f:
        d = json.load(f)
    nI = d['nI']
    t_start_total = time.time()

    inst = inst_from_json(d)
    time_limit = get_time_limit(nI)

    # 1. Z_true (CG + MILP esatto)
    t0 = time.perf_counter()
    ws = column_generation(inst, n_threads=THREADS_PER_SOLVE)
    res_true = solve_free(inst, ws, n_threads=THREADS_PER_SOLVE, time_limit=time_limit, mip_gap=MIP_GAP_TARGET)
    t_ztrue = time.perf_counter() - t0
    if res_true is None:
        print(f"  #{d['instance_id']} (nI={nI}): Z_true INFEASIBLE", flush=True)
        return None
    Z_true = res_true['obj']

    # 2. Forward pass GNN (SOLO inferenza)
    graph = load_instance_graph(d)
    t0 = time.perf_counter()
    with torch.no_grad():
        x_dict = model(graph)
        am_prob = torch.sigmoid(model.predict_am(x_dict)).numpy()
        dc_prob = torch.sigmoid(model.predict_dc(x_dict, nI, d['nD'])).numpy()
        y_logits = model.predict_y(x_dict, graph, nI, d['nD'], d['nC']).numpy()
    t_gnn = time.perf_counter() - t0

    # 3. Repair (CORRETTA)
    t0 = time.perf_counter()
    am_pred, dc_pred, y_pred = repair_predictions(am_prob, dc_prob, y_logits, nI, d['nD'], d['nC'])
    t_repair = time.perf_counter() - t0

    # 4. Z_gnn (solve_fixed)
    t0 = time.perf_counter()
    res_gnn = solve_fixed(inst, am_pred, dc_pred, y_pred, n_threads=THREADS_PER_SOLVE)
    t_fixed = time.perf_counter() - t0

    elapsed_total = time.time() - t_start_total

    timing = {'t_ztrue_s': t_ztrue, 't_gnn_inference_s': t_gnn,
              't_repair_s': t_repair, 't_solve_fixed_s': t_fixed}

    if res_gnn is None:
        print(f"  #{d['instance_id']} (nI={nI}): Z_gnn INFEASIBLE (residuo, non dovrebbe più accadere "
              f"con la repair corretta) [t_gnn={t_gnn*1000:.1f}ms]", flush=True)
        return {'instance_id': d['instance_id'], 'nI': nI, 'Z_true': Z_true, 'Z_gnn': None,
                'gap_pct': None, 'time': elapsed_total, **timing}

    Z_gnn = res_gnn['obj']
    gap_pct = 100*(Z_gnn - Z_true)/Z_true

    print(f"  #{d['instance_id']} (nI={nI}): Z_true={Z_true:,.0f}, Z_gnn={Z_gnn:,.0f}, "
          f"gap_ML={gap_pct:+.2f}%, t_tot={elapsed_total:.1f}s "
          f"[t_gnn={t_gnn*1000:.1f}ms, t_repair={t_repair*1000:.1f}ms, t_fixed={t_fixed:.2f}s]", flush=True)

    return {'instance_id': d['instance_id'], 'nI': nI, 'Z_true': Z_true, 'Z_gnn': Z_gnn,
            'gap_pct': gap_pct, 'time': elapsed_total, **timing}


def save_incremental(results, path):
    """Riscrive il file dei risultati -- chiamata dopo OGNI istanza."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp_path, path)  # scrittura atomica: mai un file corrotto a metà


def already_done(results, instance_id):
    return any(r['instance_id'] == instance_id for r in results)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("="*78); print("OPTIMALITY GAP VALIDATION v2 -- repair corretta + timing + save incrementale"); print("="*78)

    files = sorted(glob.glob(os.path.join(DATASET_DIR, "instance_*.json")))
    print(f"Trovati {len(files)} file in {DATASET_DIR}")
    dims = []
    for f in files:
        with open(f) as fh: dims.append(json.load(fh)['nI'])
    indices = list(range(len(files)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=dims, random_state=SEED)
    temp_dims = [dims[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_dims, random_state=SEED)
    test_files = [files[i] for i in test_idx]
    test_dims = [dims[i] for i in test_idx]
    print(f"Test set: {len(test_files)} istanze (STESSO split dell'originale, SEED={SEED})")

    small_medium = [f for f,n in zip(test_files, test_dims) if n <= 20]
    print(f"  Piccole/medie (nI<=20): {len(small_medium)} -- validazione piena")
    print(f"  Grandi (nI>20): {sum(1 for n in test_dims if n>20)} -- ESCLUSE (istanze di riferimento separate)")

    # Carica risultati parziali se esiste già un file da run precedenti
    # (permette di riprendere se questo stesso script viene interrotto)
    results = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        print(f"Trovato file parziale esistente: {len(results)} istanze già processate, si riprende da lì")

    # Carica modello UNA VOLTA
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = HeteroGNN(HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    t_start = time.time()
    for fpath in small_medium:
        with open(fpath) as fh:
            iid_check = json.load(fh)['instance_id']
        if already_done(results, iid_check):
            continue  # già fatta in un run precedente interrotto

        res = validate_instance(fpath, model)
        if res is not None:
            results.append(res)
            save_incremental(results, OUTPUT_FILE)  # <-- SALVA DOPO OGNI ISTANZA

    elapsed = time.time() - t_start

    # Riepilogo
    valid = [r for r in results if r.get('gap_pct') is not None]
    print(f"\n{'='*78}\nRIEPILOGO ({elapsed/60:.1f} min per le istanze processate in questa esecuzione)\n{'='*78}")
    print(f"Istanze validate: {len(valid)}/{len(results)} (su {len(small_medium)} target)")
    if valid:
        gaps = [r['gap_pct'] for r in valid]
        print(f"Gap ML medio: {np.mean(gaps):.2f}%  |  mediano: {np.median(gaps):.2f}%  |  max: {np.max(gaps):.2f}%")
        print(f"Gap <=5%: {sum(1 for g in gaps if g<=5)}/{len(gaps)}")
        print("\nPer dimensione:")
        for nI_val in sorted(set(r['nI'] for r in valid)):
            sub = [r['gap_pct'] for r in valid if r['nI']==nI_val]
            print(f"  nI={nI_val:>3}: gap medio={np.mean(sub):.2f}%, n={len(sub)}")

        print(f"\n{'='*78}\nTEMPI ISOLATI\n{'='*78}")
        t_gnn_all = [r['t_gnn_inference_s'] for r in valid]
        t_repair_all = [r['t_repair_s'] for r in valid]
        t_fixed_all = [r['t_solve_fixed_s'] for r in valid]
        t_ztrue_all = [r['t_ztrue_s'] for r in valid]
        print(f"Forward pass GNN:   media={np.mean(t_gnn_all)*1000:.2f}ms  mediana={np.median(t_gnn_all)*1000:.2f}ms")
        print(f"Repair:             media={np.mean(t_repair_all)*1000:.2f}ms  mediana={np.median(t_repair_all)*1000:.2f}ms")
        print(f"Solve_fixed (Z_gnn):media={np.mean(t_fixed_all):.2f}s  mediana={np.median(t_fixed_all):.2f}s")
        print(f"Z_true (MILP esatto):media={np.mean(t_ztrue_all):.1f}s  mediana={np.median(t_ztrue_all):.1f}s")
        speedup = np.mean(t_ztrue_all) / (np.mean(t_gnn_all)+np.mean(t_repair_all)+np.mean(t_fixed_all))
        print(f"\nSpeedup medio pipeline GNN vs MILP esatto: {speedup:.1f}x")

    print(f"\nSalvato in {OUTPUT_FILE}")

    # Istanze di riferimento 50/100 SKU (facoltativo, come nell'originale)
    print(f"\n{'='*78}\nISTANZE DI RIFERIMENTO 50/100 SKU\n{'='*78}")
    if os.path.exists(REFERENCE_CASE_JSON):
        with open(REFERENCE_CASE_JSON) as f:
            ref_results = json.load(f)
        import sys
        sys.path.insert(0, '.')
        from reference_case_50_100 import generate_instance as gen_ref_instance
        for rr in ref_results:
            nI, nD, nC = rr['nI'], rr['nD'], rr['nC']
            Z_true_ref = rr['milp_obj']
            inst = gen_ref_instance(nI, nD, nC, seed=SEED_BASE)
            d_fake = {'nI':nI,'nD':nD,'nC':nC,'ratio':inst['ratio'].tolist(),'c_CM':inst['c_CM'].tolist(),
                      'L_CM':inst['L_CM'].tolist(),'L_AM':inst['L_AM'].tolist(),'b_i':inst['b_i'].tolist(),
                      'SL':inst['SL'].tolist(),'mu_ic':inst['mu_ic'].tolist(),'Leas':inst['Leas'],
                      'costo_km':inst['costo_km'],'coord_DC':inst['coord_DC'].tolist(),'coord_C':inst['coord_C'].tolist()}
            graph = load_instance_graph(d_fake)
            t0 = time.perf_counter()
            with torch.no_grad():
                x_dict = model(graph)
                am_prob = torch.sigmoid(model.predict_am(x_dict)).numpy()
                dc_prob = torch.sigmoid(model.predict_dc(x_dict, nI, nD)).numpy()
                y_logits = model.predict_y(x_dict, graph, nI, nD, nC).numpy()
            t_gnn_ref = time.perf_counter() - t0
            am_pred, dc_pred, y_pred = repair_predictions(am_prob, dc_prob, y_logits, nI, nD, nC)
            res_gnn = solve_fixed(inst, am_pred, dc_pred, y_pred, n_threads=THREADS_PER_SOLVE)
            if res_gnn:
                gap_ref = 100*(res_gnn['obj']-Z_true_ref)/Z_true_ref
                print(f"  nI={nI}: Z_true={Z_true_ref:,.0f} (gap MIP originale {rr['milp_gap']:.1f}%), "
                      f"Z_gnn={res_gnn['obj']:,.0f}, gap_ML={gap_ref:+.2f}%, t_gnn={t_gnn_ref*1000:.1f}ms")
            else:
                print(f"  nI={nI}: Z_gnn INFEASIBLE anche con repair corretta")
    else:
        print(f"  {REFERENCE_CASE_JSON} non trovato -- salta questa sezione")
