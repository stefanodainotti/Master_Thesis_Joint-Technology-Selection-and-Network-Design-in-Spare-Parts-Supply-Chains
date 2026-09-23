#!/usr/bin/env python3
"""
================================================================================
CASO DI RIFERIMENTO 50 e 100 SKU -- gap minimo, budget esteso simmetrico
================================================================================

Obiettivo: ottenere UNA istanza pulita e ben ottimizzata per ciascuna
dimensione (50 e 100 SKU), da usare come caso di riferimento verificato nel
capitolo Risultati -- analogo ai casi 20 SKU / 9 SKU reale / 6 SKU ablation
gia' usati altrove nella tesi. Non e' il test di scalabilita' sui tempi
(quello resta scalability_test_50_100.py) -- qui l'obiettivo e' il gap piu'
basso possibile, non il tempo di risoluzione.

STESSO SEED dello scalability test (SEED_BASE=42): stessa istanza, non una
nuova -- coerenza tra i due esperimenti, si osserva solo l'effetto di piu'
tempo sullo stesso problema.

TUTTI I FIX APPLICATI (identico a scalability_test_50_100.py):
  1. NO doppio conteggio raw material (n_i include gia' il materiale).
  2. Base-stock tecnologia-specifico (McCormick).
  3. Costo di backorder analitico da lead time (bo_LT, PWL + Big-M).

BUDGET: 8 ore per istanza, simmetrico, esecuzione sequenziale (1 processo,
20 thread) -- totale fino a 16 ore. MIP_GAP target 1%.

NOTA IMPORTANTE: se scalability_test_50_100.py e' ancora in esecuzione,
ATTENDI che finisca prima di lanciare questo script -- due job Gurobi
pesanti in contemporanea su queste dimensioni rischiano Out of Memory
(gia' osservato in precedenza).

LANCIO (VM, in background persistente):
  cd percorso_progetto_Tesi
  $env:GRB_LICENSE_FILE = "$env:USERPROFILE\gurobi.lic"
  Start-Process -FilePath ".\ramp_env\Scripts\python.exe" \`
    -ArgumentList "reference_case_50_100.py" \`
    -RedirectStandardOutput "log_reference_case.txt" \`
    -RedirectStandardError "log_reference_case_err.txt" \`
    -WindowStyle Hidden
================================================================================
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import poisson
import time, json, os

# ============================================================================
# CONFIG
# ============================================================================
THREADS = 20
N_SCENARIOS = 30
N_PWL = 20
MIP_GAP_TARGET = 0.01     # 1%
SEED_BASE = 42

# (nI, nD, nC, time_limit_secondi)
CONFIGS = [
    (50,  4, 8,  2 * 3600),   # PILOT: 2 ore (era 8h) -- test rapido con tuning
    (100, 4, 10, 2 * 3600),   # PILOT: 2 ore (era 8h)
]

H_D_VAL, CAP_3DP = 0.25, 5000.0
O_RAW, C_RAW, DEN_RAW, UNIT_RAW = 30.0, 50.0, 1200.0, 0.5
L_RAW, SL_RAW, O_VAL = 1.0/52, 0.85, 50.0

PROFILES = {
    'A': {'c_CM': (2000,5000), 'ratio': (0.35,0.70), 'L_CM': (20,35), 'L_AM': (1,2),
          'b_i': (30000,100000), 'SL': (0.95,0.99), 'weight': 0.20},
    'B': {'c_CM': (1000,3000), 'ratio': (0.70,1.20), 'L_CM': (12,25), 'L_AM': (1,2.5),
          'b_i': (10000,50000), 'SL': (0.90,0.97), 'weight': 0.25},
    'C': {'c_CM': (500,1500), 'ratio': (1.00,2.00), 'L_CM': (8,18), 'L_AM': (1.5,3),
          'b_i': (3000,15000), 'SL': (0.85,0.95), 'weight': 0.25},
    'D': {'c_CM': (100,500), 'ratio': (2.00,4.00), 'L_CM': (3,10), 'L_AM': (2,3.5),
          'b_i': (500,5000), 'SL': (0.80,0.92), 'weight': 0.20},
    'E': {'c_CM': (50,200), 'ratio': (3.50,6.00), 'L_CM': (2,7), 'L_AM': (2.5,4),
          'b_i': (50,1000), 'SL': (0.75,0.88), 'weight': 0.10},
}


# ============================================================================
# FUNZIONI ANALITICHE
# ============================================================================
def expected_backorder(mu_L, beta):
    if mu_L < 1e-9 or beta <= 0: return mu_L
    return max(mu_L*(1-poisson.cdf(beta-1, mu_L)) - beta*(1-poisson.cdf(beta, mu_L)), 0.0)

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
    lam = mu_val * L_val
    if lam <= 1e-9: return 0
    for s in range(100000):
        if poisson.cdf(s, lam) >= SL_target: return int(s)
    return 99999


# ============================================================================
# GENERAZIONE ISTANZA (stessa logica del dataset GNN -- profili A-E)
# ============================================================================
def generate_instance(nI, nD, nC, seed):
    rng = np.random.default_rng(seed)
    profile_names = list(PROFILES.keys())
    weights = np.array([PROFILES[p]['weight'] for p in profile_names])
    weights = weights / weights.sum()
    sku_profiles = rng.choice(profile_names, size=nI, p=weights)

    c_CM = np.zeros(nI); ratio = np.zeros(nI); L_CM = np.zeros(nI)
    L_AM = np.zeros(nI); b_i = np.zeros(nI); SL = np.zeros(nI)
    for i in range(nI):
        p = PROFILES[sku_profiles[i]]
        c_CM[i] = rng.uniform(*p['c_CM'])
        ratio[i] = rng.uniform(*p['ratio'])
        L_CM[i] = rng.uniform(*p['L_CM']) / 52.0
        L_AM[i] = rng.uniform(*p['L_AM']) / 52.0
        b_i[i] = rng.uniform(*p['b_i'])
        SL[i] = rng.uniform(*p['SL'])

    sort_idx = np.argsort(ratio)
    c_CM, ratio, L_CM, L_AM, b_i, SL = (c_CM[sort_idx], ratio[sort_idx], L_CM[sort_idx],
                                          L_AM[sort_idx], b_i[sort_idx], SL[sort_idx])
    c_AM = c_CM * ratio
    n_i = ratio

    coord_DC = rng.uniform(0, 10, size=(nD, 2))
    coord_C = rng.uniform(0, 10, size=(nC, 2))
    dist_dc_c = np.sqrt(((coord_DC[:,None,:]-coord_C[None,:,:])**2).sum(axis=2))

    Leas = rng.uniform(5000, 30000)
    costo_km = rng.uniform(20, 200)

    mu_totals = 3.0 / (1 + c_CM/2000)
    alpha_dir = rng.uniform(1, 5)
    mu_ic = np.zeros((nI, nC))
    for i in range(nI):
        w = rng.dirichlet(np.ones(nC)*alpha_dir)
        mu_ic[i,:] = mu_totals[i] * w * nC

    h_d = np.full(nD, H_D_VAL)
    t_dic = np.stack([dist_dc_c * costo_km] * nI, axis=1)
    L_AM_mat = np.column_stack([L_AM]*nD)

    return {'nI':nI,'nD':nD,'nC':nC,'c_CM':c_CM,'ratio':ratio,'c_AM':c_AM,'n_i':n_i,
            'L_CM':L_CM,'L_AM':L_AM,'L_AM_mat':L_AM_mat,'b_i':b_i,'SL':SL,
            'mu_ic':mu_ic,'Leas':Leas,'costo_km':costo_km,'coord_DC':coord_DC,
            'coord_C':coord_C,'dist_dc_c':dist_dc_c,'t_dic':t_dic,'h_d':h_d}


# ============================================================================
# COLUMN GENERATION (warm-start)
# ============================================================================
def column_generation(inst, n_threads):
    t0 = time.time()
    nI, nD, nC = inst['nI'], inst['nD'], inst['nC']
    p = inst
    table = {}
    for i in range(nI):
        table[i] = {}
        for d in range(nD):
            table[i][d] = {}
            for tech in ['CM','AM']:
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
                        # NO consumo diretto raw (n_i include gia' il materiale)
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
    return {'am':am_a,'dc':dc_s,'y':y_s,'B':B_s,'n3dp':n3,'time':time.time()-t0}


# ============================================================================
# MILP STOCASTICO COMPLETO (bo_LT + BS tech-specifico, NO cost_raw diretto)
# ============================================================================
def solve_stochastic(inst, warm_start, n_threads, time_limit):
    t0 = time.time()
    nI, nD, nC = inst['nI'], inst['nD'], inst['nC']
    nS = N_SCENARIOS
    I,D,C,S = range(nI),range(nD),range(nC),range(nS)
    pi_s = 1.0/nS
    c_CM,SL_i,b_i_arr = inst['c_CM'],inst['SL'],inst['b_i']
    L_CM,L_AM_mat = inst['L_CM'],inst['L_AM_mat']
    mu_ic,t_dic,h_d = inst['mu_ic'],inst['t_dic'],inst['h_d']
    Leas = inst['Leas']
    n_i,c_AM = inst['n_i'],inst['c_AM']
    prod_i = c_AM
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

    mdl = gp.Model("scalability"); mdl.setParam("OutputFlag",1)
    mdl.setParam("MIPGap", MIP_GAP_TARGET); mdl.setParam("TimeLimit", time_limit)
    mdl.setParam("Threads", n_threads)
    mdl.setParam("MIPFocus", 3)   # PILOT: focus sul bound, non sull'incumbent
    mdl.setParam("Cuts", 2)       # PILOT: cutting planes aggressivi

    am = mdl.addVars(nI, vtype=GRB.BINARY); cm = mdl.addVars(nI, vtype=GRB.BINARY)
    y = mdl.addVars(nD,nI,nC, vtype=GRB.BINARY); dc = mdl.addVars(nD,nI, vtype=GRB.BINARY)
    v = mdl.addVars(nI,nD, vtype=GRB.BINARY); w = mdl.addVars(nI,nD, vtype=GRB.BINARY)

    # PILOT: branching priority esplicita sulle binarie che, dall'analisi
    # della traiettoria, sembrano sbloccare il salto di bound a meta' albero
    # (nodo ~9.900-11.400 nelle run precedenti). Priorita' piu' alta =
    # Gurobi le decide prima nel branching.
    for i in range(nI):
        am[i].BranchPriority = 10
        cm[i].BranchPriority = 10
    for i in range(nI):
        for d in range(nD):
            v[i,d].BranchPriority = 8
            w[i,d].BranchPriority = 8

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

    # --- Obiettivo: NO cost_raw (consumo diretto) -- n_i include gia' il materiale ---
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

    # --- Warm-start minimale dalla CG ---
    ws = warm_start
    for i in I: am[i].Start=float(ws['am'][i]); cm[i].Start=float(1-ws['am'][i])
    for d in D:
        ha=False
        for i in I:
            dc[d,i].Start=float(ws['dc'][d,i])
            v[i,d].Start=float(ws['am'][i] and ws['dc'][d,i])
            w[i,d].Start=float((not ws['am'][i]) and ws['dc'][d,i])
            if ws['am'][i] and ws['dc'][d,i]: ha=True
            for c in C: y[d,i,c].Start=float(ws['y'][d,i,c]>0.5)
        dcAM[d].Start=float(ha)
    for d in D:
        pk=0
        for s in S:
            hrs = sum(a_i[i]*max(0.0, sum(d_ics[i,c,s] for c in C if ws['y'][d,i,c]>0.5)-float(ws['B'][i,d]))
                      for i in I if ws['am'][i] and ws['dc'][d,i])
            pk = max(pk, int(np.ceil(hrs/CAP_3DP)) if hrs>0 else 0)
        n3DP[d].Start=float(max(ws['n3dp'][d], pk))
    mdl.setParam("StartNodeLimit", 100)
    mdl.update()

    print(f"\n  [MILP] Avvio ottimizzazione (nI={nI}, time_limit={time_limit}s)...", flush=True)
    mdl.optimize()
    solve_time = time.time()-t0

    if mdl.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        return None

    am_sol = np.array([am[i].X>0.5 for i in I])
    n_am = int(am_sol.sum())
    return {'am': am_sol, 'n_am': n_am, 'obj': mdl.ObjVal, 'gap': mdl.MIPGap*100,
            'time': solve_time, 'status': mdl.Status}


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    results = []
    print("="*78)
    print("CASO DI RIFERIMENTO -- 50 e 100 SKU, gap minimo, budget esteso")
    print(f"nS={N_SCENARIOS}, N_PWL={N_PWL}, MIP_GAP_target={MIP_GAP_TARGET*100:.0f}%")
    print("="*78)

    for nI, nD, nC, time_limit in CONFIGS:
        print(f"\n{'#'*78}")
        print(f"# ISTANZA: nI={nI}, nD={nD}, nC={nC}, time_limit={time_limit/3600:.0f}h")
        print(f"{'#'*78}")

        t_inst_start = time.time()
        inst = generate_instance(nI, nD, nC, seed=SEED_BASE)

        print(f"\n[CG] Avvio column generation warm-start...")
        ws = column_generation(inst, n_threads=THREADS)
        cg_n_am = int(ws['am'].sum())
        print(f"[CG] Completata in {ws['time']:.1f}s -- n_AM={cg_n_am}/{nI}")

        res = solve_stochastic(inst, ws, n_threads=THREADS, time_limit=time_limit)

        if res is None:
            print(f"\n[MILP] INFEASIBLE o errore per nI={nI}")
            results.append({'nI':nI,'nD':nD,'nC':nC,'status':'INFEASIBLE'})
            continue

        total_time = time.time()-t_inst_start
        print(f"\n{'='*78}")
        print(f"RISULTATO nI={nI}: n_AM={res['n_am']}/{nI}, gap={res['gap']:.2f}%, "
              f"obj={res['obj']:,.0f} EUR")
        print(f"Tempo CG: {ws['time']:.1f}s | Tempo MILP: {res['time']:.1f}s | "
              f"Totale: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Concordanza CG-MILP: n_AM CG={cg_n_am} vs MILP={res['n_am']} "
              f"(diff={res['n_am']-cg_n_am:+d})")
        print(f"{'='*78}")

        results.append({
            'nI': nI, 'nD': nD, 'nC': nC,
            'cg_time': ws['time'], 'cg_n_am': cg_n_am,
            'milp_time': res['time'], 'milp_n_am': res['n_am'],
            'milp_gap': res['gap'], 'milp_obj': res['obj'],
            'total_time': total_time, 'status': str(res['status']),
        })

        # Salva incrementalmente (nel caso il secondo run fallisca/venga interrotto)
        with open('pilot_tuned_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*78}")
    print("RIEPILOGO FINALE")
    print(f"{'='*78}")
    print(f"{'nI':>5} {'CG(s)':>8} {'MILP(s)':>10} {'Tot(min)':>9} {'gap%':>6} {'n_AM':>6}")
    for r in results:
        if r.get('status') == 'INFEASIBLE':
            print(f"{r['nI']:>5}  INFEASIBLE")
            continue
        print(f"{r['nI']:>5} {r['cg_time']:>8.1f} {r['milp_time']:>10.1f} "
              f"{r['total_time']/60:>9.1f} {r['milp_gap']:>6.2f} {r['milp_n_am']:>6d}")

    print(f"\nRisultati completi salvati in pilot_tuned_results.json")
