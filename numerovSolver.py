# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:21:09 2020

@author: Kurz, Niels

tested on harmonic oscillator, morse potential and square well potential.
"""

'some standard imports for plotting, fitting etc.'
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from scipy.optimize import curve_fit, root_scalar, fmin, least_squares
from scipy.interpolate import interp1d, splev, splrep



def get_delta(E, meff, func, N, pos_array, psi, matching_index):
    
    dx = (pos_array[1]-pos_array[0])
    dx2 = dx**2
    func_vals = func(pos_array, meff, E)
    a = np.abs(func_vals)
    #determine crossings of energy with potential energy curve
    imiddle = np.argmax(func_vals)
    ileft = np.argmin(a)
    iright = imiddle + np.argmin(a[imiddle:])
    mask = (pos_array >= ((5/np.sqrt(a)) + pos_array[iright]))
    mask2 = (pos_array <= (pos_array[ileft]-5/np.sqrt(a)))
     # determine indices for integration range
    if(len(pos_array[mask2]) == 0):
        imin = 0
    else:
        imin = np.argmin(np.abs(pos_array - pos_array[mask2][-1]))
    if(len(pos_array[mask]) == 0):
        imax = N-1
    else:
        imax = np.argmin(np.abs(pos_array - pos_array[mask][0]))
    imin = 0
    imax = N-1
    ic = matching_index
    psileft = np.zeros(np.shape(pos_array))
    psileft[imin] = psi[0]
    psileft[imin+1] = psi[1]
    psileft[:imin] = 0
    psiright = np.zeros(np.shape(pos_array))
    psiright[imax] = psi[-1]
    psiright[imax-1] = psi[-2]
    psiright[imax+1:] = 0

    
    'apply shooting method to solve the differential equation for the given initial value'
    i = imin+1
    while i <= imax-1:
        psileft[i+1] = (2*(1-5*dx2/12*func_vals[i])*psileft[i]-(1+dx2/12*func_vals[i-1])*psileft[i-1])/(1+dx2/12*func_vals[i+1])
        i += 1
    i = imax-1
    while i >= imin+1:    
        psiright[i-1] = (2*(1-5*dx2/12*func_vals[i])*psiright[i]-(1+dx2/12*func_vals[i+1])*psiright[i+1])/(1+dx2/12*func_vals[i-1])
        i -= 1
        
    #calculate logarithmic derivatives
    derivleft = np.gradient(psileft, pos_array)
    derivright = np.gradient(psiright, pos_array)
    
    gammal = derivleft[ic]
    gammar = derivright[ic]
    
    if(np.isnan(gammal) or np.isnan(gammar)):
        print("Something is nan")
        print("Energy guess = ", E)
        print("Boundary conditions: ", psileft[imin], psileft[imin+1], psiright[imax], psiright[imax-1])
        print("imin = ", imin, " imax = ", imax)
        print("gammal, gammar: ", gammal, gammar)
        print("derivleft[ic], derivright[ic]: ", derivleft[ic], derivright[ic])
        print("psileft[ic], psiright[ic]: ", psileft[ic], psiright[ic])
   
    error = 0
    error = gammar-gammal
    
    return error

def bisection(meff, E0, delta0, E1, delta1, func, N, pos_array, psi_value, matching_index,max_it = 5000):
    '''Assuming, that between E1 and E2 the true energy eigenvalue is to be found, determines it by repeatedly integrating
    psi(x) until the energy dependent deviation to the boundary value at the end of the integration range is smaller than
    1E-6'''  
    
    i = 0
    while(i < max_it):
        E2 = (E0+E1)/2.0
        delta2 = get_delta(E2, meff, func, N, pos_array, psi_value, matching_index)
        if(np.abs(delta2)<1e-8 and np.abs(E1-E0)<1e-10):
            i = max_it
        i += 1
        #setting new interval
        if(delta2*delta0 > 0):
            E0 = E2
            delta0 = delta2
        else:
            E1 = E2
            delta1 = delta2
    
    return (delta0, delta1, E0)
 
# pot_func assumes a potential energy with one definite minimum
def get_integration_range(E, kin_energy, neg_exp_left=200, neg_exp_right=80):
    global meff, pos_array
    xmiddle = pos_array[np.argmax(kin_energy(pos_array, meff, E))]
    f = lambda x: kin_energy(np.array([x]), meff, E)[0]
    xleft = root_scalar(f, method='brentq', bracket=[pos_array[0], xmiddle]).root
    xright = root_scalar(f, method='brentq', bracket=[xmiddle, pos_array[-1]]).root
    fl = lambda x: (xleft - neg_exp_left/(abs(kin_energy(np.array([x]),meff, E)[0]))-x)
    if (fl(pos_array[0])*fl(xleft)<0):
        xleftlimit = root_scalar(fl, method='brentq', bracket=[pos_array[0], xleft]).root                  
        xleftlimit = pos_array[np.argmin(np.abs(pos_array-xleftlimit))]
    else:
        xleftlimit = pos_array[0]
    fr = lambda x: (xright + neg_exp_right/(abs(kin_energy(np.array([x]), meff, E)[0]))-x)
    if (fr(xright)*fr(pos_array[-1])<0):
        xrightlimit = root_scalar(fr, method='brentq', bracket=[xright, pos_array[-1]]).root
        xrightlimit = pos_array[np.argmin(np.abs(pos_array-xrightlimit))]
    else:
        xrightlimit = pos_array[-1]
    return (xleftlimit, xrightlimit)      

def getUpperEigenstate(meff,E, dE, kin_energy, N, pos_array, psi_value, matching_index, max_it = 5000):
    """Returns the next-higher energy of the energy eigenstate of the equation psi''(x)=func(x)psi(x), by 
    integrating func(x) on the integration range given by pos_array using Numerov's method.
    Two neighboring starting values of psi(x), motivated by boundary conditions, at the beginning of the 
    integration range are needed. At the end of the integration range a third boundary value bound_val_end 
    is used to check if the normalized solution corresponds to an eigen energy."""
       
    #find two energies which frame the true eigen energy
    E0 = E+dE
    E1 = E+2*dE
    delta0 = get_delta(E0, meff, kin_energy, N, pos_array, psi_value, matching_index)
    delta1 = get_delta(E1,meff, kin_energy, N, pos_array, psi_value, matching_index)
    
    i=0
    while ((delta0*delta1 > 0) and (i < max_it)):
        E0 += dE
        E1 += dE
        # calculate delta
        delta0 = get_delta(E0, meff, kin_energy, N, pos_array, psi_value, matching_index)
        delta1 = get_delta(E1, meff, kin_energy, N, pos_array, psi_value, matching_index)
        i +=1
           
    return root_scalar(lambda E: get_delta(E, meff, kin_energy, N, pos_array, psi_value, matching_index)
    , method='brentq', bracket=[E0, E1], rtol=0.00000001).root
    
def returnEigenWaveFunction(meff, E, func, N, pos_array, psi, matching_index):
    
    dx = (pos_array[1]-pos_array[0])
    dx2 = dx**2
    func_vals = func(pos_array, meff, E)
    ic = matching_index
    psileft = np.zeros(np.shape(pos_array))
    psileft[0] = psi[0]
    psileft[1] = psi[1]
    psiright = np.zeros(np.shape(pos_array))
    psiright[-1] = psi[-1]
    psiright[-2] = psi[-2]
    N = np.shape(pos_array)[0]
    'apply shooting method to solve the differential equation for the given initial value'
    i = 1
    while i <= N-2:
        psileft[i+1] = (2*(1-5*dx2/12*func_vals[i])*psileft[i]-(1+dx2/12*func_vals[i-1])*psileft[i-1])/(1+dx2/12*func_vals[i+1])
#        if (np.abs(psileft[i+1])>1):
#            psileft /= psileft[i+1]
        i += 1
    i = N-2
    while i >= 1 :    
        psiright[i-1] = (2*(1-5*dx2/12*func_vals[i])*psiright[i]-(1+dx2/12*func_vals[i+1])*psiright[i+1])/(1+dx2/12*func_vals[i-1])
#        if (np.abs(psiright[i-1])>1):
#            psiright /= psiright[i-1]
        i -= 1    
    
    eigenpsi = np.zeros(np.shape(psi))
    eigenpsi[:ic] = psileft[:ic]
    eigenpsi[ic:] = np.abs(psileft[ic]/psiright[ic])*psiright[ic:]
    
    eigenpsi /= np.sqrt(np.trapz(eigenpsi**2, pos_array))
    
    return (psileft, psiright, eigenpsi)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)


##### all information about potential energy curve that you
#### want to calculate vibrational wave functions for #########
# from I. Schmidt-Mink et. al. / Ground- and excited state properties of
# Li_2 and Li_2^+
#triplett potentials of li_2 aconnected by selection rules, given in 10^(-6)*Hartree
r0, pot1, pot2, pot3, pot4, pot5= np.loadtxt("singlett_li2_dipole_allowed_pecs.csv", unpack=True)
pot6, pot7, pot8, pot9, pot10 = np.loadtxt("triplett_li2_dipole_allowed_pecs2", unpack=True, usecols=(1,2,3,4,5))
# lowest energetical li2+ potential
pot11 = np.loadtxt("li2plus_pecs", unpack=True, usecols=(1,))


# convert into Hartrees
pot1 *= 10**(-6)
pot6 *= 10**(-6)
pot11 *= 10**(-6)

# dissociation energies of in hartrees

# X^1S_g^+  according to J. Chem. Phys. 131, 204309 200
dissE = 8516.774/219474.631363 # dissociation energy of X^1S_g^+ in Hartrees, pot1 is X^1S_g+

dissE2 = 322/219474.631363 # dissociation energy of A^3S_u^+ in Hartrees, pot6
dissE3 = 10441/219474.631363 # dissociation energy of 1^2S_g^+ in Hartrees, pot11

#shift potentials by dissociation energies given above
pot1 += dissE
pot6 += dissE2
pot11 += dissE3

# equilibrium bond length in Bohr radii
# from I. Schmidt-Mink et. al. / Ground- and excited state properties of
# Li_2 and Li_2^+
bond_length1 = 5.0550174  # X^1S_g^+
bond_length2 = 7.9028347 # A^3S_u^+
bond_length3 = 5.8562613 #12S_g^+

N = 1000
psi_val = np.zeros(N, dtype='f8')
xlim_left, xlim_right = 1,40
pos_array = np.linspace(xlim_left, xlim_right, N, dtype='f8')
dx = pos_array[1]-pos_array[0]
#ax.plot(r0, pot6, 'ro')

##################################################################

# long-range dispersion coefficients for Li-Li for two ground state atoms
# from , Phys. Rev. A 54, 2824 1996.
c6 = 1393.39 # a.u.
c8 = 83425.8
c10 = 7372100

coeffs = np.zeros(11)
coeffs[6] = c6
coeffs[8] = c8
coeffs[10] = c10
coeffs = tuple(coeffs)

# returns value of a polynomial of the form
# coeffs[0]*func**0+coeffs[1]*func*+1 + ...
# at position pos
def funcPoly(pos, func, *coeffs):
    coeffs = np.array(list(coeffs))
    powers = np.arange(np.shape(coeffs)[0])
    return np.sum(np.transpose(coeffs*np.power(np.transpose(np.tile(func(pos),[np.shape(powers)[0],1])), powers)), axis=0)

#define long range potential
def u(r):
    global coeffs
    return funcPoly(r, lambda x: 1/x, *coeffs)


# returns function used in definition of MLR model
def y(pos, p, r_ref):
    return (pos**p-r_ref**p)/(pos**p + r_ref**p)

# exponential polynomial in definition of MLR model
def beta(pos, De, r_ref, p, q, *coeffs):
    global r_eq
    beta_inf = np.log(2*De/u(r_eq)) 
    return (y(pos, p, r_ref)*beta_inf+(1-y(pos, p, r_ref))*funcPoly(pos, lambda x: y(x, q, r_ref), *coeffs))

# good potential energy model commonly used in modern spectroscopy
# thoroughly explained in J. Chem. Phys. 131, 204309 200
# p = usually chosen bigger than the difference between largest and smallest
# inverse power in the long range potential u(r)
# q < p is a small value
# r_ref = can roughly be chosen between 1.1r_eq and 1.5r_eq
def MLR_pot(pos, De, r_ref, *coeff_list,p=4, q=2):
    global r_eq
    return De*(1- u(pos)/u(r_eq)*np.exp(-beta(pos, De, r_ref, p, q, *coeff_list)*y(pos, p, r_eq)))**2

# #fit long range morse potential to e.g. A^3S_u^+, X1sg^+
r_eq = bond_length2
rref = 1.3*r_eq

# # use of least_squares
# # function to be minimized
# # x0 is a numpy array with parameters
# def F(x0):
#     global r0, dissE, dissE2, dissE3, rref
#     return (MLR_pot(r0, dissE2, rref, *tuple(x0))-pot6)
# x0_init = np.array([0])
# #res = least_squares(F, x0_init, method='trf', loss='soft_l1')
# res = least_squares(F, x0_init, method='lm')
# for i in np.arange(13):
#     x0_init = np.array(tuple(res.x)+(0,))
#     try:
#         res = least_squares(F, x0_init, method='lm')
#         #res = least_squares(F, x0_init, method='trf', loss='soft_l1')
#     except ValueError:
#         break
#     except RuntimeError:
#         break
# print(res.x)
# print(res.cost)

#pickle.dump(res.x, open("x1sgplus.pot","wb"))
# pickle.dump(res.x, open("a3suplus.pot", "wb"))
# pickle.dump(res.x, open("one2sgplus.pot", "wb"))

a3suplus_pot = lambda pos: MLR_pot(pos, dissE2, 1.3*bond_length2, *tuple(pickle.load(open("a3suplus.pot", "rb"))),p=4, q=2)
x1sgplus_pot = lambda pos: MLR_pot(pos, dissE, 1.6*bond_length1, *tuple(pickle.load(open("x1sgplus.pot", "rb"))), p=4, q=3)
one2sgplus_pot = lambda pos: MLR_pot(pos, dissE3, 1.3*bond_length3, *tuple(pickle.load(open("one2sgplus.pot", "rb"))), p=4, q=3)

#this function returns the finite well potential
#depending on its total energy E in finite square well potential
# in the limiting case of an infinitely deep square well, the
# energy levels are given by En = pi**2/(2*meff*a**2)
def square_well(x,De,a,x0=0):
    y = np.empty(np.shape(x))
    y[np.abs(x-x0) <= 0.5*a] = 0
    y[np.abs(x-x0) > 0.5*a] = De
    return y

def get_En_infinite_well(n, a):
    global meff
    return 2*(n*0.5*np.pi)**2/(meff*a**2)

# analytic eigenvalues given by E_n = (n+1/2)
def harmonic_pot(x):
    return 0.5*x**2

# morse potential, not used in modern spectroscopy
# analytic eigenvalues given by 
# E_n = a*np.sqrt(2*De/meff)*(n+0.5)-(a*np.sqrt(2*De/meff)*(n+0.5))**2/(4*De))
def morsePot(r, De, a, re):
    return De*(1- np.exp(-a*(r-re)))**2

# return energy levels of morse potential for given parameters
def get_En_morsePot(n,De,a):
    global meff
    return a*np.sqrt(2*De/meff)*(n+0.5)-(a*np.sqrt(2*De/meff)*(n+0.5))**2/(4*De)
    
#chosen_potential, meff = harmonic_pot, 1
#De, a, re, meff = 5, 1, 10, 1
#nu0 = a/(2*np.pi)*np.sqrt(2*De/meff)
#chosen_potential = lambda x: morsePot(x, De, a, re)
#De, a, x0, meff = 50000, 10, 0, 1
#chosen_potential = lambda x: square_well(x, De, a, x0)
# Optimal parameters fit of MLR model to A3S_^+-potential:
# r_ref = 1.3 r_eq, N=13, p=4, q=2 @ cost = 1.70129E-9
chosen_potential, meff = a3suplus_pot, 5468.67
#chosen_potential, meff = x1sgplus_pot, 5468.67
#chosen_potential, meff = one2sgplus_pot, 5468.67

# plot_factor = 0.0002
# pot_along_array = chosen_potential(pos_array)
# ax.set_xlim(xlim_left,xlim_right)
# ax.set_ylim(0, 2*1.5*(pot_along_array[-1]-pot_along_array[np.argmin(pot_along_array)]))
# ax.plot(pos_array, pot_along_array)




def kin_energy(r,meff,E):
    global params, chosen_potential
    return 2*meff*(E-chosen_potential(r))


#negative logarithms of relative decays (determines integration range)
eps_left, eps_right = 150, 100

E,EigenE, n,Emax = 0,0,0,dissE2
dE = Emax/1000
psi_val[0], psi_val[1], psi_val[-1], psi_val[-2] = 0.0, (-1.0)**n, 0.0, 1.0


wavefunctions = []
energies = []


# the matching index must be chosen with care, if one of the
# wavefunctions exhibits a zero crossing at the matching position,
# a slight variation in energy will lead to a zero crossing for 
# the energy dependent deviation of the logarithmic derivative of the 
# left and right wavefunction, although the logarithmic derivatives
# don't match at all
# therefore it's best to choose the matching point somewhere near or at the
# beginning of the integration range to obtain the highest precision
# on the determined energy eigenvalues
# while E < Emax:
#     E = EigenE+dE
#     if E <= Emax:
#         xmin, xmax = get_integration_range(E, kin_energy, eps_left, eps_right)
#         integration_range = np.arange(xmin, xmax, dx)
#         N = np.shape(integration_range)[0]
#         match_index = 1
#         psi_val = np.zeros(N, dtype='f8')
#         psi_val[0], psi_val[1], psi_val[-1], psi_val[-2] = 0.0, (-1.0)**n, 0.0, 1.0
#         EigenE = getUpperEigenstate(meff, E, dE, kin_energy, N, integration_range, psi_val, match_index)
#         psileft,psiright,psi_tmp =  returnEigenWaveFunction(meff, EigenE, kin_energy, N, integration_range, psi_val, match_index)
#     if (EigenE <= Emax):
#           print("energy guess = ", E)
#           print("n, psi_val[0], psi_val[1], psi_val[-1], psi_val[-2]: ", n, psi_val[0], psi_val[1], psi_val[-1], psi_val[-2])
#           print("EigenE = ", EigenE, "Hartrees")
#           # conversion from Hartrees into cm-1
#           print(EigenE*219474.631363, " cm^-1")
#           # print analytic energies of morse potential
#           #print("Morse potential: ", get_En_morsePot(n, De, a))
#           #print analytic energies of infinite square well
#           #print("Infinite well: ", get_En_infinite_well(n, a))
#           energies.append(EigenE)
#           wavefunctions.append((integration_range, psi_tmp))
#           ax.plot(integration_range, EigenE + plot_factor*psi_tmp)
#           if (n>=1):
#               dE = (energies[-1]-energies[-2])/5.0
#           n+=1
# save energies and wavefunctions in picklet object          
#pickle.dump(zip(energies,wavefunctions), open("x1sgplus_solutions.dat", "wb"))
#pickle.dump(zip(energies,wavefunctions), open("a3suplus_solutions.dat", "wb"))
#pickle.dump(zip(energies,wavefunctions), open("one2sgplus_solutions.dat", "wb"))


solutions_singlett = list(pickle.load(open("x1sgplus_solutions.dat", "rb")))
solutions_triplett = list(pickle.load(open("a3suplus_solutions.dat", "rb")))
solutions_li2ion = list(pickle.load(open("one2sgplus_solutions.dat", "rb")))
#ax.plot(np.array(solutions_singlett[-1][1][0]), solutions_singlett[-1][0]+0.04*np.array(solutions_singlett[-1][1][1]))
#ax.plot(np.array(solutions_triplett[-1][1][0]), solutions_triplett[-1][0]+0.04*np.array(solutions_triplett[-1][1][1]))

highest_singlett_pos, highest_singlett_wav = np.array(solutions_singlett[-1][1][0]), np.array(solutions_singlett[-1][1][1])
ax.plot(highest_singlett_pos, highest_singlett_wav, label='Highest singlett wavefunction')
highest_triplett_pos, highest_triplett_wav = np.array(solutions_triplett[-1][1][0]), np.array(solutions_triplett[-1][1][1])
ax.plot(highest_triplett_pos, highest_triplett_wav, label='Highest triplett wave function')

#ax.plot(solutions_singlett[0][1][0], solutions_singlett[0][1][1])
#ax.plot(solutions_triplett[0][1][0], solutions_triplett[0][1][1])
#ax.legend(fontsize=18)

# #this function returns the franck-condon overlap between two
# # vibrational wavefunctions
def getOverlap(pos1, pos2, wf1, wf2):
    dx = pos1[1]-pos1[0]
    minpos = max(pos1[0], pos2[0])
    maxpos = min(pos1[-1], pos2[-1])
    pos_new = np.arange(minpos, maxpos,dx)
    imin1 = np.argmin(np.abs(pos1-minpos))
    imin2 = np.argmin(np.abs(pos2-minpos))
    imax1 = np.argmin(np.abs(pos1-maxpos))
    imax2 = np.argmin(np.abs(pos2-maxpos))
    #print(imin1, imin2, imax1, imax2)
    #print(np.shape(wf1[imin1:imax1]))
    #print(np.shape(wf2[imin2:imax2]))
    #print(np.shape(pos_new))
    return np.trapz(wf1[imin1:imax1]*wf2[imin2:imax2], pos1[imin1:imax1])**2
    # if(pos_new[0]==pos2[0]):
    #     wf2_new = wf2[:len(pos_new)]
    #     # re-calculate wf1 whose domain is pos1 for new domain pos_new
    #     # calculate minimum shift between the new and the old integration range
    #     imin = np.argmin(np.abs(pos_new[0]-pos1))
    #     a = pos_new[0] - pos1[imin]
    #     m = np.gradient(wf1[imin:imin+len(pos_new)], pos1[imin:imin+len(pos_new)])
    #     wf1_new = wf1[imin:imin+len(pos_new)]+a*m        
    # else:
    #     wf1_new = wf1[:len(pos_new)]
    #     # re-calculate wf2 whose domain is pos2 for new domain pos
    #     imin = np.argmin(np.abs(pos_new[0]-pos2))
    #     a = pos_new[0] - pos2[imin]
    #     m = np.gradient(wf2[imin:imin+len(pos_new)], pos2[imin:imin+len(pos_new)])
    #     wf2_new = wf2[imin:imin+len(pos_new)]+a*m
        
#     return np.trapz(wf1_new*wf2_new, pos_new)**2

factor_w_singlett = []
factor_w_triplett = []

for elem in solutions_li2ion:
    factor_w_singlett.append(getOverlap(elem[1][0],highest_singlett_pos,elem[1][1], highest_singlett_wav))
    factor_w_triplett.append(getOverlap(elem[1][0],highest_triplett_pos,elem[1][1], highest_triplett_wav))
        
fig2 = plt.figure(figsize=(10,10))
ax2 = fig2.add_subplot(111)
#ax2.plot(wavefunctions[0][0], wavefunctions[0][1])

ax2.plot(factor_w_singlett, label='Overlap w highest wf of singulett potential')       
ax2.plot(factor_w_triplett, label='Overlap w highest wf of triplett potential')        
ax2.legend(fontsize=18)