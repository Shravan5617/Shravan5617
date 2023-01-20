import sys
import numpy as np

from trajectory import SingleExcitationWithCollectiveCoupling


plotResult = False
printOutput = False
if '--print' in sys.argv:
    printOutput = True
if '--plot' in sys.argv: 
    plotResult=True
    from matplotlib import pyplot as plt
    #plt.style.use('classic')
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='Times New Roman', size='10')


if 'param.in' in sys.argv:
    exec(open('param.in').read())
else:
    dt = 0.001
    Ntimes = 30000
    Nskip = 10

    Nmol = 101
    Wmol =  0.0
    Wgrd = -1.0
    Vndd = -0.3

    Wcav = 0.0 + 2.0*Vndd
    Vcav = 0.0
    Kcav = 0  #Kcav*pi/Nmol

    useStaticNeighborDisorder = False
    useDynamicNeighborDisorder = False
    DeltaNN = 0.0
    TauNN = 0.0

    useStaticDiagonalDisorder = False
    useDynamicDiagonalDisorder = False
    DeltaDD = 0.0
    TauDD = 0.0

    
model1 = SingleExcitationWithCollectiveCoupling(Nmol,0)

model1.initialHamiltonian_Cavity_nonHermitian(Wgrd,Wcav,Wmol,Vndd,Vcav,Kcav,Gamma=0.0)

if useStaticNeighborDisorder:
    model1.updateNeighborStaticDisorder(DeltaNN)
if useDynamicNeighborDisorder:
    model1.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
if useStaticDiagonalDisorder:
    model1.updateDiagonalStaticDisorder(DeltaDD)
if useDynamicDiagonalDisorder:
    model1.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)

# model1.initialCj_Bright()
# model1.initialCj_middle()
model1.initialCj_middle()
# model1.initialCj_Gaussian(2.0)
# model1.initialCj_Cavity()
# model1.initialCj_Boltzman(hbar,kBT,most_prob=True)
# model1.initialCj_Polariton()

times = []
Pmol1, Pmol2 = [], []
IPR1, IPR2 = [], []

Xj_list, Vj_list = [], []
distr_list = []
Displacement_list = []
Correlation_list = []
Current_list = []
for it in range(Ntimes):

    if useDynamicNeighborDisorder:
        model1.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
    if useDynamicDiagonalDisorder:
        model1.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)

    Javg, CJJ = model1.getCurrentCorrelation()

    model1.propagateCj_RK4(dt)
    model1.propagateJ0Cj_RK4(dt)
    
    if it%Nskip==0:
        times.append( it*dt )
        Pmol1.append( model1.getPopulation_system()  )
        # Pmol2.append( model2.getPopulation_system() )

        IPR1.append( model1.getIPR() )
        # IPR2.append( model2.getIPR() )


        # Xj_list.append(model1.Xj)
        # Vj_list.append(model1.Vj)
        distr = np.abs(model1.Cj[model1.Imol:model1.Imol+model1.Nmol])**2
        distr_list.append(distr[:,0])
        Displacement_list.append(model1.getDisplacement())
        Correlation_list.append(CJJ)
        Current_list.append(Javg)
        if printOutput:
            print("{t}\t{d}\t{dP}".format(t=it*dt,d=model1.getDisplacement(),dP=model1.getPopulation_system()))
                                                # dE=model1.getEnergy()-E0 ))

if not plotResult:
    # write to output 
    fpop = open('Pmol.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fpop.write("{t}\t{Pmol}\n".format(t=times[it],Pmol=Pmol1[it]))

    fdis = open('Displacement.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fdis.write("{t}\t{Displacement}\n".format(t=times[it],Displacement=Displacement_list[it]))

    fcorr = open('Correlation.dat'+sys.argv[-1], 'w')
    for it in range(len(times)):
        fcorr.write("{t}\t{Corr_real}\t{Corr_imag}\n".format(t=times[it],Corr_real=np.real(Correlation_list[it]),Corr_imag=np.imag(Correlation_list[it])))

    # fwfn = open('wfn.dat'+sys.argv[-1], 'w')
    # for j in range(model1.Nmol):
    #     fwfn.write(str(j))
    #     for i in range(len(distr_list)):
    #         fwfn.write('\t'+'{:2.10f}'.format(distr_list[i][j]))
    #     fwfn.write('\n')

if plotResult:
    from scipy import stats, special

    distr_list = np.array(distr_list)

    fig, ax= plt.subplots(1,7, figsize=(18.0,3.0))
    ax[0].plot(times,Pmol1, '-r', lw=2, label='Q matrix', alpha=0.7)
    # ax[0].plot(times,Pmol2, '-k', lw=2, label='Explicit', alpha=0.7)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("$P_{mol}$")
    ax[0].legend()

    # # cuts = [1.0,9.0]
    # cuts = [0.0,Ntimes*dt]
    # cut1=np.argmin(np.abs(np.array(times)-cuts[0]))
    # cut2=np.argmin(np.abs(np.array(times)-cuts[1]))

    # res = stats.linregress(np.log10(times[cut1:cut2]), np.log10(np.array(Displacement_list)[cut1:cut2]))
    # slope = res.slope
    # intercept = res.intercept
    # print(slope)

    # ax[1].plot(times,np.array(Displacement_list)/np.array(Pmol1)*Pmol1[0])
    ax[1].plot(times,np.array(Displacement_list),label='numerical')
    # ax[1].plot(times[cut1:cut2],10.0**(slope*np.log10(times[cut1:cut2])+intercept),'--',label=r'fitting: $\alpha=$'+str(round(slope,2))
    # for j in range(Nmol):
    #     ax2[1].plot(Xj_list[j],Vj_list[j])
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('Displacement')

    Xgrid = np.array(range(len(distr_list[0])))
    Ygrid = times
    Xgrid, Ygrid = np.meshgrid(Xgrid, Ygrid)
    Zgrid = distr_list
    CS = ax[2].contourf(Xgrid, Ygrid, Zgrid, 100, cmap=plt.cm.jet)
    ax[2].set_xlabel('sites')
    ax[2].set_ylabel('time')

    times = np.array(times) #+dt*Nskip
    
    Xmean = np.zeros(len(times))
    Xvar = np.zeros(len(times))
    # for x_ind in range(int(Nmol/2),int(Nmol/2)+5):
    pop_ana_list = []
    for x_ind in range(Nmol):
        index = x_ind-int(Nmol/2)
        ln = ax[3].plot(times,distr_list[:,x_ind],alpha=0.5,label='numerical')
        lncolor = ln[0].get_color()
        
        if Vcav==0.0:
            pop_ana = special.jv(index,2*np.abs(Vndd)*times)**2
        else:
            Omega_ana = 0.5*(Wcav + Wmol - 2.0*np.abs(Vndd))
            Delta_ana = 0.5*(Wcav - Wmol + 2.0*np.abs(Vndd))
            energy_gap = np.sqrt(Delta_ana**2 + np.abs(Vcav)**2*Nmol)
            coef_ana = np.exp(-1j*Wmol*times)*(1j**index)*special.jv(index,2*np.abs(Vndd)*times) \
                    - np.exp(-1j*(Wmol-2*np.abs(Vndd))*times)/Nmol \
                    + np.exp(-1j*Omega_ana*times)*(np.cos(energy_gap*times)+1j*Delta_ana/energy_gap*np.sin(energy_gap*times))/Nmol
            pop_ana = np.abs(coef_ana)**2
            # if index==0: 
            #     print(distr_list[:,x_ind])
            #     print(pop_ana)
        pop_ana_list.append(pop_ana)
        ax[3].plot(times,pop_ana,'--',color=lncolor,lw=2,alpha=0.5,label='analytical')
    #     Xmean = Xmean + (x_ind) * pop_ana
    #     Xvar = Xvar + (x_ind)**2 * pop_ana
    # ax[1].plot(times,Xvar-Xmean**2,'--k',label="direct Bessel summation")
    ax[3].set_xlabel('time')
    ax[3].set_ylabel('population')

    ### calculate the displacement using analytical result. 
    pop_ana_list = np.array(pop_ana_list).T
    displace_ana = []
    for it in range(len(times)):
        Rj = np.array(range(Nmol))
        R =  np.abs( np.sum( Rj       *pop_ana_list[it]) ) 
        R2 = np.abs( np.sum((Rj-R)**2 *pop_ana_list[it]) )     
        displace_ana.append(R2)
    ax[1].plot(times,displace_ana,'--k',label="direct Bessel summation")
    ax[1].plot(times,0.5*(2*np.abs(Vndd)*np.array(times))**2,':',label='analytical (no cavity)')
    ax[1].legend()

    W,U = np.linalg.eigh(model1.Ht0)
    eigen_num = np.delete(W, np.where(W == Wgrd))
    if Vcav==0.0:
        eigen_num = np.delete(eigen_num, np.where(eigen_num == Wcav))
        # print(eigen_num)
        eigen_ana = Wmol-2.0*np.abs(Vndd)*np.cos(2*np.pi*np.arange(1,Nmol+1)/Nmol)
        print('Check eigenvalues:\n',np.isclose(eigen_num,np.sort(eigen_ana)))
    else:
        eigen_ana = Wmol-2.0*np.abs(Vndd)*np.cos(2*np.pi*np.arange(1,Nmol+1)/Nmol)
        eigen_ana = np.delete(eigen_ana, np.where(eigen_ana == Wmol-2.0*np.abs(Vndd)))
        energy_gap = np.sqrt(0.25*(Wcav - Wmol + 2.0*np.abs(Vndd))**2+np.abs(Vcav)**2*Nmol)
        eigen_ana = np.append(eigen_ana,0.5*(Wcav + Wmol - 2.0*np.abs(Vndd)) + energy_gap)
        eigen_ana = np.append(eigen_ana,0.5*(Wcav + Wmol - 2.0*np.abs(Vndd)) - energy_gap)
        print('Check eigenvalues:\n',np.isclose(eigen_num,np.sort(eigen_ana)))
    if Vcav!=0.0:
        Wcav_max = 5.0
        Wcav_list = np.linspace(-Wcav_max, Wcav_max, num=101)
        eigen_ana_list = []
        for iWcav in range(len(Wcav_list)):
            eigen_ana = Wmol-2.0*np.abs(Vndd)*np.cos(2*np.pi*np.arange(1,Nmol+1)/Nmol)
            eigen_ana = np.delete(eigen_ana, np.where(eigen_ana == Wmol-2.0*np.abs(Vndd)))
            energy_gap = np.sqrt(0.25*(Wcav_list[iWcav] - Wmol + 2.0*np.abs(Vndd))**2+np.abs(Vcav)**2*Nmol)
            eigen_ana = np.append(eigen_ana,0.5*(Wcav_list[iWcav] + Wmol - 2.0*np.abs(Vndd)) + energy_gap)
            eigen_ana = np.append(eigen_ana,0.5*(Wcav_list[iWcav] + Wmol - 2.0*np.abs(Vndd)) - energy_gap)
            eigen_ana_list.append(eigen_ana)
        eigen_ana_list = np.array(eigen_ana_list).T
        for i in range(len(eigen_ana)):
            ax[4].plot(Wcav_list,eigen_ana_list[i],'-')
        ax[4].plot(Wcav_list,(Wmol-2.0*np.abs(Vndd))*np.ones(len(Wcav_list)),'--k')
        ax[4].plot(Wcav_list,Wcav_list,'--r')
        ax[4].set_xlabel('$\omega_c$')
        ax[4].set_ylabel('eigenenergy')
        # ax[4].axvline(x=Wcav,ls='--')

        # ax[5].plot(times,np.array(displace_ana)-0.5*(2*np.abs(Vndd)*np.array(times))**2)
        # Delta_ana = 0.5*(Wcav - Wmol + 2.0*np.abs(Vndd))
        # shift = (Nmol-1)*((Nmol-1)+1)*(2*(Nmol-1)+1)/6/(Nmol-1)**2 \
        #         *(Delta_ana**2/(Delta_ana**2+np.abs(Vcav)**2*Nmol)+1.0)
        # ax[5].plot(times,np.ones(len(times))*shift,'--')
        # ax[5].set_xlabel('time')
        # ax[5].set_ylabel('$\Delta$ displacement')
        # ax[1].plot(times,0.5*(2*np.abs(Vndd)*np.array(times))**2+shift,':')
        ax[5].plot(times,np.real(Correlation_list))
        ax[5].set_xlabel('time')
        ax[5].set_ylabel('current correlation')
        ax[6].plot(times,np.real(Current_list),alpha=0.5)
        ax[6].set_xlabel('time')
        ax[6].set_ylabel('current')
    else:# WITHOUT CAVITY
        ax[5].plot(times,np.real(Correlation_list),alpha=0.5)
        ax[5].set_xlabel('time')
        ax[5].set_ylabel('current correlation')
        ax[6].plot(times,np.real(Current_list),alpha=0.5)
        ax[6].set_xlabel('time')
        ax[6].set_ylabel('current')

    
    # ax[1].set_xscale('log')
    # ax[1].set_yscale('log')
    plt.tight_layout()
    plt.show()  
