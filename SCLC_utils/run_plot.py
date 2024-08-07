######### Package Imports #########################################################################
import os, uuid, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from joblib import Parallel, delayed

from SCLC_utils.JV_steady_state import *
from SCLC_utils.SCLC_funcs import *
from SCLC_utils.addons import *
# from pySIMsalabim.plots.plot_def import *
# from pySIMsalabim.experiments.JV_steady_state import *

timeout = str(9 * 60 ) # 9 minutes
######### Functions #################################################################################

# function to run the simulation with joblib
def run(ID, cmd_pars,simss_device_parameters,session_path=os.path.join(os.getcwd(), 'SIMsalabim','SimSS')):
    # cwd = os.getcwd()
    # session_path = os.path.join(os.getcwd(), 'SIMsalabim','SimSS')
    G_fracs = None
    JV_file_name = os.path.join(session_path,'JV.dat')

    res = run_SS_JV(simss_device_parameters,session_path,JV_file_name,G_fracs,parallel=True,force_multithreading=True,cmd_pars=cmd_pars, UUID=ID)

def run_all(L_pero, eps_r_pero, ions_pero, N_t_bulk_list, Vmax, Vscan, NP, tolPois, tolDens, couplePC, minAcc, maxAcc, grad, simss_device_parameters, str_save, with_TL, rerun=False, ions_in_TLs =False,figsize = (21,15),to_plot=[1e21,5e21,1e22,1.5e22]):

    res_dir = os.path.join(os.getcwd(),'results')

    nons_list,ninf_list,nend_list,mu  = [],[],[],[]
   
    if rerun:
        cmd_pars_list, ID_list = [], []
        # prepare cmd_pars_list
        if with_TL:
            layer = 'l2'
        else:
            layer = 'l1'
        for n in N_t_bulk_list:
            cmd_pars = [{'par': layer+'.L', 'val': str(L_pero)},{'par':layer+'.N_t_bulk','val':str(n)},{'par':layer+'.N_anion','val':str(ions_pero)},{'par':layer+'.N_cation','val':str(ions_pero)},{'par':'Vmax','val':str(Vmax)},{'par':'NP','val':str(NP)},{'par':'Vscan','val':str(Vscan)},{'par':'timeout','val':timeout},{'par':'tolPois','val':str(tolPois)},{'par':'tolDens','val':str(tolDens)},{'par':'couplePC','val':str(couplePC)},{'par':'minAcc','val':str(minAcc)},{'par':'maxAcc','val':str(maxAcc)},{'par':'grad','val':str(grad)}]
            if ions_in_TLs:
                cmd_pars.append({'par':'l1.ionsMayEnter','val':str(1)})
                cmd_pars.append({'par':'l3.ionsMayEnter','val':str(1)})
            cmd_pars_list.append(cmd_pars)
            ID_list.append(str(uuid.uuid4()))

        # run the simulation with joblib
        Parallel(n_jobs=min(len(cmd_pars_list),15))(delayed(run)(ID, cmd_pars,simss_device_parameters) for ID, cmd_pars in zip(ID_list, cmd_pars_list))

        # store values in a csv file
        df = pd.DataFrame({'ID':ID_list,'N_t_bulk':N_t_bulk_list})
        df.to_csv(os.path.join(res_dir,f'{L_pero*1e9}_'+str_save+'_N_t_bulk_list.csv'),index=False)
    else:
        df = pd.read_csv(os.path.join(res_dir,f'{L_pero*1e9}_'+str_save+'_N_t_bulk_list.csv'))
        N_t_bulk_list = df['N_t_bulk'].values
        ID_list = df['ID'].values

    # plot the results
    colors = plt.cm.viridis(np.linspace(0,1,len(N_t_bulk_list)+1))
    # remove the last color
    colors = colors[:-1]
    #invert colors
    colors = colors[::-1]
    fig = plt.figure(figsize=figsize)
    ax1,ax2,ax3,ax4 = plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224)
    # ax = plt.gca()
    minJ,maxJ = 1e20,1e-20
    slopemax = 0
    idx = -len(to_plot)
    idx =-1
    minVminMG,maxVmaxMG = 1e3,0
    for i,n in enumerate(N_t_bulk_list):
        ID = ID_list[i]
        JV_file_name = os.path.join('SIMsalabim','SimSS',f'JV_{ID}.dat')
        data_JV = pd.read_csv(JV_file_name,sep=r'\s+')
        label = sci_notation(n/1e6,0)

        data_JV = data_JV[data_JV['Vext']>0]
        data_JV = data_JV[data_JV['Jext']>0]
        minJ = min(minJ,data_JV['Jext'].min())
        maxJ = max(maxJ,data_JV['Jext'].max())
        if Vscan == -1:
            # reverse the data
            data_JV = data_JV.iloc[::-1]

        V_slopef,J_slopef,slopesf,get_tangentf,idx_maxf,max_slopesf,tang_val_V1f,tang_val_V2f,tang_val_V3f,V1f,J1f,V2f,J2f,Vinf,Jinf = SCLC_get_data_plot(data_JV['Vext'],data_JV['Jext'])
        # print(max_slopesf)
        if max_slopesf > 2:
            nons_list.append(calc_net_charge(V1f,L_pero,eps_r_pero)/1e6)
            ninf_list.append(calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6)
            nend_list.append(calc_net_charge(V2f,L_pero,eps_r_pero)/1e6)
        else:
            nons_list.append(np.nan)
            ninf_list.append(np.nan)
            nend_list.append(np.nan)

        ## Calc Voltages
        Vnet = calc_Vnet_with_ions(ions_pero,n,L_pero,eps_r_pero)
        Vtfl = calc_Vtfl(n,L_pero,eps_r_pero)
        Vsat = calc_Vsat(L_pero,5e24,0,eps_r_pero,295)
        Ntmin = calc_nt_min(L_pero,eps_r_pero,295)
        Vmin = calc_Vtfl(Ntmin,L_pero,eps_r_pero)

        # get mobility
        data_JV_mott = copy.deepcopy(data_JV)
        
        if Vsat > Vtfl:
            data_JV_mott['slopesf'] = slopesf
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=1]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']<=Vsat]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=Vtfl]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']<=2.2]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']>=1.8]
            if len(data_JV_mott) <= 5:
                mu.append(0)
                continue
            VminMG = min(data_JV_mott['Vext'])
            VmaxMG = max(data_JV_mott['Vext'])
            Mott_Gurney_fit = fit_MottGurney(data_JV_mott['Vext'],data_JV_mott['Jext'],1e-4,eps_r_pero,0,L_pero,var2fit=['mu'])
            mu.append(Mott_Gurney_fit[0])
            # print('Mott_Gurney_fit:',Mott_Gurney_fit)
        else:
            mu.append(0)

        # remove Vext = 0 from data_JV
        if n in to_plot:
            data_JV = data_JV[data_JV['Vext']!=0]
            ax1.plot(data_JV['Vext'],data_JV['Jext']/10,label=label,color=colors[i])
            if max_slopesf > 2:
                slopemax = max(slopemax,max(slopesf))
                ax1.plot(V1f,J1f/10,marker='s',color=colors[i])
                ax1.plot(Vinf,Jinf/10,marker='^',color=colors[i])
                ax1.plot(V2f,J2f/10,marker='o',color=colors[i])
                ax1.plot(V_slopef,tang_val_V1f/10,'--',color=colors[i])
                ax1.plot(V_slopef,tang_val_V2f/10,'--',color=colors[i])
                ax1.plot(V_slopef,tang_val_V3f/10,'--',color=colors[i])
            minVminMG = min(minVminMG,VminMG)
            maxVmaxMG = max(maxVmaxMG,VmaxMG)
            ax2.plot(V_slopef,slopesf,label=label,color=colors[i])
            ax2.axvspan(VminMG, VmaxMG, color=colors[i], alpha=0.3,zorder=-idx)
            idx -= 1


        if max_slopesf > 2:
            ax3.plot(n/1e6,calc_net_charge(V1f,L_pero,eps_r_pero)/1e6,marker='s',color=colors[i])
            ax3.plot(n/1e6,calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6,marker='^',color=colors[i])
            ax3.plot(n/1e6,calc_net_charge(V2f,L_pero,eps_r_pero)/1e6,marker='o',color=colors[i])
        # add diagonal line
        ax3.plot([1e20/1e6,1e23/1e6],[1e20/1e6,1e23/1e6],'-',color='black')
        if ions_pero > 0:
            # plot diagonal - ions
            ax3.plot([(ions_pero+1e21)/1e6,1e23/1e6],[1e21/1e6,(1e23-ions_pero)/1e6],'--',color='black')
        # grey shadow ntmin to ntmax
        ax3.axvspan(1e20/1e6, calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.5,zorder=-3)
        # ax3.axvspan(calc_nt_min(L_pero,eps_r_pero,295)/1e6,3*calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.05,zorder=-2)

    ax1.set_yscale('log')
    # ax1.set_ylim([1e-8,1e7])
    ax1.set_ylabel('Current density [mA m$^{-2}$]')
    ax1.set_xscale('log')
    ax1.set_xlim([1e-2,Vmax])
    ax1.set_ylim([10**np.floor(np.log10(minJ)-1),10**np.floor(np.log10(maxJ)+1)])
    ax1.set_xlabel('Applied voltage [V]')
    ax1.grid(True,which='both',axis='both',color='gray', linestyle='-')

    ax2.set_yscale('linear')
    ax2.set_ylim([0,int(slopemax)+1])
    ticks = ax2.get_yticks()
    ticks = ticks[ticks>3]
    ticks = np.append(ticks,[1,2])

    ticks = np.unique(ticks)
    ticks = np.sort(ticks)
    ax2.set_yticks(ticks)

    ax2.axhline(y=1, color='k', linestyle='-')
    ax2.axhline(y=2, color='k', linestyle='-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Applied voltage [V]')
    ax2.set_ylabel('Slope [-]')
    ax2.set_xlim([1e-2,Vmax])
    ax2.set_ylim([0,int(slopemax)+1])
    ax2.grid(True,which='both',axis='both',color='gray', linestyle='-')
    # add arrow between VminMG and VmaxMG
    ax2.annotate('', xy=(minVminMG, int(slopemax)+1), xytext=(maxVmaxMG, int(slopemax)+1), arrowprops=dict(arrowstyle='<->',color='black',linewidth=3))

    ax2.text(minVminMG + 10**((np.log10(minVminMG)+np.log10(maxVmaxMG))/2), int(slopemax)+1.5, 'MG region', fontsize=20, ha='center')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylabel('Calc. density [cm$^{-3}$]')
    ax3.set_xlabel('Trap density [cm$^{-3}$]')
    ax3.grid(True,which='both',axis='both',color='gray', linestyle='-',zorder=-1)
    ax3.set_xlim([1e20/1e6,5e22/1e6])
    ax3.set_ylim([1e20/1e6,5e22/1e6])

    # custum legend
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='V$_{ons}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='V$_{inf}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='V$_{end}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], linestyle='-', color='k', label='N$_T$', markersize=10)]
    if ions_pero > 0:
        legend_elements.append(Line2D([0], [0], linestyle='--', color='k', label='N$_{net}}$', markersize=10))
    ax3.legend(handles=legend_elements, loc='upper left')

    ticks = []
    for i in N_t_bulk_list:
        ticks.append(sci_notation(i/1e6, sig_fig=1))
        # ticks.append(sci_notation(i*1e4, sig_fig=-1))
    mu = np.asarray(mu)*1e4
    ax4.bar(np.arange(len(mu)),mu,color=colors)
    # plt.bar(loop,mu*1e4,color=colors[:ind],width=loop[1]-loop[0])
    ax4.axhline(1,linestyle='-',color='k')
    ax4.set_xticks(np.arange(len(mu)), ticks,fontsize=20,rotation=45)

    ax4.set_ylabel('Mobility [cm$^{2}$V$^{-1}$s$^{-1}$]')
    ax4.set_xlabel('Trap density [cm$^{-3}$]')
    ax4.set_yscale('log')
    ax4.set_ylim([1e-4 , 10])
    ax4.grid(b=True,which='both',axis='y',zorder=-1)

    # add super title to axes
    ax1.set_title('a)',position=(-0.2,0.97))
    ax2.set_title('b)',position=(-0.25,0.97))
    ax3.set_title('c)',position=(-0.2,0.95))
    ax4.set_title('d)',position=(-0.25,0.95))
    plt.tight_layout()


    plt.savefig(os.path.join(res_dir,f'{L_pero*1e9}_'+str_save+'_N_t_bulk_list.png'),dpi=300)
    
def run_all_ions(L_pero, eps_r_pero, ions_bulk_list, traps_pero, Vmax, Vscan, NP, tolPois, tolDens, couplePC, minAcc, maxAcc, grad, simss_device_parameters, str_save, with_TL, rerun=False, ions_in_TLs =False,figsize = (21,15),to_plot=[1e21,5e21,1e22]):
    
    res_dir = os.path.join(os.getcwd(),'results')

    nons_list,ninf_list,nend_list,mu  = [],[],[],[]
    ions_bulk_list = np.asarray(ions_bulk_list)
    if rerun:
        cmd_pars_list, ID_list = [], []
        # prepare cmd_pars_list
        if with_TL:
            layer = 'l2'
        else:
            layer = 'l1'
        for ions in ions_bulk_list:
            cmd_pars = [{'par': layer+'.L', 'val': str(L_pero)},{'par':layer+'.N_t_bulk','val':str(traps_pero)},{'par':layer+'.N_anion','val':str(ions)},{'par':layer+'.N_cation','val':str(ions)},{'par':'Vmax','val':str(Vmax)},{'par':'NP','val':str(NP)},{'par':'Vscan','val':str(Vscan)},{'par':'timeout','val':timeout},{'par':'tolPois','val':str(tolPois)},{'par':'tolDens','val':str(tolDens)},{'par':'couplePC','val':str(couplePC)},{'par':'minAcc','val':str(minAcc)},{'par':'maxAcc','val':str(maxAcc)},{'par':'grad','val':str(grad)}]
            if ions_in_TLs:
                cmd_pars.append({'par':'l1.ionsMayEnter','val':str(1)})
                cmd_pars.append({'par':'l3.ionsMayEnter','val':str(1)})
            cmd_pars_list.append(cmd_pars)
            ID_list.append(str(uuid.uuid4()))

        # run the simulation with joblib
        Parallel(n_jobs=min(len(cmd_pars_list),15))(delayed(run)(ID, cmd_pars,simss_device_parameters) for ID, cmd_pars in zip(ID_list, cmd_pars_list))

        # store values in a csv file
        df = pd.DataFrame({'ID':ID_list,'ions_bulk':ions_bulk_list})
        df.to_csv(os.path.join(res_dir,f'{L_pero*1e9}_'+str_save+'_ions_bulk_list.csv'),index=False)
    else:
        df = pd.read_csv(os.path.join(res_dir,f'{L_pero*1e9}_'+str_save+'_ions_bulk_list.csv'))
        ions_bulk_list = df['ions_bulk'].values
        ID_list = df['ID'].values

    # plot the results
    colors = plt.cm.viridis(np.linspace(0,1,len(ions_bulk_list)+1))
    # remove the last color
    colors = colors[:-1]
    #invert colors
    colors = colors[::-1]
    fig = plt.figure(figsize=figsize)
    ax1,ax2,ax3,ax4 = plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224)
    # ax = plt.gca()
    minJ,maxJ = 1e20,1e-20
    slopemax = 0
    idx = -1
    minVminMG,maxVmaxMG = 1e3,0
    for i,ions in enumerate(ions_bulk_list):
        ID = ID_list[i]
        JV_file_name = os.path.join('SIMsalabim','SimSS',f'JV_{ID}.dat')
        data_JV = pd.read_csv(JV_file_name,sep=r'\s+')
        label = sci_notation(ions/1e6,0)

        data_JV = data_JV[data_JV['Vext']>0]
        data_JV = data_JV[data_JV['Jext']>0]
        minJ = min(minJ,data_JV['Jext'].min())
        maxJ = max(maxJ,data_JV['Jext'].max())
        if Vscan == -1:
            # reverse the data
            data_JV = data_JV.iloc[::-1]

        V_slopef,J_slopef,slopesf,get_tangentf,idx_maxf,max_slopesf,tang_val_V1f,tang_val_V2f,tang_val_V3f,V1f,J1f,V2f,J2f,Vinf,Jinf = SCLC_get_data_plot(data_JV['Vext'],data_JV['Jext'])
        # print(max_slopesf)
        if max_slopesf > 2:
            nons_list.append(calc_net_charge(V1f,L_pero,eps_r_pero)/1e6)
            ninf_list.append(calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6)
            nend_list.append(calc_net_charge(V2f,L_pero,eps_r_pero)/1e6)
        else:
            nons_list.append(np.nan)
            ninf_list.append(np.nan)
            nend_list.append(np.nan)

        ## Calc Voltages
        Vnet = calc_Vnet_with_ions(ions,traps_pero,L_pero,eps_r_pero)
        Vtfl = calc_Vtfl(traps_pero,L_pero,eps_r_pero)
        Vsat = calc_Vsat(L_pero,5e24,0,eps_r_pero,295)
        Ntmin = calc_nt_min(L_pero,eps_r_pero,295)
        Vmin = calc_Vtfl(Ntmin,L_pero,eps_r_pero)

        # get mobility
        data_JV_mott = copy.deepcopy(data_JV)
        
        if Vsat > Vtfl:
            data_JV_mott['slopesf'] = slopesf
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=1]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']<=Vsat]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=Vtfl]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']<=2.2]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']>=1.8]
            if len(data_JV_mott) <= 5:
                mu.append(0)
                continue
            VminMG = min(data_JV_mott['Vext'])
            VmaxMG = max(data_JV_mott['Vext'])
            Mott_Gurney_fit = fit_MottGurney(data_JV_mott['Vext'],data_JV_mott['Jext'],1e-4,eps_r_pero,0,L_pero,var2fit=['mu'])
            mu.append(Mott_Gurney_fit[0])
            # print('Mott_Gurney_fit:',Mott_Gurney_fit)
        else:
            mu.append(0)

        # remove Vext = 0 from data_JV
        if ions in to_plot:
            data_JV = data_JV[data_JV['Vext']!=0]
            ax1.plot(data_JV['Vext'],data_JV['Jext']/10,label=label,color=colors[i])
            if max_slopesf > 2:
                slopemax = max(slopemax,max(slopesf))
                ax1.plot(V1f,J1f/10,marker='s',color=colors[i])
                ax1.plot(Vinf,Jinf/10,marker='^',color=colors[i])
                ax1.plot(V2f,J2f/10,marker='o',color=colors[i])
                ax1.plot(V_slopef,tang_val_V1f/10,'--',color=colors[i])
                ax1.plot(V_slopef,tang_val_V2f/10,'--',color=colors[i])
                ax1.plot(V_slopef,tang_val_V3f/10,'--',color=colors[i])
            minVminMG = min(minVminMG,VminMG)
            maxVmaxMG = max(maxVmaxMG,VmaxMG)
            ax2.plot(V_slopef,slopesf,label=label,color=colors[i])
            ax2.axvspan(VminMG, VmaxMG, color=colors[i], alpha=0.3,zorder=idx)
            idx -= 1


        if max_slopesf > 2:
            ax3.plot((traps_pero-ions)/1e6,calc_net_charge(V1f,L_pero,eps_r_pero)/1e6,marker='s',color=colors[i])
            ax3.plot((traps_pero-ions)/1e6,calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6,marker='^',color=colors[i])
            ax3.plot((traps_pero-ions)/1e6,calc_net_charge(V2f,L_pero,eps_r_pero)/1e6,marker='o',color=colors[i])
        # add diagonal line
        ax3.plot([1e20/1e6,1e23/1e6],[1e20/1e6,1e23/1e6],'--',color='black')
        # if ions > 0:
        #     # plot diagonal - ions
        #     ax3.plot([(ions+1e21)/1e6,1e23/1e6],[1e21/1e6,(1e23-ions)/1e6],'--',color='black')
        # horizontal line at trap density
        ax3.axhline(y=traps_pero/1e6,linestyle='-',color='black')
        # grey shadow ntmin to ntmax
        ax3.axvspan(1e20/1e6, calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.5,zorder=-3)
        # ax3.axvspan(calc_nt_min(L_pero,eps_r_pero,295)/1e6,3*calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.05,zorder=-2)

    ax1.set_yscale('log')
    # ax1.set_ylim([1e-8,1e7])
    ax1.set_ylabel('Current density [mA m$^{-2}$]')
    ax1.set_xscale('log')
    ax1.set_xlim([1e-2,Vmax])
    ax1.set_ylim([10**np.floor(np.log10(minJ)-1),10**np.floor(np.log10(maxJ)+1)])
    ax1.set_xlabel('Applied voltage [V]')
    ax1.grid(True,which='both',axis='both',color='gray', linestyle='-')

    ax2.set_yscale('linear')
    ax2.set_ylim([0,int(slopemax)+1])
    ticks = ax2.get_yticks()
    ticks = ticks[ticks>3]
    ticks = np.append(ticks,[1,2])

    ticks = np.unique(ticks)
    ticks = np.sort(ticks)
    ax2.set_yticks(ticks)

    ax2.axhline(y=1, color='k', linestyle='-')
    ax2.axhline(y=2, color='k', linestyle='-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Applied voltage [V]')
    ax2.set_ylabel('Slope [-]')
    ax2.set_xlim([1e-2,Vmax])
    ax2.set_ylim([0,int(slopemax)+1])
    ax2.grid(True,which='both',axis='both',color='gray', linestyle='-')
    # add arrow between VminMG and VmaxMG
    ax2.annotate('', xy=(minVminMG, int(slopemax)+1), xytext=(maxVmaxMG, int(slopemax)+1), arrowprops=dict(arrowstyle='<->',color='black',linewidth=3))
    # text in the middle of the arrow on a log scale

    ax2.text(minVminMG + 10**((np.log10(minVminMG)+np.log10(maxVmaxMG))/2), int(slopemax)+1.5, 'MG region', fontsize=20, ha='center')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylabel('Calc. density [cm$^{-3}$]')
    ax3.set_xlabel('Net charge density [cm$^{-3}$]')
    ax3.grid(True,which='both',axis='both',color='gray', linestyle='-',zorder=-1)
    ax3.set_xlim([1e20/1e6,5e22/1e6])
    ax3.set_ylim([1e20/1e6,5e22/1e6])

    # custum legend
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='V$_{ons}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='V$_{inf}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='V$_{end}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], linestyle='-', color='k', label='N$_{T}$', markersize=10)]
    if ions > 0:
        legend_elements.append(Line2D([0], [0], linestyle='--', color='k', label='N$_{net}}$', markersize=10))
    ax3.legend(handles=legend_elements, loc='upper left')

    ticks = []
    for i in ions_bulk_list:
        ticks.append(sci_notation(i/1e6, sig_fig=1))
        # ticks.append(sci_notation(i*1e4, sig_fig=-1))
    mu = np.asarray(mu)*1e4
    ax4.bar(np.arange(len(mu)),mu,color=colors)
    # plt.bar(loop,mu*1e4,color=colors[:ind],width=loop[1]-loop[0])
    ax4.axhline(1,linestyle='-',color='k')
    ax4.set_xticks(np.arange(len(mu)), ticks,fontsize=20,rotation=45)

    ax4.set_ylabel('Mobility [cm$^{2}$V$^{-1}$s$^{-1}$]')
    ax4.set_xlabel('Ion density [cm$^{-3}$]')
    ax4.set_yscale('log')
    ax4.set_ylim([1e-4 , 10])
    ax4.grid(b=True,which='both',axis='y',zorder=-1)

    # add super title to axes
    ax1.set_title('a)',position=(-0.1,0.97))
    ax2.set_title('b)',position=(-0.25,0.97))
    ax3.set_title('c)',position=(-0.1,0.95))
    ax4.set_title('d)',position=(-0.25,0.95))
    plt.tight_layout()


    plt.savefig(os.path.join(res_dir,f'{L_pero*1e9}_'+str_save+'_ions_bulk_list.png'),dpi=300)

def run_all_TLs(eps_TL_list,Nc_TL_list,mob_TL_list,L_pero, eps_r_pero, ions, traps_pero, Vmax, Vscan, NP, tolPois, tolDens, couplePC, minAcc, maxAcc, grad, simss_device_parameters, str_save, with_TL, rerun=False, ions_in_TLs =False,figsize = (21,15),to_plot = [3,5,10,15]):

    res_dir = os.path.join(os.getcwd(),'results')
    
    nons_list,ninf_list,nend_list,mu  = [],[],[],[]
    if rerun:
        cmd_pars_list, ID_list = [], []
        # prepare cmd_pars_list
        if with_TL:
            layer = 'l2'
        else:
            layer = 'l1'
        for i in range(len(eps_TL_list)):
            cmd_pars = [{'par':'l1.N_c','val':str(Nc_TL_list[i])},{'par':'l3.N_c','val':str(Nc_TL_list[i])},{'par':'l1.eps_r','val':str(eps_TL_list[i])},{'par':'l3.eps_r','val':str(eps_TL_list[i])},{'par': 'l2.L', 'val': str(L_pero)},{'par':'l2.N_t_bulk','val':str(traps_pero)},{'par':'l2.N_anion','val':str(ions)},{'par':'l2.N_cation','val':str(ions)},{'par':'Vmax','val':str(Vmax)},{'par':'NP','val':str(NP)},{'par':'Vscan','val':str(Vscan)},{'par':'timeout','val':timeout},{'par':'tolPois','val':str(tolPois)},{'par':'tolDens','val':str(tolDens)},{'par':'couplePC','val':str(couplePC)},{'par':'minAcc','val':str(minAcc)},{'par':'maxAcc','val':str(maxAcc)},{'par':'grad','val':str(grad)},{'par':'l1.mu_n','val':str(mob_TL_list[i])},{'par':'l1.mu_p','val':str(mob_TL_list[i])},{'par':'l3.mu_n','val':str(mob_TL_list[i])},{'par':'l3.mu_p','val':str(mob_TL_list[i])}]
            if ions_in_TLs:
                cmd_pars.append({'par':'l1.ionsMayEnter','val':str(1)})
                cmd_pars.append({'par':'l3.ionsMayEnter','val':str(1)})
            cmd_pars_list.append(cmd_pars)
            ID_list.append(str(uuid.uuid4()))

        # run the simulation with joblib
        Parallel(n_jobs=min(len(cmd_pars_list),15))(delayed(run)(ID, cmd_pars,simss_device_parameters) for ID, cmd_pars in zip(ID_list, cmd_pars_list))

        # store values in a csv file
        df = pd.DataFrame({'ID':ID_list,'eps_TL':eps_TL_list,'Nc_TL':Nc_TL_list})
        df.to_csv(os.path.join(res_dir,f'{L_pero*1e9}_TL_type_list.csv'),index=False)
    else:
        df = pd.read_csv(os.path.join(res_dir,f'{L_pero*1e9}_TL_type_list.csv'))
        ID_list = df['ID'].values

    # plot the results
    colors = plt.cm.viridis(np.linspace(0,1,len(eps_TL_list)+1))
    # remove the last color
    colors = colors[:-1]
    #invert colors
    colors = colors[::-1]
    fig = plt.figure(figsize=figsize)
    ax1,ax2,ax3,ax4 = plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224)
    # ax = plt.gca()
    minJ,maxJ = 1e20,1e-20
    VminMG,VmaxMG = 1e3,0
    slopemax = 0
    idx = -1
    minVminMG,maxVmaxMG = 1e3,0
    for i,ep in enumerate(eps_TL_list):
        ID = ID_list[i]
        JV_file_name = os.path.join('SIMsalabim','SimSS',f'JV_{ID}.dat')
        data_JV = pd.read_csv(JV_file_name,sep=r'\s+')
        label = ''
        
        data_JV = data_JV[data_JV['Vext']>0]
        data_JV = data_JV[data_JV['Jext']>0]
        minJ = min(minJ,data_JV['Jext'].min())
        maxJ = max(maxJ,data_JV['Jext'].max())
        if Vscan == -1:
            # reverse the data
            data_JV = data_JV.iloc[::-1]

        V_slopef,J_slopef,slopesf,get_tangentf,idx_maxf,max_slopesf,tang_val_V1f,tang_val_V2f,tang_val_V3f,V1f,J1f,V2f,J2f,Vinf,Jinf = SCLC_get_data_plot(data_JV['Vext'],data_JV['Jext'])

        # print(max_slopesf)
        if max_slopesf > 2:
            nons_list.append(calc_net_charge(V1f,L_pero,eps_r_pero)/1e6)
            ninf_list.append(calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6)
            nend_list.append(calc_net_charge(V2f,L_pero,eps_r_pero)/1e6)
        else:
            nons_list.append(np.nan)
            ninf_list.append(np.nan)
            nend_list.append(np.nan)

        ## Calc Voltages
        Vnet = calc_Vnet_with_ions(ions,traps_pero,L_pero,eps_r_pero)
        Vtfl = calc_Vtfl(traps_pero,L_pero,eps_r_pero)
        Vsat = calc_Vsat(L_pero,5e24,0,eps_r_pero,295)
        Ntmin = calc_nt_min(L_pero,eps_r_pero,295)
        Vmin = calc_Vtfl(Ntmin,L_pero,eps_r_pero)

        # get mobility
        data_JV_mott = copy.deepcopy(data_JV)
        
        if Vsat > Vtfl:
            data_JV_mott['slopesf'] = slopesf
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=1]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']<=Vsat]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=Vtfl]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']<=2.2]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']>=1.8]
            if len(data_JV_mott) <= 5:
                mu.append(0)
            else:
                VminMG = min(data_JV_mott['Vext'])
                VmaxMG = max(data_JV_mott['Vext'])
                Mott_Gurney_fit = fit_MottGurney(data_JV_mott['Vext'],data_JV_mott['Jext'],1e-4,eps_r_pero,0,L_pero,var2fit=['mu'])
                mu.append(Mott_Gurney_fit[0])
            # print('Mott_Gurney_fit:',Mott_Gurney_fit)
        else:
            mu.append(0)

        # remove Vext = 0 from data_JV
        # if eps_TL_list[i] in to_plot:
        data_JV = data_JV[data_JV['Vext']!=0]

        ax1.plot(data_JV['Vext'],data_JV['Jext']/10,label=label,color=colors[i])
        if max_slopesf > 2:
            slopemax = max(slopemax,max(slopesf))
            ax1.plot(V1f,J1f/10,marker='s',color=colors[i])
            ax1.plot(Vinf,Jinf/10,marker='^',color=colors[i])
            ax1.plot(V2f,J2f/10,marker='o',color=colors[i])
            ax1.plot(V_slopef,tang_val_V1f/10,'--',color=colors[i])
            ax1.plot(V_slopef,tang_val_V2f/10,'--',color=colors[i])
            ax1.plot(V_slopef,tang_val_V3f/10,'--',color=colors[i])
        minVminMG = min(minVminMG,VminMG)
        maxVmaxMG = max(maxVmaxMG,VmaxMG)
        ax2.plot(V_slopef,slopesf,label=label,color=colors[i])
        if len(data_JV_mott) >= 5:
            ax2.axvspan(VminMG, VmaxMG, color=colors[i], alpha=0.3,zorder=idx)
        idx -= 1


        if max_slopesf > 2:
            ax3.plot(eps_TL_list[i],calc_net_charge(V1f,L_pero,eps_r_pero)/1e6,marker='s',color=colors[i])
            ax3.plot(eps_TL_list[i],calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6,marker='^',color=colors[i])
            ax3.plot(eps_TL_list[i],calc_net_charge(V2f,L_pero,eps_r_pero)/1e6,marker='o',color=colors[i])
        # add diagonal line
        # ax3.plot([1e20/1e6,1e23/1e6],[1e20/1e6,1e23/1e6],'--',color='black')
        # if ions > 0:
        #     # plot diagonal - ions
        # ax3.plot([(ions+1e21)/1e6,1e23/1e6],[1e21/1e6,(1e23-ions)/1e6],'--',color='black')
        # horizontal line at trap density
        ax3.axhline(y=traps_pero/1e6,linestyle='-',color='black')
        ax3.axhline(y=(traps_pero-ions)/1e6,linestyle='--',color='black')
        # grey shadow ntmin to ntmax
        ax3.axvspan(1e20/1e6, calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.5,zorder=-3)
        # ax3.axvspan(calc_nt_min(L_pero,eps_r_pero,295)/1e6,3*calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.05,zorder=-2)

    ax1.set_yscale('log')
    # ax1.set_ylim([1e-8,1e7])
    ax1.set_ylabel('Current density [mA m$^{-2}$]')
    ax1.set_xscale('log')
    ax1.set_xlim([1e-2,Vmax])
    ax1.set_ylim([10**np.floor(np.log10(minJ)-1),10**np.floor(np.log10(maxJ)+1)])
    ax1.set_xlabel('Applied voltage [V]')
    ax1.grid(True,which='both',axis='both',color='gray', linestyle='-')

    ax2.set_yscale('linear')
    ax2.set_ylim([0,int(slopemax)+1])
    ticks = ax2.get_yticks()
    ticks = ticks[ticks>3]
    ticks = np.append(ticks,[1,2])

    ticks = np.unique(ticks)
    ticks = np.sort(ticks)
    ax2.set_yticks(ticks)

    ax2.axhline(y=1, color='k', linestyle='-')
    ax2.axhline(y=2, color='k', linestyle='-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Applied voltage [V]')
    ax2.set_ylabel('Slope [-]')
    ax2.set_xlim([1e-2,Vmax])
    ax2.set_ylim([0,int(slopemax)+1])
    ax2.grid(True,which='both',axis='both',color='gray', linestyle='-')
    # add arrow between VminMG and VmaxMG
    ax2.annotate('', xy=(minVminMG, int(slopemax)+1), xytext=(maxVmaxMG, int(slopemax)+1), arrowprops=dict(arrowstyle='<->',color='black',linewidth=3))
    # text in the middle of the arrow on a log scale
    ax2.text(minVminMG + 10**((np.log10(minVminMG)+np.log10(maxVmaxMG))/2), int(slopemax)+1.5, 'MG region', fontsize=20, ha='center')

    ticks = []
    for i in range(len(eps_TL_list)):
        ticks.append('$\epsilon_r^{TL}$' + f' {eps_TL_list[i]:.0f}'+'\n$N_c^{TL}$ '+sci_notation(Nc_TL_list[i],0))
    tickpos = np.asarray(eps_TL_list)
    ax3.set_xticks(tickpos)
    # replace ticks with custom ticks
    ax3.set_xticklabels(ticks,fontsize=20,rotation=45, ha='center')
    # ax3.set_xscale('linear')
    ax3.set_yscale('log')
    ax3.set_ylabel('Calc. density [cm$^{-3}$]')
    # ax3.set_xlabel('Net charge density [cm$^{-3}$]')
    ax3.grid(True,which='both',axis='both',color='gray', linestyle='-',zorder=-1)
    ax3.set_xlim([2,17])
    ax3.set_ylim([1e20/1e6,5e22/1e6])

    # custum legend
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='V$_{ons}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='V$_{inf}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='V$_{end}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], linestyle='-', color='k', label='N$_{T}$', markersize=10)]
    if ions > 0:
        legend_elements.append(Line2D([0], [0], linestyle='--', color='k', label='N$_{net}}$', markersize=10))
    ax3.legend(handles=legend_elements, loc='lower left',ncol=2)

    # ticks = []
    # for i in range(len(eps_TL_list)):
    #     ticks.append(sci_notation(i/1e6, sig_fig=1))
        # ticks.append(sci_notation(i*1e4, sig_fig=-1))
    ticks = []
    for i in range(len(eps_TL_list)):
        ticks.append('$\epsilon_r^{TL}$' + f' {eps_TL_list[i]:.0f}'+'\n$N_c^{TL}$ '+sci_notation(Nc_TL_list[i],0))
    mu = np.asarray(mu)*1e4
    ax4.bar(np.arange(len(mu)),mu,color=colors)
    # plt.bar(loop,mu*1e4,color=colors[:ind],width=loop[1]-loop[0])
    ax4.axhline(1,linestyle='-',color='k')
    ax4.set_xticks(np.arange(len(mu)), ticks,fontsize=20)#,rotation=45)

    ax4.set_ylabel('Mobility [cm$^{2}$V$^{-1}$s$^{-1}$]')
    # ax4.set_xlabel('Ion density [cm$^{-3}$]')
    ax4.set_yscale('log')
    ax4.set_ylim([1e-4 , 10])
    ax4.grid(b=True,which='both',axis='y',zorder=-1)

    # add super title to axes
    ax1.set_title('a)',position=(-0.1,0.97))
    ax2.set_title('b)',position=(-0.25,0.97))
    ax3.set_title('c)',position=(-0.1,0.95))
    ax4.set_title('d)',position=(-0.25,0.95))
    plt.tight_layout()


    plt.savefig(os.path.join(res_dir,f'{L_pero*1e9}_TL_type_list.png'),dpi=300)
    
def run_all_nrjs(nrj_TL_list,eps_TL,Nc_TL,L_pero, eps_r_pero, ions, traps_pero, Vmax, Vscan, NP, tolPois, tolDens, couplePC, minAcc, maxAcc, grad, simss_device_parameters, str_save, with_TL, rerun=False, ions_in_TLs =False,figsize = (21,15),to_plot = [3.9,4,4.1,4.2]):

    res_dir = os.path.join(os.getcwd(),'results')
    Ec_pero = 3.9
    nons_list,ninf_list,nend_list,mu  = [],[],[],[]
    if rerun:
        cmd_pars_list, ID_list = [], []
        # prepare cmd_pars_list
        if with_TL:
            layer = 'l2'
        else:
            layer = 'l1'
        for i in range(len(nrj_TL_list)):
            cmd_pars = [{'par':'l3.E_c','val':str(nrj_TL_list[i])},{'par':'W_R','val':str(nrj_TL_list[i])}, {'par':'useExpData','val':str(1)}, {'par':'expJV','val':'exp_JV.csv'},
                {'par':'l1.N_c','val':str(Nc_TL)},{'par':'l3.N_c','val':str(Nc_TL)},{'par':'l1.eps_r','val':str(eps_TL)},{'par':'l3.eps_r','val':str(eps_TL)},{'par': 'l2.L', 'val': str(L_pero)},{'par':'l2.N_t_bulk','val':str(traps_pero)},{'par':'l2.N_anion','val':str(ions)},{'par':'l2.N_cation','val':str(ions)},{'par':'Vmax','val':str(Vmax)},{'par':'NP','val':str(NP)},{'par':'Vscan','val':str(Vscan)},{'par':'timeout','val':timeout},{'par':'tolPois','val':str(tolPois)},{'par':'tolDens','val':str(tolDens)},{'par':'couplePC','val':str(couplePC)},{'par':'minAcc','val':str(minAcc)},{'par':'maxAcc','val':str(maxAcc)},{'par':'grad','val':str(grad)}]
            if ions_in_TLs:
                cmd_pars.append({'par':'l1.ionsMayEnter','val':str(1)})
                cmd_pars.append({'par':'l3.ionsMayEnter','val':str(1)})
            cmd_pars_list.append(cmd_pars)
            ID_list.append(str(uuid.uuid4()))

        # run the simulation with joblib
        Parallel(n_jobs=min(len(cmd_pars_list),15))(delayed(run)(ID, cmd_pars,simss_device_parameters) for ID, cmd_pars in zip(ID_list, cmd_pars_list))

        # store values in a csv file
        df = pd.DataFrame({'ID':ID_list,'nrj_TL':nrj_TL_list})
        df.to_csv(os.path.join(res_dir,f'{L_pero*1e9}'+str_save+'_nrj_TL_list.csv'),index=False)
    else:
        df = pd.read_csv(os.path.join(res_dir,f'{L_pero*1e9}'+str_save+'_nrj_TL_list.csv'))
        ID_list = df['ID'].values

    # plot the results
    colors = plt.cm.viridis(np.linspace(0,1,len(nrj_TL_list)+1))
    # remove the last color
    colors = colors[:-1]
    #invert colors
    colors = colors[::-1]
    fig = plt.figure(figsize=figsize)
    ax1,ax2,ax3,ax4 = plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224)
    # ax = plt.gca()
    minJ,maxJ = 1e20,1e-20
    VminMG,VmaxMG = 1e3,0
    slopemax = 0
    idx = -2*len(nrj_TL_list)
    minVminMG,maxVmaxMG = 1e3,0
    for i,ep in enumerate(nrj_TL_list):
        ID = ID_list[i]
        JV_file_name = os.path.join('SIMsalabim','SimSS',f'JV_{ID}.dat')
        data_JV_main = pd.read_csv(JV_file_name,sep=r'\s+')
        label = ''
        
        for sign_V in [-1,1]:
            data_JV = copy.deepcopy(data_JV_main)
            data_JV['Vext'] = data_JV['Vext']*sign_V
            data_JV['Jext'] = data_JV['Jext']*sign_V
            data_JV = data_JV[data_JV['Vext']>0]
            data_JV = data_JV[data_JV['Jext']>0]
            minJ = min(minJ,data_JV['Jext'].min())
            maxJ = max(maxJ,data_JV['Jext'].max())
            if sign_V == -1:
                # reverse the data
                data_JV = data_JV.iloc[::-1]

            V_slopef,J_slopef,slopesf,get_tangentf,idx_maxf,max_slopesf,tang_val_V1f,tang_val_V2f,tang_val_V3f,V1f,J1f,V2f,J2f,Vinf,Jinf = SCLC_get_data_plot(data_JV['Vext'],data_JV['Jext'])

            # print(max_slopesf)
            if max_slopesf > 2:
                nons_list.append(calc_net_charge(V1f,L_pero,eps_r_pero)/1e6)
                ninf_list.append(calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6)
                nend_list.append(calc_net_charge(V2f,L_pero,eps_r_pero)/1e6)
            else:
                nons_list.append(np.nan)
                ninf_list.append(np.nan)
                nend_list.append(np.nan)

            ## Calc Voltages
            Vnet = calc_Vnet_with_ions(ions,traps_pero,L_pero,eps_r_pero)
            Vtfl = calc_Vtfl(traps_pero,L_pero,eps_r_pero)
            Vsat = calc_Vsat(L_pero,5e24,0,eps_r_pero,295)
            Ntmin = calc_nt_min(L_pero,eps_r_pero,295)
            Vmin = calc_Vtfl(Ntmin,L_pero,eps_r_pero)

            # get mobility
            data_JV_mott = copy.deepcopy(data_JV)
            
            if Vsat > Vtfl:
                data_JV_mott['slopesf'] = slopesf
                data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=1]
                data_JV_mott = data_JV_mott[data_JV_mott['Vext']<=Vsat]
                data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=Vtfl]
                data_JV_mott = data_JV_mott[data_JV_mott['slopesf']<=2.2]
                data_JV_mott = data_JV_mott[data_JV_mott['slopesf']>=1.8]
                if len(data_JV_mott) <= 5:
                    mu.append(0)
                else:
                    VminMG = min(data_JV_mott['Vext'])
                    VmaxMG = max(data_JV_mott['Vext'])
                    Mott_Gurney_fit = fit_MottGurney(data_JV_mott['Vext'],data_JV_mott['Jext'],1e-4,eps_r_pero,0,L_pero,var2fit=['mu'])
                    mu.append(Mott_Gurney_fit[0])
                # print('Mott_Gurney_fit:',Mott_Gurney_fit)
            else:
                mu.append(0)

            # remove Vext = 0 from data_JV
            # if nrj_TL_list[i] in to_plot:
            data_JV = data_JV[data_JV['Vext']!=0]
            if sign_V == -1:
                linestyle = '--'
            else:
                linestyle = '-'
            ax1.plot(data_JV['Vext'],data_JV['Jext']/10,label=label,color=colors[i],linestyle=linestyle)
            # marker fill color if sign_V == 1: else no color
            if sign_V == 1:
                markerfacecolor = colors[i]
            else:
                markerfacecolor = 'None'


            if max_slopesf > 2:
                slopemax = max(slopemax,max(slopesf))
                # ax1.plot(V1f,J1f/10,marker='s',color=colors[i],markerfacecolor=markerfacecolor)
                # ax1.plot(Vinf,Jinf/10,marker='^',color=colors[i],markerfacecolor=markerfacecolor)
                # ax1.plot(V2f,J2f/10,marker='o',color=colors[i],markerfacecolor=markerfacecolor)
                # ax1.plot(V_slopef,tang_val_V1f/10,'--',color=colors[i])
                # ax1.plot(V_slopef,tang_val_V2f/10,'--',color=colors[i])
                # ax1.plot(V_slopef,tang_val_V3f/10,'--',color=colors[i])
            minVminMG = min(minVminMG,VminMG)
            maxVmaxMG = max(maxVmaxMG,VmaxMG)
            ax2.plot(V_slopef,slopesf,label=label,color=colors[i],linestyle=linestyle)
            # if len(data_JV_mott) >= 5:
            #     ax2.axvspan(VminMG, VmaxMG, color=colors[i], alpha=0.5,zorder=idx)
            idx += 1


            if max_slopesf > 2:
                ax3.plot(Ec_pero-nrj_TL_list[i],calc_net_charge(V1f,L_pero,eps_r_pero)/1e6,marker='s',color=colors[i],markerfacecolor=markerfacecolor)
                ax3.plot(Ec_pero-nrj_TL_list[i],calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6,marker='^',color=colors[i],markerfacecolor=markerfacecolor)
                ax3.plot(Ec_pero-nrj_TL_list[i],calc_net_charge(V2f,L_pero,eps_r_pero)/1e6,marker='o',color=colors[i],markerfacecolor=markerfacecolor)
            # add diagonal line
            # ax3.plot([1e20/1e6,1e23/1e6],[1e20/1e6,1e23/1e6],'--',color='black')
            # if ions > 0:
            #     # plot diagonal - ions
            # ax3.plot([(ions+1e21)/1e6,1e23/1e6],[1e21/1e6,(1e23-ions)/1e6],'--',color='black')
            # horizontal line at trap density
            ax3.axhline(y=traps_pero/1e6,linestyle='-',color='black')
            ax3.axhline(y=(traps_pero-ions)/1e6,linestyle='--',color='black')
            # grey shadow ntmin to ntmax
            ax3.axvspan(1e20/1e6, calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.5,zorder=-3)
            # ax3.axvspan(calc_nt_min(L_pero,eps_r_pero,295)/1e6,3*calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.05,zorder=-2)

    ax1.set_yscale('log')
    # ax1.set_ylim([1e-8,1e7])
    ax1.set_ylabel('Current density [mA m$^{-2}$]')
    ax1.set_xscale('log')
    ax1.set_xlim([1e-2,Vmax])
    ax1.set_ylim([10**np.floor(np.log10(minJ)-1),10**np.floor(np.log10(maxJ)+1)])
    ax1.set_xlabel('Applied voltage [V]')
    ax1.grid(True,which='both',axis='both',color='gray', linestyle='-')
    legend_elements = [Line2D([0], [0], linestyle='-', color='k', label='V>0', markersize=10),
                        Line2D([0], [0], linestyle='--', color='k', label='V<0', markersize=10),
                        ]
    ax1.legend(handles=legend_elements, loc='upper left',ncol = 1)

    ax2.set_yscale('linear')
    ax2.set_ylim([0,int(slopemax)+1])
    ticks = ax2.get_yticks()
    ticks = ticks[ticks>3]
    ticks = np.append(ticks,[1,2])

    ticks = np.unique(ticks)
    ticks = np.sort(ticks)
    ax2.set_yticks(ticks)

    ax2.axhline(y=1, color='k', linestyle='-')
    ax2.axhline(y=2, color='k', linestyle='-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Applied voltage [V]')
    ax2.set_ylabel('Slope [-]')
    ax2.set_xlim([1e-2,Vmax])
    ax2.set_ylim([0,int(slopemax)+1])
    ax2.grid(True,which='both',axis='both',color='gray', linestyle='-')
    # add arrow between VminMG and VmaxMG
    ax2.annotate('', xy=(minVminMG, int(slopemax)+1), xytext=(maxVmaxMG, int(slopemax)+1), arrowprops=dict(arrowstyle='<->',color='black',linewidth=3))
    # text in the middle of the arrow on a log scale
    ax2.text(minVminMG + 10**((np.log10(minVminMG)+np.log10(maxVmaxMG))/2), int(slopemax)+1.5, 'MG region', fontsize=20, ha='center')
    # ticks = []
    # for i in range(len(nrj_TL_list)):
    #     ticks.append('$\epsilon_r^{TL}$' + f' {nrj_TL_list[i]:.1f}'+'\n$N_c^{TL}$ '+sci_notation(Nc_TL_list[i],0))
    # tickpos = np.asarray(nrj_TL_list)
    # ax3.set_xticks(tickpos)
    # replace ticks with custom ticks
    # ax3.set_xticklabels(ticks,fontsize=20,rotation=45, ha='center')
    ax3.set_xscale('linear')
    ax3.set_yscale('log')
    ax3.set_ylabel('Calc. density [cm$^{-3}$]')
    ax3.set_xlabel('TL energy offset [eV]')
    ax3.grid(True,which='both',axis='both',color='gray', linestyle='-',zorder=-1)
    ax3.set_xlim([min(Ec_pero-np.asarray(nrj_TL_list))-0.05,max(Ec_pero-np.asarray(nrj_TL_list))+0.05])
    # ax3.set_ylim([1e20/1e6,5e22/1e6])

    # custum legend
    legend_elements = [
                        Line2D([0], [0], marker='s', color='w', label='V$_{ons}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='V$_{inf}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='V$_{end}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='V>0', markerfacecolor='k', markersize=10),
                        
                        Line2D([0], [0], marker='o', color='w', label='V<0', markerfacecolor='None', markersize=10,markeredgecolor='k'),
                        Line2D([0], [0], linestyle='-', color='k', label='N$_{T}$', markersize=10),
                        Line2D([0], [0], linestyle='--', color='k', label='N$_{net}}$', markersize=10),
                        ]
    # if ions > 0:
    #     legend_elements.append(Line2D([0], [0], linestyle='--', color='k', label='N$_{net}}$', markersize=10))
    ax3.legend(handles=legend_elements, loc='best',ncol = 3)

    ticks = []
    for i in range(len(nrj_TL_list)):
        ticks.append(f'{Ec_pero-nrj_TL_list[i]:.1f}')
        

    mu = np.asarray(mu)*1e4
    # split mu in two parts
    mu_rev,mu_for = [],[]
    for i in range(len(mu)):
        if i%2 == 0:
            mu_rev.append(mu[i])
        else:
            mu_for.append(mu[i])
    ax4.bar(np.arange(len(mu_for)),mu_for,color=colors)
    # bar plot not color filled but with black edges
    ax4.bar(np.arange(len(mu_rev)),mu_rev,color='None',edgecolor='black',linewidth=2,linestyle='--')
    # ax4.scatter(np.arange(len(mu_rev)),mu_rev,marker='*',color=colors)
    # ax4.bar(np.arange(len(mu)),mu,color=colors)
    # plt.bar(loop,mu*1e4,color=colors[:ind],width=loop[1]-loop[0])
    ax4.axhline(1,linestyle='-',color='k')
    ax4.set_xticks(np.arange(len(mu_for)), ticks,fontsize=20)#,rotation=45)

    ax4.set_ylabel('Mobility [cm$^{2}$V$^{-1}$s$^{-1}$]')
    ax4.set_xlabel('TL energy offset [eV]')
    ax4.set_yscale('log')
    ax4.set_ylim([1e-4 , 10])
    ax4.grid(b=True,which='both',axis='y',zorder=-1)

    # add super title to axes
    ax1.set_title('a)',position=(-0.15,0.98))
    ax2.set_title('b)',position=(-0.25,0.98))
    ax3.set_title('c)',position=(-0.15,0.95))
    ax4.set_title('d)',position=(-0.25,0.95))
    plt.tight_layout()


    plt.savefig(os.path.join(res_dir,f'{L_pero*1e9}'+str_save+'_nrj_TL_list.png'),dpi=300)
    
def run_all_mobs(mob_TL_list,L_pero, eps_r_pero, ions, traps_pero, Vmax, Vscan, NP, tolPois, tolDens, couplePC, minAcc, maxAcc, grad, simss_device_parameters, str_save, with_TL, rerun=False, ions_in_TLs =False,figsize = (21,15),to_plot = [1e-8,1e-6,1e-4,1e-3]):
    
    res_dir = os.path.join(os.getcwd(),'results')

    nons_list,ninf_list,nend_list,mu  = [],[],[],[]
    if rerun:
        cmd_pars_list, ID_list = [], []
        # prepare cmd_pars_list
        if with_TL:
            layer = 'l2'
        else:
            layer = 'l1'
        for i in range(len(mob_TL_list)):
            cmd_pars = [{'par':'l1.mu_n','val':str(mob_TL_list[i])},{'par':'l3.mu_n','val':str(mob_TL_list[i])},{'par': 'l2.L', 'val': str(L_pero)},{'par':'l2.N_t_bulk','val':str(traps_pero)},{'par':'l2.N_anion','val':str(ions)},{'par':'l2.N_cation','val':str(ions)},{'par':'Vmax','val':str(Vmax)},{'par':'NP','val':str(NP)},{'par':'Vscan','val':str(Vscan)},{'par':'timeout','val':timeout},{'par':'tolPois','val':str(tolPois)},{'par':'tolDens','val':str(tolDens)},{'par':'couplePC','val':str(couplePC)},{'par':'minAcc','val':str(minAcc)},{'par':'maxAcc','val':str(maxAcc)},{'par':'grad','val':str(grad)}]
            if ions_in_TLs:
                cmd_pars.append({'par':'l1.ionsMayEnter','val':str(1)})
                cmd_pars.append({'par':'l3.ionsMayEnter','val':str(1)})
            cmd_pars_list.append(cmd_pars)
            ID_list.append(str(uuid.uuid4()))

        # run the simulation with joblib
        Parallel(n_jobs=min(len(cmd_pars_list),15))(delayed(run)(ID, cmd_pars,simss_device_parameters) for ID, cmd_pars in zip(ID_list, cmd_pars_list))

        # store values in a csv file
        df = pd.DataFrame({'ID':ID_list,'mob_TL':mob_TL_list})
        df.to_csv(os.path.join(res_dir,f'{L_pero*1e9}_mob_TL_list.csv'),index=False)
    else:
        df = pd.read_csv(os.path.join(res_dir,f'{L_pero*1e9}_mob_TL_list.csv'))
        ID_list = df['ID'].values

    # plot the results
    colors = plt.cm.viridis(np.linspace(0,1,len(mob_TL_list)+1))
    # remove the last color
    colors = colors[:-1]
    #invert colors
    colors = colors[::-1]
    fig = plt.figure(figsize=figsize)
    ax1,ax2,ax3,ax4 = plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224)
    # ax = plt.gca()
    minJ,maxJ = 1e20,1e-20
    VminMG,VmaxMG = 1e3,0
    slopemax = 0
    idx = -len(mob_TL_list)
    minVminMG,maxVmaxMG = 1e3,0
    for i,ep in enumerate(mob_TL_list):
        ID = ID_list[i]
        JV_file_name = os.path.join('SIMsalabim','SimSS',f'JV_{ID}.dat')
        data_JV = pd.read_csv(JV_file_name,sep=r'\s+')
        label = ''
        
        data_JV = data_JV[data_JV['Vext']>0]
        data_JV = data_JV[data_JV['Jext']>0]
        minJ = min(minJ,data_JV['Jext'].min())
        maxJ = max(maxJ,data_JV['Jext'].max())
        if Vscan == -1:
            # reverse the data
            data_JV = data_JV.iloc[::-1]

        V_slopef,J_slopef,slopesf,get_tangentf,idx_maxf,max_slopesf,tang_val_V1f,tang_val_V2f,tang_val_V3f,V1f,J1f,V2f,J2f,Vinf,Jinf = SCLC_get_data_plot(data_JV['Vext'],data_JV['Jext'])

        # print(max_slopesf)
        if max_slopesf > 2:
            nons_list.append(calc_net_charge(V1f,L_pero,eps_r_pero)/1e6)
            ninf_list.append(calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6)
            nend_list.append(calc_net_charge(V2f,L_pero,eps_r_pero)/1e6)
        else:
            nons_list.append(np.nan)
            ninf_list.append(np.nan)
            nend_list.append(np.nan)

        ## Calc Voltages
        Vnet = calc_Vnet_with_ions(ions,traps_pero,L_pero,eps_r_pero)
        Vtfl = calc_Vtfl(traps_pero,L_pero,eps_r_pero)
        Vsat = calc_Vsat(L_pero,5e24,0,eps_r_pero,295)
        Ntmin = calc_nt_min(L_pero,eps_r_pero,295)
        Vmin = calc_Vtfl(Ntmin,L_pero,eps_r_pero)

        # get mobility
        data_JV_mott = copy.deepcopy(data_JV)
        
        if Vsat > Vtfl:
            data_JV_mott['slopesf'] = slopesf
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=1]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']<=Vsat]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=Vtfl]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']<=2.2]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']>=1.8]
            if len(data_JV_mott) <= 5:
                mu.append(0)
            else:
                VminMG = min(data_JV_mott['Vext'])
                VmaxMG = max(data_JV_mott['Vext'])
                Mott_Gurney_fit = fit_MottGurney(data_JV_mott['Vext'],data_JV_mott['Jext'],1e-4,eps_r_pero,0,L_pero,var2fit=['mu'])
                mu.append(Mott_Gurney_fit[0])
            # print('Mott_Gurney_fit:',Mott_Gurney_fit)
        else:
            mu.append(0)

        # remove Vext = 0 from data_JV
        # if mob_TL_list[i] in to_plot:
        data_JV = data_JV[data_JV['Vext']!=0]

        ax1.plot(data_JV['Vext'],data_JV['Jext']/10,label=label,color=colors[i])
        if max_slopesf > 2:
            slopemax = max(slopemax,max(slopesf))
            ax1.plot(V1f,J1f/10,marker='s',color=colors[i])
            ax1.plot(Vinf,Jinf/10,marker='^',color=colors[i])
            ax1.plot(V2f,J2f/10,marker='o',color=colors[i])
            ax1.plot(V_slopef,tang_val_V1f/10,'--',color=colors[i])
            ax1.plot(V_slopef,tang_val_V2f/10,'--',color=colors[i])
            ax1.plot(V_slopef,tang_val_V3f/10,'--',color=colors[i])
        minVminMG = min(minVminMG,VminMG)
        maxVmaxMG = max(maxVmaxMG,VmaxMG)
        ax2.plot(V_slopef,slopesf,label=label,color=colors[i])
        if len(data_JV_mott) >= 5:
            ax2.axvspan(VminMG, VmaxMG, color=colors[i], alpha=0.3,zorder=idx)
        idx += 1


        if max_slopesf > 2:
            ax3.plot(mob_TL_list[i],calc_net_charge(V1f,L_pero,eps_r_pero)/1e6,marker='s',color=colors[i])
            ax3.plot(mob_TL_list[i],calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6,marker='^',color=colors[i])
            ax3.plot(mob_TL_list[i],calc_net_charge(V2f,L_pero,eps_r_pero)/1e6,marker='o',color=colors[i])
        # add diagonal line
        # ax3.plot([1e20/1e6,1e23/1e6],[1e20/1e6,1e23/1e6],'--',color='black')
        # if ions > 0:
        #     # plot diagonal - ions
        # ax3.plot([(ions+1e21)/1e6,1e23/1e6],[1e21/1e6,(1e23-ions)/1e6],'--',color='black')
        # horizontal line at trap density
        ax3.axhline(y=traps_pero/1e6,linestyle='-',color='black')
        ax3.axhline(y=(traps_pero-ions)/1e6,linestyle='--',color='black')
        # grey shadow ntmin to ntmax
        # ax3.axvspan(1e20/1e6, calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.5,zorder=-3)
        # ax3.axvspan(calc_nt_min(L_pero,eps_r_pero,295)/1e6,3*calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.05,zorder=-2)

    ax1.set_yscale('log')
    # ax1.set_ylim([1e-8,1e7])
    ax1.set_ylabel('Current density [mA m$^{-2}$]')
    ax1.set_xscale('log')
    ax1.set_xlim([1e-2,Vmax])
    ax1.set_ylim([10**np.floor(np.log10(minJ)-1),10**np.floor(np.log10(maxJ)+1)])
    ax1.set_xlabel('Applied voltage [V]')
    ax1.grid(True,which='both',axis='both',color='gray', linestyle='-')

    ax2.set_yscale('linear')
    ax2.set_ylim([0,int(slopemax)+1])
    ticks = ax2.get_yticks()
    ticks = ticks[ticks>3]
    ticks = np.append(ticks,[1,2])

    ticks = np.unique(ticks)
    ticks = np.sort(ticks)
    ax2.set_yticks(ticks)

    ax2.axhline(y=1, color='k', linestyle='-')
    ax2.axhline(y=2, color='k', linestyle='-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Applied voltage [V]')
    ax2.set_ylabel('Slope [-]')
    ax2.set_xlim([1e-2,Vmax])
    ax2.set_ylim([0,int(slopemax)+1])
    ax2.grid(True,which='both',axis='both',color='gray', linestyle='-')
    # add arrow between VminMG and VmaxMG
    ax2.annotate('', xy=(minVminMG, int(slopemax)+1), xytext=(maxVmaxMG, int(slopemax)+1), arrowprops=dict(arrowstyle='<->',color='black',linewidth=3))
    # text in the middle of the arrow on a log scale
    ax2.text(minVminMG + 10**((np.log10(minVminMG)+np.log10(maxVmaxMG))/2), int(slopemax)+1.5, 'MG region', fontsize=20, ha='center')

    # ticks = []
    # for i in range(len(mob_TL_list)):
    #     ticks.append(sci_notation(mob_TL_list[i],0))
    # tickpos = np.asarray(mob_TL_list)
    # ax3.set_xticks(tickpos)
    # # replace ticks with custom ticks
    # ax3.set_xticklabels(ticks,fontsize=20,rotation=45, ha='center')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylabel('Calc. density [cm$^{-3}$]')
    ax3.set_xlabel('TL mobility [cm$^{2}$V$^{-1}$s$^{-1}$]')
    ax3.grid(True,which='both',axis='both',color='gray', linestyle='-',zorder=-1)
    # ax3.set_xlim([2,17])
    ax3.set_ylim([1e20/1e6,5e22/1e6])

    # custum legend
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='V$_{ons}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='V$_{inf}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='V$_{end}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], linestyle='-', color='k', label='N$_{T}$', markersize=10)]
    if ions > 0:
        legend_elements.append(Line2D([0], [0], linestyle='--', color='k', label='N$_{net}}$', markersize=10))
    ax3.legend(handles=legend_elements, loc='upper right',ncol=1)

    # ticks = []
    # for i in range(len(mob_TL_list)):
    #     ticks.append(sci_notation(i/1e6, sig_fig=1))
        # ticks.append(sci_notation(i*1e4, sig_fig=-1))
    ticks = []
    for i in range(len(mob_TL_list)):
        ticks.append(sci_notation(mob_TL_list[i],0))
    mu = np.asarray(mu)*1e4
    ax4.bar(np.arange(len(mu)),mu,color=colors)
    # plt.bar(loop,mu*1e4,color=colors[:ind],width=loop[1]-loop[0])
    ax4.axhline(1,linestyle='-',color='k')
    ax4.set_xticks(np.arange(len(mu)), ticks,fontsize=20)#,rotation=45)

    ax4.set_ylabel('Mobility [cm$^{2}$V$^{-1}$s$^{-1}$]')
    ax4.set_xlabel('TL mobility [cm$^{2}$V$^{-1}$s$^{-1}$]')
    ax4.set_yscale('log')
    ax4.set_ylim([1e-4 , 10])
    ax4.grid(b=True,which='both',axis='y',zorder=-1)

    # add super title to axes
    ax1.set_title('a)',position=(-0.1,0.97))
    ax2.set_title('b)',position=(-0.25,0.97))
    ax3.set_title('c)',position=(-0.1,0.95))
    ax4.set_title('d)',position=(-0.25,0.95))
    plt.tight_layout()


    plt.savefig(os.path.join(res_dir,f'{L_pero*1e9}_mob_TL_list.png'),dpi=300)

def run_all_dops(dop_TL_list,L_pero, eps_r_pero, ions, traps_pero, Vmax, Vscan, NP, tolPois, tolDens, couplePC, minAcc, maxAcc, grad, simss_device_parameters, str_save, with_TL, rerun=False, ions_in_TLs =False,figsize = (21,15),to_plot = [1e18,1e20,1e22,1e24]):

    res_dir = os.path.join(os.getcwd(),'results')
    
    nons_list,ninf_list,nend_list,mu  = [],[],[],[]
    if rerun:
        cmd_pars_list, ID_list = [], []
        # prepare cmd_pars_list
        if with_TL:
            layer = 'l2'
        else:
            layer = 'l1'
        for i in range(len(dop_TL_list)):
            cmd_pars = [{'par':'l1.N_D','val':str(dop_TL_list[i])},{'par':'l3.N_D','val':str(dop_TL_list[i])},{'par': 'l2.L', 'val': str(L_pero)},{'par':'l2.N_t_bulk','val':str(traps_pero)},{'par':'l2.N_anion','val':str(ions)},{'par':'l2.N_cation','val':str(ions)},{'par':'Vmax','val':str(Vmax)},{'par':'NP','val':str(NP)},{'par':'Vscan','val':str(Vscan)},{'par':'timeout','val':timeout},{'par':'tolPois','val':str(tolPois)},{'par':'tolDens','val':str(tolDens)},{'par':'couplePC','val':str(couplePC)},{'par':'minAcc','val':str(minAcc)},{'par':'maxAcc','val':str(maxAcc)},{'par':'grad','val':str(grad)}]
            if ions_in_TLs:
                cmd_pars.append({'par':'l1.ionsMayEnter','val':str(1)})
                cmd_pars.append({'par':'l3.ionsMayEnter','val':str(1)})
            cmd_pars_list.append(cmd_pars)
            ID_list.append(str(uuid.uuid4()))

        # run the simulation with joblib
        Parallel(n_jobs=min(len(cmd_pars_list),15))(delayed(run)(ID, cmd_pars,simss_device_parameters) for ID, cmd_pars in zip(ID_list, cmd_pars_list))

        # store values in a csv file
        df = pd.DataFrame({'ID':ID_list,'dop_TL':dop_TL_list})
        df.to_csv(os.path.join(res_dir,f'{L_pero*1e9}_dop_TL_list.csv'),index=False)
    else:
        df = pd.read_csv(os.path.join(res_dir,f'{L_pero*1e9}_dop_TL_list.csv'))
        ID_list = df['ID'].values

    # plot the results
    colors = plt.cm.viridis(np.linspace(0,1,len(dop_TL_list)+1))
    # remove the last color
    colors = colors[:-1]
    #invert colors
    colors = colors[::-1]
    fig = plt.figure(figsize=figsize)
    ax1,ax2,ax3,ax4 = plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224)
    # ax = plt.gca()
    minJ,maxJ = 1e20,1e-20
    VminMG,VmaxMG = 1e3,0
    slopemax = 0
    idx = -len(dop_TL_list)
    minVminMG,maxVmaxMG = 1e3,0
    for i,ep in enumerate(dop_TL_list):
        ID = ID_list[i]
        JV_file_name = os.path.join('SIMsalabim','SimSS',f'JV_{ID}.dat')
        data_JV = pd.read_csv(JV_file_name,sep=r'\s+')
        label = ''
        
        data_JV = data_JV[data_JV['Vext']>0]
        data_JV = data_JV[data_JV['Jext']>0]
        minJ = min(minJ,data_JV['Jext'].min())
        maxJ = max(maxJ,data_JV['Jext'].max())
        if Vscan == -1:
            # reverse the data
            data_JV = data_JV.iloc[::-1]

        V_slopef,J_slopef,slopesf,get_tangentf,idx_maxf,max_slopesf,tang_val_V1f,tang_val_V2f,tang_val_V3f,V1f,J1f,V2f,J2f,Vinf,Jinf = SCLC_get_data_plot(data_JV['Vext'],data_JV['Jext'])

        # print(max_slopesf)
        if max_slopesf > 2:
            nons_list.append(calc_net_charge(V1f,L_pero,eps_r_pero)/1e6)
            ninf_list.append(calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6)
            nend_list.append(calc_net_charge(V2f,L_pero,eps_r_pero)/1e6)
        else:
            nons_list.append(np.nan)
            ninf_list.append(np.nan)
            nend_list.append(np.nan)

        ## Calc Voltages
        Vnet = calc_Vnet_with_ions(ions,traps_pero,L_pero,eps_r_pero)
        Vtfl = calc_Vtfl(traps_pero,L_pero,eps_r_pero)
        Vsat = calc_Vsat(L_pero,5e24,0,eps_r_pero,295)
        Ntmin = calc_nt_min(L_pero,eps_r_pero,295)
        Vmin = calc_Vtfl(Ntmin,L_pero,eps_r_pero)

        # get mobility
        data_JV_mott = copy.deepcopy(data_JV)
        
        if Vsat > Vtfl:
            data_JV_mott['slopesf'] = slopesf
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=2]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']<=Vsat]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=Vtfl]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']<=2.2]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']>=1.8]
            if len(data_JV_mott) <= 5:
                mu.append(0)
            else:
                VminMG = min(data_JV_mott['Vext'])
                VmaxMG = max(data_JV_mott['Vext'])
                Mott_Gurney_fit = fit_MottGurney(data_JV_mott['Vext'],data_JV_mott['Jext'],1e-4,eps_r_pero,0,L_pero,var2fit=['mu'])
                mu.append(Mott_Gurney_fit[0])
            # print('Mott_Gurney_fit:',Mott_Gurney_fit)
        else:
            mu.append(0)

        # remove Vext = 0 from data_JV
        # if dop_TL_list[i] in to_plot:
        data_JV = data_JV[data_JV['Vext']!=0]

        ax1.plot(data_JV['Vext'],data_JV['Jext']/10,label=label,color=colors[i])
        if max_slopesf > 2:
            slopemax = max(slopemax,max(slopesf))
            ax1.plot(V1f,J1f/10,marker='s',color=colors[i])
            ax1.plot(Vinf,Jinf/10,marker='^',color=colors[i])
            ax1.plot(V2f,J2f/10,marker='o',color=colors[i])
            ax1.plot(V_slopef,tang_val_V1f/10,'--',color=colors[i])
            ax1.plot(V_slopef,tang_val_V2f/10,'--',color=colors[i])
            ax1.plot(V_slopef,tang_val_V3f/10,'--',color=colors[i])
        minVminMG = min(minVminMG,VminMG)
        maxVmaxMG = max(maxVmaxMG,VmaxMG)
        ax2.plot(V_slopef,slopesf,label=label,color=colors[i])
        if len(data_JV_mott) >= 5:
            ax2.axvspan(VminMG, VmaxMG, color=colors[i], alpha=0.3,zorder=idx)
        idx += 1


        if max_slopesf > 2:
            ax3.plot(dop_TL_list[i],calc_net_charge(V1f,L_pero,eps_r_pero)/1e6,marker='s',color=colors[i])
            ax3.plot(dop_TL_list[i],calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6,marker='^',color=colors[i])
            ax3.plot(dop_TL_list[i],calc_net_charge(V2f,L_pero,eps_r_pero)/1e6,marker='o',color=colors[i])
        # add diagonal line
        # ax3.plot([1e20/1e6,1e23/1e6],[1e20/1e6,1e23/1e6],'--',color='black')
        # if ions > 0:
        #     # plot diagonal - ions
        # ax3.plot([(ions+1e21)/1e6,1e23/1e6],[1e21/1e6,(1e23-ions)/1e6],'--',color='black')
        # horizontal line at trap density
        ax3.axhline(y=traps_pero/1e6,linestyle='-',color='black')
        ax3.axhline(y=(traps_pero-ions)/1e6,linestyle='--',color='black')
        # grey shadow ntmin to ntmax
        # ax3.axvspan(1e20/1e6, calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.5,zorder=-3)
        # ax3.axvspan(calc_nt_min(L_pero,eps_r_pero,295)/1e6,3*calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.05,zorder=-2)

    ax1.set_yscale('log')
    # ax1.set_ylim([1e-8,1e7])
    ax1.set_ylabel('Current density [mA m$^{-2}$]')
    ax1.set_xscale('log')
    ax1.set_xlim([1e-2,Vmax])
    ax1.set_ylim([10**np.floor(np.log10(minJ)-1),10**np.floor(np.log10(maxJ)+1)])
    ax1.set_xlabel('Applied voltage [V]')
    ax1.grid(True,which='both',axis='both',color='gray', linestyle='-')

    ax2.set_yscale('linear')
    ax2.set_ylim([0,int(slopemax)+1])
    ticks = ax2.get_yticks()
    ticks = ticks[ticks>3]
    ticks = np.append(ticks,[1,2])

    ticks = np.unique(ticks)
    ticks = np.sort(ticks)
    ax2.set_yticks(ticks)

    ax2.axhline(y=1, color='k', linestyle='-')
    ax2.axhline(y=2, color='k', linestyle='-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Applied voltage [V]')
    ax2.set_ylabel('Slope [-]')
    ax2.set_xlim([1e-2,Vmax])
    ax2.set_ylim([0,int(slopemax)+1])
    ax2.grid(True,which='both',axis='both',color='gray', linestyle='-')
    # add arrow between VminMG and VmaxMG
    ax2.annotate('', xy=(minVminMG, int(slopemax)+1), xytext=(maxVmaxMG, int(slopemax)+1), arrowprops=dict(arrowstyle='<->',color='black',linewidth=3))
    # text in the middle of the arrow on a log scale
    ax2.text(minVminMG + 10**((np.log10(minVminMG)+np.log10(maxVmaxMG))/2), int(slopemax)+1.5, 'MG region', fontsize=20, ha='center')

    # ticks = []
    # for i in range(len(dop_TL_list)):
    #     ticks.append(sci_notation(dop_TL_list[i],0))
    # tickpos = np.asarray(dop_TL_list)
    # ax3.set_xticks(tickpos)
    # # replace ticks with custom ticks
    # ax3.set_xticklabels(ticks,fontsize=20,rotation=45, ha='center')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylabel('Calc. density [cm$^{-3}$]')
    ax3.set_xlabel('Doping density [cm$^{-3}$]')
    ax3.grid(True,which='both',axis='both',color='gray', linestyle='-',zorder=-1)
    # ax3.set_xlim([2,17])
    ax3.set_ylim([1e20/1e6,5e22/1e6])

    # custum legend
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='V$_{ons}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='V$_{inf}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='V$_{end}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], linestyle='-', color='k', label='N$_{T}$', markersize=10)]
    if ions > 0:
        legend_elements.append(Line2D([0], [0], linestyle='--', color='k', label='N$_{net}}$', markersize=10))
    ax3.legend(handles=legend_elements, loc='upper right',ncol=1)

    # ticks = []
    # for i in range(len(dop_TL_list)):
    #     ticks.append(sci_notation(i/1e6, sig_fig=1))
        # ticks.append(sci_notation(i*1e4, sig_fig=-1))
    ticks = []
    for i in range(len(dop_TL_list)):
        ticks.append(sci_notation(dop_TL_list[i],0))
    mu = np.asarray(mu)*1e4
    ax4.bar(np.arange(len(mu)),mu,color=colors)
    # plt.bar(loop,mu*1e4,color=colors[:ind],width=loop[1]-loop[0])
    ax4.axhline(1,linestyle='-',color='k')
    ax4.set_xticks(np.arange(len(mu)), ticks,fontsize=20)#,rotation=45)

    ax4.set_ylabel('Mobility [cm$^{2}$V$^{-1}$s$^{-1}$]')
    ax4.set_xlabel('Doping density [cm$^{-3}$]')
    ax4.set_yscale('log')
    ax4.set_ylim([1e-4 , 10])
    ax4.grid(b=True,which='both',axis='y',zorder=-1)

    # add super title to axes
    ax1.set_title('a)',position=(-0.1,0.97))
    ax2.set_title('b)',position=(-0.25,0.97))
    ax3.set_title('c)',position=(-0.1,0.95))
    ax4.set_title('d)',position=(-0.25,0.95))
    plt.tight_layout()


    plt.savefig(os.path.join(res_dir,f'{L_pero*1e9}_dop_TL_list.png'),dpi=300)

def run_fig_2(L_pero, eps_r_pero, ions_bulk_list, traps_pero, Vmax, Vscan, NP, tolPois, tolDens, couplePC, minAcc, maxAcc, grad, simss_device_parameters, str_save, with_TL, rerun=False, ions_in_TLs =False,figsize = (23, 7), to_plot = [1e21,5e21,1e22]):

    res_dir = os.path.join(os.getcwd(),'results')
    nons_list,ninf_list,nend_list,mu  = [],[],[],[]
    ions_bulk_list = np.asarray(ions_bulk_list)
    if rerun:
        cmd_pars_list, ID_list = [], []
        # prepare cmd_pars_list
        if with_TL:
            layer = 'l2'
        else:
            layer = 'l1'
        for ions in ions_bulk_list:
            cmd_pars = [{'par': layer+'.L', 'val': str(L_pero)},{'par':layer+'.N_t_bulk','val':str(traps_pero)},{'par':layer+'.N_anion','val':str(ions)},{'par':layer+'.N_cation','val':str(ions)},{'par':'Vmax','val':str(Vmax)},{'par':'NP','val':str(NP)},{'par':'Vscan','val':str(Vscan)},{'par':'timeout','val':timeout},{'par':'tolPois','val':str(tolPois)},{'par':'tolDens','val':str(tolDens)},{'par':'couplePC','val':str(couplePC)},{'par':'minAcc','val':str(minAcc)},{'par':'maxAcc','val':str(maxAcc)},{'par':'grad','val':str(grad)}]
            if ions_in_TLs:
                cmd_pars.append({'par':'l1.ionsMayEnter','val':str(1)})
                cmd_pars.append({'par':'l3.ionsMayEnter','val':str(1)})
            cmd_pars_list.append(cmd_pars)
            ID_list.append(str(uuid.uuid4()))

        # run the simulation with joblib
        Parallel(n_jobs=min(len(cmd_pars_list),15))(delayed(run)(ID, cmd_pars,simss_device_parameters) for ID, cmd_pars in zip(ID_list, cmd_pars_list))

        # store values in a csv file
        df = pd.DataFrame({'ID':ID_list,'ions_bulk':ions_bulk_list})
        df.to_csv(os.path.join(res_dir,f'{L_pero*1e9}_'+str_save+'_ions_bulk_list.csv'),index=False)
    else:
        df = pd.read_csv(os.path.join(res_dir,f'{L_pero*1e9}_'+str_save+'_ions_bulk_list.csv'))
        ions_bulk_list = df['ions_bulk'].values
        ID_list = df['ID'].values

    # plot the results
    colors = plt.cm.viridis(np.linspace(0,1,len(ions_bulk_list)+1))
    # remove the last color
    colors = colors[:-1]
    #invert colors
    colors = colors[::-1]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, wspace=0.33, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    # ax1,ax2,ax3,ax4 = plt.subplot(221),plt.subplot(222),plt.subplot(223),plt.subplot(224)
    # ax = plt.gca()
    minJ,maxJ = 1e20,1e-20
    slopemax = 0
    idx = -1
    minVminMG,maxVmaxMG = 1e3,0
    for i,ions in enumerate(ions_bulk_list):
        ID = ID_list[i]
        JV_file_name = os.path.join('SIMsalabim','SimSS',f'JV_{ID}.dat')
        data_JV = pd.read_csv(JV_file_name,sep=r'\s+')
        label = sci_notation(ions/1e6,0)

        data_JV = data_JV[data_JV['Vext']>0]
        data_JV = data_JV[data_JV['Jext']>0]
        minJ = min(minJ,data_JV['Jext'].min())
        maxJ = max(maxJ,data_JV['Jext'].max())
        if Vscan == -1:
            # reverse the data
            data_JV = data_JV.iloc[::-1]

        V_slopef,J_slopef,slopesf,get_tangentf,idx_maxf,max_slopesf,tang_val_V1f,tang_val_V2f,tang_val_V3f,V1f,J1f,V2f,J2f,Vinf,Jinf = SCLC_get_data_plot(data_JV['Vext'],data_JV['Jext'])
        # print(max_slopesf)
        if max_slopesf > 2:
            nons_list.append(calc_net_charge(V1f,L_pero,eps_r_pero)/1e6)
            ninf_list.append(calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6)
            nend_list.append(calc_net_charge(V2f,L_pero,eps_r_pero)/1e6)
        else:
            nons_list.append(np.nan)
            ninf_list.append(np.nan)
            nend_list.append(np.nan)

        ## Calc Voltages
        Vnet = calc_Vnet_with_ions(ions,traps_pero,L_pero,eps_r_pero)
        Vtfl = calc_Vtfl(traps_pero,L_pero,eps_r_pero)
        Vsat = calc_Vsat(L_pero,5e24,0,eps_r_pero,295)
        Ntmin = calc_nt_min(L_pero,eps_r_pero,295)
        Vmin = calc_Vtfl(Ntmin,L_pero,eps_r_pero)

        # get mobility
        data_JV_mott = copy.deepcopy(data_JV)
        
        if Vsat > Vtfl:
            data_JV_mott['slopesf'] = slopesf
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=1]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']<=Vsat]
            data_JV_mott = data_JV_mott[data_JV_mott['Vext']>=Vtfl]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']<=2.2]
            data_JV_mott = data_JV_mott[data_JV_mott['slopesf']>=1.8]
            if len(data_JV_mott) <= 5:
                mu.append(0)
                continue
            VminMG = min(data_JV_mott['Vext'])
            VmaxMG = max(data_JV_mott['Vext'])
            Mott_Gurney_fit = fit_MottGurney(data_JV_mott['Vext'],data_JV_mott['Jext'],1e-4,eps_r_pero,0,L_pero,var2fit=['mu'])
            mu.append(Mott_Gurney_fit[0])
            # print('Mott_Gurney_fit:',Mott_Gurney_fit)
        else:
            mu.append(0)

        # remove Vext = 0 from data_JV
        if ions in to_plot:
            data_JV = data_JV[data_JV['Vext']!=0]
            ax1.plot(data_JV['Vext'],data_JV['Jext']/10,label=label,color=colors[i])
            if max_slopesf > 2:
                slopemax = max(slopemax,max(slopesf))
                # ax1.plot(V1f,J1f/10,marker='s',color=colors[i])
                # ax1.plot(Vinf,Jinf/10,marker='^',color=colors[i])
                # ax1.plot(V2f,J2f/10,marker='o',color=colors[i])
                # ax1.plot(V_slopef,tang_val_V1f/10,'--',color=colors[i],linewidth=2)
                # ax1.plot(V_slopef,tang_val_V2f/10,'--',color=colors[i],linewidth=2)
                # ax1.plot(V_slopef,tang_val_V3f/10,'--',color=colors[i],linewidth=2)
            minVminMG = min(minVminMG,VminMG)
            maxVmaxMG = max(maxVmaxMG,VmaxMG)
            ax2.plot(V_slopef,slopesf,label=label,color=colors[i])
            ax2.axvspan(VminMG, VmaxMG, color=colors[i], alpha=0.2,zorder=idx)
            idx -= 1


        if max_slopesf > 2:
            ax3.plot((traps_pero-ions)/1e6,calc_net_charge(V1f,L_pero,eps_r_pero)/1e6,marker='s',color=colors[i])
            ax3.plot((traps_pero-ions)/1e6,calc_net_charge(Vinf,L_pero,eps_r_pero)/1e6,marker='^',color=colors[i])
            ax3.plot((traps_pero-ions)/1e6,calc_net_charge(V2f,L_pero,eps_r_pero)/1e6,marker='o',color=colors[i])
        # add diagonal line
        ax3.plot([1e20/1e6,1e23/1e6],[1e20/1e6,1e23/1e6],'--',color='black')
        # if ions > 0:
        #     # plot diagonal - ions
        #     ax3.plot([(ions+1e21)/1e6,1e23/1e6],[1e21/1e6,(1e23-ions)/1e6],'--',color='black')
        # horizontal line at trap density
        ax3.axhline(y=traps_pero/1e6,linestyle='-',color='black')
        # grey shadow ntmin to ntmax
        ax3.axvspan(1e20/1e6, calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.75,zorder=-1)
        # ax3.axvspan(calc_nt_min(L_pero,eps_r_pero,295)/1e6,3*calc_nt_min(L_pero,eps_r_pero,295)/1e6, color='lightgray', alpha=0.05,zorder=-2)

    ax1.set_yscale('log')
    # ax1.set_ylim([1e-8,1e7])
    ax1.set_ylabel('Current density [mA m$^{-2}$]')
    ax1.set_xscale('log')
    ax1.set_xlim([1e-2,Vmax])
    ax1.set_ylim([10**np.floor(np.log10(minJ)-1),1e7])#10**np.floor(np.log10(maxJ)+1)])
    ax1.set_xlabel('Applied voltage [V]')
    ax1.grid(True,which='both',axis='both',color='lightgray', linestyle='-')

    ax2.set_yscale('linear')
    ax2.set_ylim([0,int(slopemax)+1])
    ticks = ax2.get_yticks()
    ticks = ticks[ticks>3]
    ticks = np.append(ticks,[1,2])

    ticks = np.unique(ticks)
    ticks = np.sort(ticks)
    ax2.set_yticks(ticks)

    ax2.axhline(y=1, color='k', linestyle='-')
    ax2.axhline(y=2, color='k', linestyle='-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Applied voltage [V]')
    ax2.set_ylabel('Slope [-]')
    ax2.set_xlim([1e-2,Vmax])
    ax2.set_ylim([0,int(slopemax)+1])
    ax2.grid(True,which='both',axis='both',color='lightgray', linestyle='-')
    # add arrow between VminMG and VmaxMG
    ax2.annotate('', xy=(minVminMG, int(slopemax)+1), xytext=(maxVmaxMG, int(slopemax)+1), arrowprops=dict(arrowstyle='<->',color='black',linewidth=3))
    # text in the middle of the arrow on a log scale

    ax2.text(minVminMG + 10**((np.log10(minVminMG)+np.log10(maxVmaxMG))/2), int(slopemax)+1.5, 'MG region', fontsize=20, ha='center')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylabel('Calc. density [cm$^{-3}$]')
    ax3.set_xlabel('Net charge density [cm$^{-3}$]')
    ax3.grid(True,which='both',axis='both',color='gray', linestyle='-',zorder=-1)
    ax3.set_xlim([1e20/1e6,5e22/1e6])
    ax3.set_ylim([1e20/1e6,5e22/1e6])

    # custum legend
    legend_elements = [Line2D([0], [0], marker='s', color='None', label='V$_{ons}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='^', color='None', label='V$_{inf}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], marker='o', color='None', label='V$_{end}$', markerfacecolor='k', markersize=10),
                        Line2D([0], [0], linestyle='-', color='k', label='N$_{T}$', markersize=10)]
    if ions > 0:
        legend_elements.append(Line2D([0], [0], linestyle='--', color='k', label='N$_{net}}$', markersize=10))
    # add lightgrey shadow
    legend_elements.append(Patch(facecolor='lightgray', edgecolor='black',label='N$_{net}$ < N$_{min}$',alpha=0.75))
    ax3.legend(handles=legend_elements, loc='lower right',ncol=2,fontsize=20,frameon=True)

    # ticks = []
    # for i in ions_bulk_list:
    #     ticks.append(sci_notation(i/1e6, sig_fig=1))
    #     # ticks.append(sci_notation(i*1e4, sig_fig=-1))
    # mu = np.asarray(mu)*1e4
    # ax4.bar(np.arange(len(mu)),mu,color=colors)
    # # plt.bar(loop,mu*1e4,color=colors[:ind],width=loop[1]-loop[0])
    # ax4.axhline(1,linestyle='-',color='k')
    # ax4.set_xticks(np.arange(len(mu)), ticks,fontsize=20,rotation=45)

    # ax4.set_ylabel('Mobility [cm$^{2}$V$^{-1}$s$^{-1}$]')
    # ax4.set_xlabel('Ion density [cm$^{-3}$]')
    # ax4.set_yscale('log')
    # ax4.set_ylim([1e-4 , 10])
    # ax4.grid(b=True,which='both',axis='y',zorder=-1)

    # add super title to axes
    ax1.set_title('a)',position=(-0.1,0.97))
    ax2.set_title('b)',position=(-0.1,0.97))
    ax3.set_title('c)',position=(-0.1,0.97))
    # ax4.set_title('d)',position=(-0.25,0.95))

    # add colorbar to plot for scan speeds log scale
    custom_colors = plt.cm.viridis(np.linspace(0,1,len(ions_bulk_list)+1))
    #remove the last color
    custom_colors = custom_colors[:-1]
    # reverse colors
    custom_colors = custom_colors[::-1]
    # make colorbar

    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    # Create a custom colormap from the array of colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=LogNorm(vmin=min(ions_bulk_list), vmax=max(ions_bulk_list)))
    sm.set_array([])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider1 = make_axes_locatable(ax3)
    cax3= divider1.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(sm, cax=cax3)
    cbar3.set_label('Ion density [m$^{-3}$]')

    plt.tight_layout()


    plt.savefig(os.path.join(res_dir,f'figure_2.pdf'),dpi=300, format="pdf", bbox_inches="tight")
        