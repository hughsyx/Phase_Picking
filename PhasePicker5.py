""" my PhasePicker, based on PhasePApy package """
from os import listdir
from os import linesep
import sys
import numpy as np
from scipy import linalg as LA
sys.path.append("/Users/Hugh/Documents/Aliyun/PhasePApy-master")
from phasepapy.phasepicker import aicdpicker
from phasepapy.phasepicker import ktpicker
from phasepapy.phasepicker import fbpicker
from obspy import *
import pdb
import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# choose the phase picker 1: aicdpicker, 2: ktpicker, 3: fbpicker

def Get_traces(detect_file):

# read filename 
	detect_dir = '/Users/Hugh/Documents/Aliyun/after_filt_templates_organized/'
	data_dir = '/Users/Hugh/Documents/Aliyun/Aliyun_after_data/'
	detect_event = read(detect_dir+detect_file)

	starttime = detect_event[0].stats.starttime
	endtime = detect_event[0].stats.endtime

	net = detect_event[0].stats.network
	sta = detect_event[0].stats.station
	year = starttime.year
	day = starttime.julday
	hour_start = starttime.hour
	minute_start = starttime.minute
	second_start = starttime.second
	#hour_end = endtime.hour
	#minute_end = endtime.minute
	#second_end = endtime.second

	picksz,picksn,pickse = [],[],[]
	trz_new,trn_new,tre_new = [],[],[]
	if year == 1970: # sac header got contaminated
		year = UTCDateTime(detect_file.split('.')[0]).datetime.year
		day = UTCDateTime(detect_file.split('.')[0]).datetime.timetuple().tm_yday
		hour_start = UTCDateTime(detect_file.split('.')[0]).datetime.hour
		minute_start = UTCDateTime(detect_file.split('.')[0]).datetime.minute
		second_start = UTCDateTime(detect_file.split('.')[0]).datetime.second
		endtime = UTCDateTime(detect_file.split('.')[0])+(endtime-starttime)

	stz = read(data_dir+net+'.'+sta+'.'+str(year)+str(day)+'*.BHZ')
	stn = read(data_dir+net+'.'+sta+'.'+str(year)+str(day)+'*.BHN')
	ste = read(data_dir+net+'.'+sta+'.'+str(year)+str(day)+'*.BHE')
	starttime = UTCDateTime(str(year)+str(day)+'T'+str(hour_start).zfill(2)+str(minute_start).zfill(2)+str(second_start).zfill(2))
	starttime2 = starttime-1 # save 1 more seconds for crosscorrelation
	
	trz = stz[0].slice(starttime2,endtime).filter('bandpass',freqmin=0.5,freqmax=20)
	trn = stn[0].slice(starttime2,endtime).filter('bandpass',freqmin=0.5,freqmax=20)
	tre = ste[0].slice(starttime2,endtime).filter('bandpass',freqmin=0.5,freqmax=20)
	endtime = trz.stats.endtime # update the endtime
	#wfstart = UTCDateTime(str(year)+str(day)+'T'+str(hour_start)+str(minute_start)+str(second_start))
	# take a look at the polarized data
	#plt.plot(trz)
	#plt.plot(trn)
	#plt.plot(tre)
	#plt.show()

	# see if one of the component might be noise
	cc = correlation_coefficient(abs(trz.data),abs(trn.data))+correlation_coefficient(abs(tre.data),abs(trn.data))+correlation_coefficient(abs(trz.data),abs(tre.data))
	if cc>1:
		# construct p and s filter
		wflen = len(trz)-100
		COR_NN,COR_NE,COR_NZ = np.zeros(wflen),np.zeros(wflen),np.zeros(wflen)
		COR_EN,COR_EE,COR_EZ = np.zeros(wflen),np.zeros(wflen),np.zeros(wflen)
		COR_ZN,COR_ZE,COR_ZZ = np.zeros(wflen),np.zeros(wflen),np.zeros(wflen)
		for i in range(0,wflen):
			COR_NN[i] = sum(trn[i:i+100]*trn[i:i+100])
			COR_NE[i] = sum(trn[i:i+100]*tre[i:i+100])
			COR_NZ[i] = sum(trn[i:i+100]*trz[i:i+100])
        		COR_EN[i] = sum(tre[i:i+100]*trn[i:i+100])
        		COR_EE[i] = sum(tre[i:i+100]*tre[i:i+100])
        		COR_EZ[i] = sum(tre[i:i+100]*trz[i:i+100])
       			COR_ZN[i] = sum(trz[i:i+100]*trn[i:i+100])
        		COR_ZE[i] = sum(trz[i:i+100]*tre[i:i+100])
        		COR_ZZ[i] = sum(trz[i:i+100]*trz[i:i+100])

		r = np.zeros(wflen)
		p = np.zeros(wflen)
		s = np.zeros(wflen)
		for i in range(0,wflen):
			eigenValues,eigenVectors = LA.eig(np.asarray([COR_NN[i],COR_NE[i],COR_NZ[i],COR_EN[i],COR_EE[i],COR_EZ[i],COR_ZN[i],COR_ZE[i],COR_ZZ[i]]).reshape(3,3))
			idx = eigenValues.argsort()
			eigenValues = eigenValues[idx]
			eigenVectors = eigenVectors[:,idx]
			r[i] = 1-(eigenValues[1]+eigenValues[0])/eigenValues[2]/2
			p[i] = r[i]*abs(eigenVectors[2,2])
			s[i] = r[i]*(1-abs(eigenVectors[2,2]))

	# polarize each component
	#pdb.set_trace()
		trz_pol = trz.slice(starttime,endtime)
		trn_pol = trn.slice(starttime,endtime)
		tre_pol = tre.slice(starttime,endtime)

		trz_pol.data = trz_pol.data*p
		trn_pol.data = trn_pol.data*s
		tre_pol.data = tre_pol.data*s
	else:
		trz_pol = trz.copy()
		trn_pol = trn.copy()
		tre_pol = tre.copy()

	return trz_pol,trn_pol,tre_pol,trz,trn,tre

def GetPicks(trz_pol,trn_pol,tre_pol,trz,trn,tre,pick_choice):
        if pick_choice == 1:
                picker = aicdpicker.AICDPicker(t_ma = 3, nsigma = 5, t_up = 0.78, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
        elif pick_choice == 2:
                picker = ktpicker.KTPicker(t_win = 1, t_ma = 3, nsigma = 6, t_up = 0.78, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
        elif pick_choice == 3:
                picker = fbpicker.FBPicker(t_long = 5, freqmin = 0.8, mode = 'rms', t_ma = 20, nsigma = 5, t_up = 0.4, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
        else:
                print('choose another picker')
	
	endtime = trz.stats.endtime
	starttime = trz.stats.starttime
	# take a look at the polarized data
	#plt.plot(trz_pol)
	#plt.plot(trn_pol)
	#plt.plot(tre_pol)
	#plt.show()
	# pick the phase on polarized traces
	scnlz, picksz, polarityz, snrz, uncertz = picker.picks(trz) # or trz_pol 
	#print('scnl:', scnlz)
	#print('picks:', picksz)
	#print('polarity:', polarityz)
	#print('signal to noise ratio:', snrz)
	#print('uncertainty:', uncertz)

	scnln, picksn, polarityn, snrn, uncertn = picker.picks(trn_pol) # or trn_pol 
	#print('scnl:', scnln)
	#print('picks:', picksn)
	#print('polarity:', polarityn)
	#print('signal to noise ratio:', snrn)
	#print('uncertainty:', uncertn)

	if len(picksn)>2:
		picksn2 = list(picksn)
		amp_tmp = np.zeros((len(picksn)))
		for i in range(0,len(picksn)-1):
			amp_tmp[i] = max(abs(trn.slice(picksn[i],picksn[i+1]).data))
		amp_tmp[-1] = max(abs(trn.slice(picksn[-1],endtime).data)) # was using np.mean
		sort_indx = amp_tmp.argsort()[::-1]
		remove_indx = sort_indx[2:]
		for idx in remove_indx:
			picksn.remove(picksn2[idx])
		del picksn2,amp_tmp,sort_indx,remove_indx,idx

	scnle, pickse, polaritye, snre, uncerte = picker.picks(tre_pol) # or tre_pol 
	#print('scnl:', scnle)
	#print('picks:', pickse)
	#print('polarity:', polaritye)
	#print('signal to noise ratio:', snre)
	#print('uncertainty:', uncerte)
	if len(pickse)>2:
                pickse2 = list(pickse)
                amp_tmp = np.zeros((len(pickse)))
                for i in range(0,len(pickse)-1):
                        amp_tmp[i] = max(abs(tre.slice(pickse[i],pickse[i+1]).data))
                amp_tmp[-1] = max(abs(tre.slice(pickse[-1],endtime).data)) # was using np.mean
                sort_indx = amp_tmp.argsort()[::-1]
                remove_indx = sort_indx[2:]
                for idx in remove_indx:
                        pickse.remove(pickse2[idx])
                del pickse2,amp_tmp,sort_indx,remove_indx,idx
	
        trz_new = trz.slice(starttime,endtime)
        trn_new = trn.slice(starttime,endtime)
        tre_new = tre.slice(starttime,endtime)

	del trz,trn,tre,trz_pol,trn_pol,tre_pol

	return picksz,picksn,pickse,trz_new,trn_new,tre_new
	#summary = aicdpicker.AICDSummary(picker, trz)
	#summary.plot_summary()
	#summary.plot_picks()

def correlation_coefficient(A,B):
	A_mA = np.mean(A)
	B_mB = np.mean(B)
	ssA = A_mA**2
	ssB = B_mB**2
	return np.dot(A_mA,B_mB.T)/np.sqrt(ssA*ssB)


def combine_picks(picks):
	InSec = np.zeros(len(picks))
	for i in range(0,len(picks)):
		InSec[i] = (picks[i].datetime-datetime.datetime(1970,1,1)).total_seconds()
	
	filtered = InSec
	filtered2 = filtered
	while len(filtered2) > 2:
		filtered = filtered2
		u = np.mean(filtered)
		s = np.std(filtered)
		filtered2 = [e for e in filtered if(u-s <e<u+s)]
	
	#return UTCDateTime(np.mean(filtered))
	return UTCDateTime(min(filtered))	

def removefromS(picks,phaseP):
	#InSec = np.zeros(len(picks))
	InSecP = (phaseP.datetime-datetime.datetime(1970,1,1)).total_seconds()
	picks_new = list(picks)
	idx_remove = []
	for i in range(0,len(picks)):
		InSecS = (picks[i].datetime-datetime.datetime(1970,1,1)).total_seconds()
		if InSecP > InSecS-1:
			idx_remove.append(i)
	idx_remove = np.sort(idx_remove)[::-1]
	#pdb.set_trace()
	for i in range(0,len(idx_remove)):
		picks_new.remove(picks[idx_remove[i]])

	return picks_new


def runPickers(detect_file):
	pickp = []
	picks = []
	#detect_file = '20080731040120.LUYA.BHZ'
	trz,trn,tre = [],[],[]
	trz_pol,trn_pol,tre_pol,trz0,trn0,tre0 = Get_traces(detect_file)
		for pick_choice in range(0,3):
			pick_choice += 1
			try:
				picksz_new,picksn_new,pickse_new,trz,trn,tre = GetPicks(trz_pol,trn_pol,tre_pol,trz0,trn0,tre0,pick_choice)
				for p in picksz_new:
					pickp.append(p)
				for s in picksn_new:
					picks.append(s)
				for s in pickse_new:
					picks.append(s)
			except Exception:
				sys.exc_clear()

	#print(pickp)
	#print(picks)
	if len(pickp) >0:
		year = trz.stats.starttime.year
        	month = trz.stats.starttime.month
        	day = trz.stats.starttime.day
		station = trz.stats.station
		phaseP = combine_picks(pickp)
		picks_new = removefromS(picks,phaseP)
		if len(picks_new)>0:
			phaseS = combine_picks(picks_new)
			print(phaseP)
			print(phaseS)
			f1 = open('submit_filt_template.csv','a')
			f1.write(station+','+(phaseP+8*3600).strftime('%Y%m%d%H%M%S.%f')+','+'P'+linesep)
			f1.write(station+','+(phaseS+8*3600).strftime('%Y%m%d%H%M%S.%f')+','+'S'+linesep)
			f1.close()
			# save in sac header
			phaseP_insec = (phaseP.datetime-datetime.datetime(year,month,day)).total_seconds()
			phaseS_insec = (phaseS.datetime-datetime.datetime(year,month,day)).total_seconds()
			# kt0 for P and kt4 for S
			trz.stats.sac.kt0 = phaseP_insec
			trz.stats.sac.kt4 = phaseS_insec
			trn.stats.sac.kt0 = phaseP_insec
			trn.stats.sac.kt4 = phaseS_insec
			tre.stats.sac.kt0 = phaseP_insec
			tre.stats.sac.kt4 = phaseS_insec
			# new file name
			dir_out = '/Users/Hugh/Documents/Aliyun/PICK_filt_template/'
		else:
			dir_out = '/Users/Hugh/Documents/Aliyun/PICK_FAIL_filt_template/'
		
		detect_file_z = phaseP.datetime.strftime('%Y%m%d%H%M%S')+'.'+detect_file.split('.')[1]+'.'+'BHZ'+'.SAC'
        	detect_file_n = phaseP.datetime.strftime('%Y%m%d%H%M%S')+'.'+detect_file.split('.')[1]+'.'+'BHN'+'.SAC'
        	detect_file_e = phaseP.datetime.strftime('%Y%m%d%H%M%S')+'.'+detect_file.split('.')[1]+'.'+'BHE'+'.SAC'
	else:
		dir_out = '/Users/Hugh/Documents/Aliyun/PICK_FAIL_filt_template/'
		detect_file_z = detect_file.split('.')[0]+'.'+detect_file.split('.')[1]+'.'+'BHZ'+'.SAC'
		detect_file_n = detect_file.split('.')[0]+'.'+detect_file.split('.')[1]+'.'+'BHN'+'.SAC'
		detect_file_e = detect_file.split('.')[0]+'.'+detect_file.split('.')[1]+'.'+'BHE'+'.SAC'
		# write sac files
	if len(trz)>0:
		trz.write(dir_out+detect_file_z,format='SAC')
	else:
		trz0.write(dir_out+detect_file_z,format='SAC')
	if len(trn)>0:
		trn.write(dir_out+detect_file_n,format='SAC')
	else:
		trn0.write(dir_out+detect_file_n,format='SAC')
	if len(tre)>0:	
		tre.write(dir_out+detect_file_e,format='SAC')
	else:
		tre0.write(dir_out+detect_file_e,format='SAC')
detect_file = '20080802075051.PWU.BHZ'
runPickers(detect_file)
print(detect_file)
#detect_dir = '/Users/Hugh/Documents/Aliyun/after_filt_templates_organized/'
#for detect_file in listdir(detect_dir):
#	print(detect_file)
#	runPickers(detect_file)
