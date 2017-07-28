""" my PhasePicker, based on PhasePApy package """

import sys
import numpy as np
from scipy import linalg as LA
sys.path.append("/Volumes/Untitled/Wenchuan_Mostafa/PhasePApy-master")
from phasepapy.phasepicker import aicdpicker
from phasepapy.phasepicker import ktpicker
from phasepapy.phasepicker import fbpicker
from obspy import *
import pdb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# choose the phase picker 1: aicdpicker, 2: ktpicker, 3: fbpicker
pick_choice = 3

if pick_choice == 1:
	picker = aicdpicker.AICDPicker(t_ma = 3, nsigma = 6, t_up = 2, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
elif pick_choice == 2:
	picker = ktpicker.KTPicker(t_win = 1, t_ma = 3, nsigma = 6, t_up = 0.78, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
elif pick_choice == 3:
	picker = fbpicker.FBPicker(t_long = 5, freqmin = 0.8, mode = 'rms', t_ma = 20, nsigma = 5, t_up = 0.4, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
else:
	print('choose another picker')
	pass

# read filename 
detect_dir = '/Volumes/Untitled/Wenchuan_Mostafa/untitled folder/'
data_dir = '/Users/Hugh/Documents/Aliyun/Aliyun_after_data/'
detect_event = read(detect_dir+'20080801210259.QCH.BHN')


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
stz = read(data_dir+net+'.'+sta+'.'+str(year)+str(day)+'000000.BHZ')
stn = read(data_dir+net+'.'+sta+'.'+str(year)+str(day)+'000000.BHN')
ste = read(data_dir+net+'.'+sta+'.'+str(year)+str(day)+'000000.BHE')
starttime = UTCDateTime(str(year)+str(day)+'T'+str(hour_start).zfill(2)+str(minute_start).zfill(2)+str(second_start).zfill(2))
starttime2 = starttime-3 # save 3 more seconds for crosscorrelation
#end_points = 100*(hour_end*3600+minute_end*60+second_end)
#trz = stz[0].filter('bandpass',freqmin=0.5,freqmax=20)[start_points:end_points]
#trn = stn[0].filter('bandpass',freqmin=0.5,freqmax=20)[start_points:end_points]
#tre = ste[0].filter('bandpass',freqmin=0.5,freqmax=20)[start_points:end_points]
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

# construct p and s filter
wflen = len(trz)-300
COR_NN,COR_NE,COR_NZ = np.zeros(wflen),np.zeros(wflen),np.zeros(wflen)
COR_EN,COR_EE,COR_EZ = np.zeros(wflen),np.zeros(wflen),np.zeros(wflen)
COR_ZN,COR_ZE,COR_ZZ = np.zeros(wflen),np.zeros(wflen),np.zeros(wflen)
for i in range(0,wflen):
	COR_NN[i] = sum(trn[i:i+300]*trn[i:i+300])
	COR_NE[i] = sum(trn[i:i+300]*tre[i:i+300])
	COR_NZ[i] = sum(trn[i:i+300]*trz[i:i+300])
        COR_EN[i] = sum(tre[i:i+300]*trn[i:i+300])
        COR_EE[i] = sum(tre[i:i+300]*tre[i:i+300])
        COR_EZ[i] = sum(tre[i:i+300]*trz[i:i+300])
        COR_ZN[i] = sum(trz[i:i+300]*trn[i:i+300])
        COR_ZE[i] = sum(trz[i:i+300]*tre[i:i+300])
        COR_ZZ[i] = sum(trz[i:i+300]*trz[i:i+300])

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

# take a look at the polarized data
#plt.plot(trz_pol)
#plt.plot(trn_pol)
#plt.plot(tre_pol)
#plt.show()
# pick the phase on polarized traces
scnlz, picksz, polarityz, snrz, uncertz = picker.picks(trz_pol) # or trz_pol 
print('scnl:', scnlz)
print('picks:', picksz)
print('polarity:', polarityz)
print('signal to noise ratio:', snrz)
print('uncertainty:', uncertz)

scnln, picksn, polarityn, snrn, uncertn = picker.picks(trn_pol) # or trn_pol 
print('scnl:', scnln)
print('picks:', picksn)
print('polarity:', polarityn)
print('signal to noise ratio:', snrn)
print('uncertainty:', uncertn)


scnle, pickse, polaritye, snre, uncerte = picker.picks(tre_pol) # or tre_pol 
print('scnl:', scnle)
print('picks:', pickse)
print('polarity:', polaritye)
print('signal to noise ratio:', snre)
print('uncertainty:', uncerte)


#summary = aicdpicker.AICDSummary(picker, trz)
#summary.plot_summary()
#summary.plot_picks()
