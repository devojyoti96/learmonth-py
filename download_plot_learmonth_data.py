import execnet,matplotlib 
import pandas as pd,seaborn as sns,matplotlib.pyplot as plt,sys,os,numpy as np
from scipy.signal import medfilt
from scipy import interpolate
import matplotlib.dates as md
from optparse import OptionParser

def call_python_version(Version, Module, Function, ArgumentList):
	'''
	Function to run Python2 code from Python 3
	Parameters
	----------
	Version : str
		Python version name
	Module : str
		Name of the module to load
	Function : str
		Function of the module to use
	ArgumentList : list
		List pof function argument
	Returns
	-------
	Function output
	'''
	gw      = execnet.makegateway("popen//python=python%s" % Version)
	channel = gw.remote_exec("""
	from %s import %s as the_function
	channel.send(the_function(*channel.receive()))
	""" % (Module, Function))
	channel.send(ArgumentList)
	return channel.receive()
    
def fill_nan(arr):
	'''
	Function to interpolate to nan values
	Parameters
	----------
	arr : np.array
		1-D numpy array
	Returns
	-------
	np.array
		1-D nan interpolated numpy array
	'''
	try:
		med_fill_value=np.nanmedian(arr)
		inds = np.arange(arr.shape[0])
		good = np.where(np.isfinite(arr))
		f=interpolate.interp1d(inds[good], arr[good],bounds_error=False,kind='linear',fill_value='extrapolate')
		out_arr=np.where(np.isfinite(arr),arr,f(inds))
	except Exception as e:
		print (e)
		out_arr=arr
	return out_arr
	
def backsub(data):	
	'''
	Function to subtract background per channel
	'''
	for sb in np.arange(data.shape[0]):
		data[sb, :] = data[sb, :]/np.median(data[sb, :])
	return data
    	

def srs_to_pd(srs_file,pd_file,bkg_sub=False,do_flag=True):	
	'''
	Function to conver Learmonth SRS datafile into pandas dataframe
	Parameters
	----------
	srs_file : str
		Name of the SRS file
	pd_file : str
		Name of the output pandas file
	bkg_sub : bool
		Do background subtraction or not
	do_flag : bool
		Flag bad data or not
	Returns
	-------
	str
		Output pandas dataframe file
	'''
	print ('Converting SRS file to pandas datafile...\n')
	raw_data = call_python_version("2.7", "srs_data", "main",[srs_file]) 
	a_band_data=raw_data[0] # 25 to 75 MHz
	b_band_data=raw_data[1] # 75 to 180 MHz
	timestamps=raw_data[2]
	timestamps=pd.to_datetime(timestamps,format='%d/%m/%y, %H:%M:%S')	
	a_band_freqs=list(a_band_data[0].keys())
	b_band_freqs=list(b_band_data[0].keys())
	freqs=a_band_freqs+b_band_freqs
	freqs=np.array(freqs)
	freqs=np.round(freqs,1)
	x=[]                                                                                                                                                                                                      
	for i in range(len(a_band_data)):
		a_data=list(a_band_data[i].values())
		b_data=list(b_band_data[i].values())	
		a_b_data=a_data+b_data
		x.append(a_b_data) 

	x=np.array(x).astype('float')
	full_band_data=pd.DataFrame(x,index=timestamps,columns=freqs)  
	full_band_data=full_band_data.sort_index(0) 
	full_band_data=full_band_data.sort_index(1)
	full_band_data=full_band_data.transpose()

	final_data=full_band_data.to_numpy().astype('float')
	# Flagging bad channels
	if do_flag:
		final_data[488:499,:]=np.nan
		final_data[524:533,:]=np.nan
		final_data[540:550,:]=np.nan
		final_data[638:642,:]=np.nan
		final_data[119:129,:]=np.nan
		final_data[108:111,:]=np.nan
		final_data[150:160,:]=np.nan
		final_data[197:199,:]=np.nan
		final_data[285:289,:]=np.nan
		final_data[621:632,:]=np.nan
		final_data[592:600,:]=np.nan
		final_data[700:712,:]=np.nan
		final_data[410:416,:]=np.nan
		final_data[730:741,:]=np.nan
		final_data[635:645,:]=np.nan
		# Flagging calibration times
		y=np.nanmedian(final_data,axis=0)
		c=y/medfilt(y,1001)
		c_std=np.nanstd(c)
		pos=np.where(c>1+(10*c_std))
		final_data[...,pos]=np.nan
	for i in range(final_data.shape[1]):
		final_data[:,i]=fill_nan(final_data[:,i])
	if do_flag:
		final_data[780:,:]=np.nan #Flag edge channels	
	if bkg_sub:
		final_data=backsub(final_data)
	full_band_data=pd.DataFrame(final_data,index=freqs,columns=timestamps)  
	full_band_data.to_pickle(pd_file+'.pd')
	
	return pd_file+'.pd'
	
	
def plot_learmonth_DS(pd_file,save_file='',start_time='',end_time=''):
	'''
	Function to plot dynamic spectrum
	Parameters
	----------
	pd_file : str
		Pandas datafile of Learmonth dynamic spectrum
	save_file : str
		Filename to save plot
	start_time : str
		Start time (format : dd-mm-yyyy hh:mm:ss)
	end_time : str
		End time (format : dd-mm-yyyy hh:mm:ss)
	Returns
	-------
	str
		Save plot file name
	'''
	print ('Making final dynamic spectrum\n')
	if save_file=='':
		save_file=pd_file.split('.pd')[0]+'.pdf'
	pd_data=pd.read_pickle(pd_file)
	start_time=pd.to_datetime(start_time)	
	end_time=pd.to_datetime(end_time)
	timestamps=pd_data.columns
	freqs=pd_data.index	
	if start_time!='' and end_time!='':
		pos=((timestamps>=start_time) & (timestamps<=end_time))
		sel_timestamps=timestamps[pos]	
	elif start_time==''  and end_time!='':
		pos=(timestamps<=end_time)	
		sel_timestamps=timestamps[pos]	
	elif start_time!='' and end_time=='':
		pos=(timestamps>=start_time)
		sel_timestamps=timestamps[pos]	
	else:
		sel_timestamps=timestamps	
	sel_data=pd_data[sel_timestamps]	
	matplotlib.rcParams.update({'font.size': 15})
	plt.style.use('seaborn-colorblind')
	freq_ind=[]
	freq_list=[]
	time_ind=[]
	time_list=[]
	for i in range(0,len(sel_timestamps),int(len(sel_timestamps)/10)): 
		time_ind.append(i) 
		time_list.append(sel_timestamps[i].time())
		
	for i in range(0,len(freqs),int(len(freqs)/10)): 
		freq_ind.append(i) 
		freq_list.append(freqs[i])
	plt.figure(figsize=(20,8))
	s=sns.heatmap(sel_data,robust=True,cbar_kws={'label': 'Flux density (arbitrary unit)'},rasterized=True)
	s.invert_yaxis()
	plt.yticks(freq_ind[:-1],freq_list[:-1])
	plt.xticks(time_ind[:-1],time_list[:-1],rotation=30)
	plt.xlabel('Timestamp (UTC)')
	plt.ylabel('Frequency (MHz)')
	t=timestamps[int(len(timestamps)/2)] 
	datestamp=t.date()
	plt.title('Learmonth Spectrograph\nDate : '+str(t.day)+' '+t.month_name()+' '+str(t.year))
	plt.tight_layout()
	plt.savefig(save_file)
	plt.show() 
	return save_file

def download_learmonth(start_time='',end_time=''):
	'''
	Function to dwnload Learmonth spectrograph data
	Parameters
	----------
	start_time : str
		Start time (format : dd-mm-yyyy hh:mm:ss)
	end_time : str
		End time (format : dd-mm-yyyy hh:mm:ss)
	Returns
	-------
	str
		SRS file name
	'''
	if start_time=='':
		print ('Please provide start time.\n')
		return
	if end_time=='':
		print ('Please provide end time.\n')
		return
	start_time=pd.to_datetime(start_time)	
	end_time=pd.to_datetime(end_time)	
	datestamp=start_time.date()
	year_stamp=str(datestamp.year)[2:]
	month=datestamp.month
	if month<10:
		month_stamp=str(0)+str(month)
	else:
		month_stamp=str(month)
	day=datestamp.day
	if day<10:
		day_stamp=str(0)+str(day)
	else:
		day_stamp=str(day)
	file_name='LM'+year_stamp+month_stamp+day_stamp+'.srs'
	if os.path.exists(file_name)==False:
		print ('Dpwnloading data....\n')
		download_link='https://downloads.sws.bom.gov.au/wdc/wdc_spec/data/learmonth/raw/'+year_stamp+'/'+file_name
		os.system('wget '+download_link)
	return file_name
	
	
def main():
	usage= 'Download and plot dynamic spectrum from Learmonth Solar Radiograph'
	parser = OptionParser(usage=usage)
	parser.add_option('--starttime',dest="start_time",default=None,help="Start time of the dynamic spectrum",metavar="Datetime String (format : dd-mm-yyyy hh:mm:ss)")
	parser.add_option('--endtime',dest="end_time",default=None,help="End time of the dynamic spectrum",metavar="Datetime String (format : dd-mm-yyyy hh:mm:ss)")
	parser.add_option('--background_subtract',dest="bkg_sub",default=False,help="Perform background subtraction",metavar="Boolean")
	parser.add_option('--flag',dest="flag",default=True,help="Perform flagging",metavar="Boolean")
	parser.add_option('--plot_format',dest="ext",default='pdf',help="Final dynamic spectrum format",metavar="String (pdf,png,jpg,eps)")
	(options, args) = parser.parse_args()	
	
	srs_file=download_learmonth(start_time=options.start_time,end_time=options.end_time)
	print ('Downloaded SRS file : '+srs_file+'\n')
	pd_file=srs_file.split('.srs')[0]
	if os.path.exists(pd_file+'.pd')==False:
		pd_file=srs_to_pd(srs_file,pd_file,bkg_sub=eval(str(options.bkg_sub)),do_flag=eval(str(options.flag)))
	else:
		pd_file=pd_file+'.pd'
	print ('Pandas datafile : '+pd_file+'\n')
	save_file=srs_file.split('.srs')[0]+str(options.ext)
	final_plot=plot_learmonth_DS(pd_file,save_file=save_file,start_time=options.start_time,end_time=options.end_time)
	print ('Dynamic spectrum saved at : '+final_plot+'\n')
	
	
	
if __name__=='__main__':
	main()	
	
	
	
	
	
		


