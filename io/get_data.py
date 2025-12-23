import json
import act

startdate = '2025-12-01'
enddate = '2025-12-31'

#Read in ARM Live Data Webservice Token and Username
with open('./token.json') as f:
    data = json.load(f)
username = data['username']
token = data['token']

#Specify datastream and date range for KAZR data

sdate = ''.join(startdate.split('-'))
edate = ''.join(enddate.split('-'))

datastreams = ['sgpminimplC1.b1', 'sgpmplpolfsC1.b1', 'sgpceil10mC1.b1', 'sgpceilpolC1.b1',
               'sgpdlfptC1.b1', 'sgpdlfptE13.b1', 'sgprlC1.a0', 'sgphsrlC1.a1']

for ds in datastreams:
    data_dir = './data/' + ds
    result = act.discovery.download_arm_data(username, token, ds, startdate, enddate, output=data_dir)
