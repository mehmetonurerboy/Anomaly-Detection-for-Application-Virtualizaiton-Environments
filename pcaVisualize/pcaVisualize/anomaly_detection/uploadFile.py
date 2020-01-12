import requests
import json

GITHUB_API="https://api.github.com"
API_TOKEN='ea7d69f23a5d3fe5e35d3f774950ac5c1c488654'

#form a request URL
url=GITHUB_API+"/gists"
gist_data_link = ""

def upload():

	file = open("general_anomaly.csv")
	file_content = file.read()
	file.close()

	headers={'Authorization':'token %s'%API_TOKEN}
	params={'scope':'gist'}
	payload={"description":"GIST created by python code","public":False,"files":{"clusterInfo":{"content":file_content}}}

	res=requests.post(url,headers=headers,params=params,data=json.dumps(payload))

	#print response --> JSON
	# print(res.status_code)
	# print(res.url)
	# print(res.text)
	j=json.loads(res.text)
	return j['files']['clusterInfo']['raw_url']
