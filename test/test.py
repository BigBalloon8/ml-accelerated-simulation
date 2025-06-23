from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

def get_drive():
    gauth = GoogleAuth(settings_file="settings.yaml")
    gauth.CommandLineAuth()  # prompts only first run
    if gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("credentials.json")
    return GoogleDrive(gauth)

drive = get_drive()

about = drive.GetAbout()

print('Current user name:{}'.format(about['name']))
print('Root folder ID:{}'.format(about['rootFolderId']))
print('Total quota (bytes):{}'.format(about['quotaBytesTotal']))
print('Used quota (bytes):{}'.format(about['quotaBytesUsed']))

file_list = drive.ListFile().GetList()
for file1 in file_list:
  print('title: %s, id: %s' % (file1['title'], file1['id']))

# file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
# for f in file_list:
#     print(f"title: {f['title']}, id: {f['id']}")

# file2 = drive.CreateFile({'title': 'sim.mp4', 'parents': [{'id': "1YqPwoWjmd4WDad_lAXDLX8Z6wZ2Bd8O_"}]})
# file2.SetContentFile('/home/crae/projects/ml-accelerated-simulation/sim.mp4')
# file2.Upload()