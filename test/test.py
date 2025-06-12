from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
for f in file_list:
    print(f"title: {f['title']}, id: {f['id']}")

file2 = drive.CreateFile({'title': 'sim.mp4'})
file2.SetContentFile('/home/crae/projects/ml-accelerated-simulation/sim.mp4')
file2.Upload()