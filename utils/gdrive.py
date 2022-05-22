# Tiếng Việt:
# Lấy xác thực google để upload/download file
# Vui lòng bấm vào link khi được yêu cầu và lấy mã để nhập vào

# English:
# Google drive download/upload made easy
# Please click the url when prompted and paste the link as instructed to get google credential

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
from pathlib import Path

class GDrive():
    def __init__(self, 
                 client_config_file: str='client_secrets.json', 
                 auth_type: str='cli'):
        self._gauth = GoogleAuth()
        
        # Enforce config
        # to ensure that the credential automatically get refreshed 
        # when run remotely
        self._gauth.settings['client_config_file'] = client_config_file
        self._gauth.settings['get_refresh_token'] = True
        self._gauth.settings['save_credentials'] = True
        self._gauth.settings['save_credentials_backend'] = 'file'
        self._gauth.settings['save_credentials_file'] = 'credentials.json'

        # Get credential
        # suggest using cli for most of the case
        if auth_type == 'cli':
            self._gauth.CommandLineAuth()
        elif auth_type == 'localhost':
            self._gauth.LocalWebserverAuth()
        else:
            raise NotImplementedError()

        self._drive = GoogleDrive(self._gauth)

    def CreateFile(self, file_name=None, parent_id=None):
        file = self._drive.CreateFile({'title': file_name, 
                                       'parents': [{'id': parent_id}]})
        return file

    def CreateFolder(self, folder_name=None, parent_id=None):
        folder = self._drive.CreateFile({'title': folder_name, 
                                       'parents': [{'id': parent_id}],
                                       'mimeType': 'application/vnd.google-apps.folder'})
        folder.Upload()
        return folder

    def UploadFile(self, file_path, parent_id, file_name=None):
        file_path = Path(file_path)
        if file_name == None:
          file_name = file_path.name
        file = self.GetSingleFile(file_name, parent_id, True)
        file.SetContentFile(str(file_path))
        file.Upload()

    def DownloadFile(self, file_name, parent_id):
        file = self.GetSingleFile(file_name, parent_id)    
        file.GetContentFile(file_name)

    def UploadFolder(self, folder_path, parent_id, folder_name=None, child_only=False):
        folder_path = Path(folder_path)
        id_map = {}
        if not child_only:
            folder = self.CreateFolder(folder_path.name, parent_id)
            id_map[str(folder_path)] = folder["id"]
        else:
            id_map[str(folder_path)] = parent_id

        for f in folder_path.rglob("*"):
            if f.is_dir():
                folder = self.CreateFolder(f.name,  id_map[str(f.parent)])
                id_map[str(f)] = folder["id"]
            else:
                self.UploadFile(f, id_map[str(f.parent)])


#########################################################################################
    def SearchInFolder(self, parent_id, file_name):
        return self._drive.ListFile({'q': f"'{parent_id}' in parents and title = '{file_name}' and trashed=false"}).GetList()

    def GetSingleFile(self, file_name, parent_id, auto_create=False, is_folder=False):
        # Kiểm tra file tồn tại
        file_list = self.SearchInFolder(parent_id, file_name)
        if len(file_list) > 1:
            for file in file_list:
                print('title: %s, id: %s' % (file['title'], file['id']))
            raise NameError('More than 1 file with same name exist, please resolve this')
        elif len(file_list) == 0:
            if auto_create:
                # File chưa có thì tạo mới
                if is_folder:
                    return self.CreateFolder(file_name, parent_id)
                else:
                    return self.CreateFile(file_name, parent_id)
            else:
                raise NameError(f'File named {file_name} not exist')
        # Tồn tại duy nhất 1 file
        return file_list[0]