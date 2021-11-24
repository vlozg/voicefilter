# Tiếng Việt:
# Lấy xác thực google để upload/download file
# Vui lòng bấm vào link khi được yêu cầu và lấy mã để nhập vào

# English:
# Google drive download/upload made easy
# Please click the url when prompted and paste the link as instructed to get google credential

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.client import GoogleCredentials

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

    def SearchInFolder(self, parent_id, file_name):
        return self._drive.ListFile({'q': f"'{parent_id}' in parents and title = '{file_name}'"}).GetList()

    def CreateFile(self, file_name=None, parent_id=None):
        file = self._drive.CreateFile({'title': file_name, 
                                       'parents': [{'id': parent_id}]})
        return file

    def Upload(self, file_path, parent_id, file_name=None):
        if file_name == None:
          file_name = file_path.split('/')[-1]
        # Kiểm tra file tồn tại
        file_list = self.SearchInFolder(parent_id, file_name)
        if len(file_list) > 1:
          for file in file_list:
            print('title: %s, id: %s' % (file['title'], file['id']))
          raise NameError('More than 1 file with same name exist, please resolve this')
        
        elif len(file_list) == 0:
          # File chưa có thì tạo mới
          file = self.CreateFile(file_name, parent_id)
        else:
          # Tồn tại duy nhất 1 file
          file = file_list[0]
        
        file.SetContentFile(file_path)
        file.Upload()

    def Download(self, file_name, parent_id):
        # Kiểm tra file tồn tại
        file_list = self.SearchInFolder(parent_id, file_name)
        if len(file_list) > 1:
            for file in file_list:
                print('title: %s, id: %s' % (file['title'], file['id']))
            raise NameError('More than 1 file with same name exist, please resolve this')
        elif len(file_list) == 0:
            raise NameError(f'File named {file_name} not exist')
        else:
            # Tồn tại duy nhất 1 file
            file = file_list[0]
        
        file.GetContentFile(file_name)