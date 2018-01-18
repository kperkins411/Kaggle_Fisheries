import os, shutil
import zipfile
import logging

class utils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("utils object created")

    def makeDir(self,dir):
        try:
            root = os.path.abspath(os.path.dirname(__file__))
            dir = os.path.join(root, dir)
            os.mkdir(dir)
        except OSError as exc:
            print("error making "+ dir + "-"+exc.strerror)
            self.clear_folder_files(dir)

    def clear_folder_files(self,dir):
        root = os.path.abspath(os.path.dirname(__file__))
        dir = os.path.join(root, dir)

        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def unzip_to_dir(self,zippedfile,dest_directory):
        '''
        :param zippedfile: fully qualified filename to unzip
        :param dest_directory: where it goes
        :return:
        '''
        ret = True
        try:
             #unzip into directory(make it if necessary)
            tmpZip = zipfile.ZipFile(zippedfile)
            tmpZip.extractall(dest_directory)
            self.logger.info("Extracting files to:"+dest_directory)
            return True
        except Exception as err:
            self.logger.debug("Error Unzipping file:" + zippedfile + " info:",err)
            return False