import subprocess as sp
import os, time

class farmCheck (object):

    def __init__(self,cmd):
        self.cmd = cmd

    def check(self):
        jobs = 1
        while jobs > 0:
            time.sleep(10)
            out =sp.check_output(self.cmd,shell = True,executable = os.environ.get('SHELL', '/bin/tcsh'),env = os.environ)            
            jobs = len(out.split("\n"))-2
        return True

