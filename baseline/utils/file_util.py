import os

def write_file(filepath, strTmp):
    output = open(filepath, 'w')
    output.write(strTmp.strip())
    output.close()