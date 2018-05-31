import sys

def vprint(*args,**kwargs):
    if(args[0] <= args[1]):
        if sys.version_info >= (3,0):
            #to make compatible with py2..
            eval('print(*args[2:],**kwargs)')
        else:
            for arg in args[2:]:
                print(arg)
    else:
        pass
