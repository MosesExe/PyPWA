"""
.. module:: batchFarmServices
   :platform: Unix, Windows, OSX
   :synopsis: Utilities for doing PWA with the Jlab batch system.

.. moduleauthor:: Joshua Pond <jpond@jlab.org>


""" 
import numpy
import os
import sys
sys.path.append(os.path.join(sys.argv[2],"pythonPWA"))
from pythonPWA.dataTypes.resonance import resonance
from pythonPWA.fileHandlers.getWavesGen import getwaves
from pythonPWA.model.normInt import normInt
from pythonPWA.model.intensity import intensity
from pythonPWA.fileHandlers.gampReader import gampReader 
from batchFarmServices.dataSimulatorNPY import dataSimulator
from pythonPWA.model.complexV import complexV
import operator

from batchFarmServices.rhoAA import rhoAA

Control = numpy.load(os.path.join(sys.argv[2],"GUI","Control_List.npy"))

dataDir=os.path.join(sys.argv[2],"simulation",sys.argv[1]+"_MeV")
topDir=os.path.join(sys.argv[2])

alphaList=numpy.loadtxt(os.path.join(dataDir,"mc","raw","alphaevents.txt"))
              
maxNumberOfEvents=float(len(alphaList))   

testMass= int(sys.argv[1])+(int(Control[4])/2.)

productionAmplitudes=[]

waves=getwaves(os.path.join(dataDir,"mc","raw"))

normint=numpy.load(os.path.join(dataDir,"mc","raw","normint.npy"))

if not os.path.isfile(os.path.join(dataDir,"Vvalues.npy")):
    contents=numpy.load(os.path.join(dataDir,"Vvalues.npy"))    
    orderedContents=sorted(contents.tolist().iteritems(),key=operator.itemgetter(0))     
    for i in range(0,len(orderedContents),2):
        realPart=orderedContents[i][1]
        imaginaryPart=orderedContents[i+1][1]
        productionAmplitudes.append(numpy.complex(realPart,imaginaryPart))  
          
elif os.path.isfile(os.path.join(topDir,"scripts","resonances.txt")):
    resonances=[]
    res = open("resonances.txt")
    rez = res.readlines()
    for re in rez:
        if re[0] != "#" and re[0] != " " and re[0] != "\n":
            rev = re.split(" ")
            wRx = [(float(x)) for x in rev[1].split(",")]
            resonances.append(resonance(cR=float(rev[0]),wR=wRx,w0=float(rev[2]),r0=float(rev[3])))        
    for resonance in resonances:
        print resonance.toString()                
        for wave in waves:
            productionAmplitudes.append(complexV(resonance,wave,waves,normint,testMass))
    if len(productionAmplitudes) == 0:
        print "There are no resonances in resonances.txt, modify it in /scripts and try again."
        exit()
else:
    print "There is neither a resonance.txt file, or a Vvalues.npy file consult the documentation and try again."
    exit()
    
if sys.argv[3] == "i":
    if os.path.isfile(os.path.join(dataDir,"mc","raw","rhoAA.npy")):
        rhoAA = numpy.load(os.path.join(dataDir,"mc","raw","rhoAA.npy"))
    if not os.path.isfile(os.path.join(dataDir,"mc","raw","rhoAA.npy")):
        rAA = rhoAA(waves=waves,alphaList=alphaList,beamPolarization=float(Control[1]))
        rhoAA = rAA.calc()  
        numpy.save(os.path.join(dataDir,"mc","raw","rhoAA.npy"),rhoAA)

    dSimulator=dataSimulator(mass=testMass,waves=waves,productionAmplitudes=productionAmplitudes,normint=normint,alphaList=alphaList,rhoAA=rhoAA)
    iList = dSimulator.calcIList()
    numpy.save(os.path.join(dataDir,"mc","raw","iList"),iList)

elif sys.argv[3] == "s":
    inputGampFile=open(os.path.join(dataDir,"mc","raw","events.gamp"),'r')
    inputPfFile=open(os.path.join(dataDir,"mc","raw","events.pf"),'r')
    outputRawGampFile=open(os.path.join(dataDir,"data","selected_events.raw.gamp"),'w')
    outputAccGampFile=open(os.path.join(dataDir,"data","selected_events.acc.gamp"),'w')
                    
    iList = numpy.load(os.path.join(dataDir,"mc","raw","iList.npy"))

    iMax = numpy.load(os.path.join(dataDir,"mc","raw","iMax.npy"))

    dSimulator=dataSimulator(mass=testMass,waves=waves,productionAmplitudes=productionAmplitudes,normint=normint,alphaList=alphaList,rhoAA=rhoAA,iList=iList,iMax=iMax)

    dSimulator.execute(inputGampFile,outputRawGampFile,outputAccGampFile,inputPfFile)
                
else:
    print "The last argument must be either i to calculate iList, or s to do the simulation."
    exit()
