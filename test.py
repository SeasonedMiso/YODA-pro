import edgeDetect
import edgeSeq
import medianFilt
import medianFiltSeq
import time
import os
import sys


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore


def enablePrint():
    sys.stdout = sys.__stdout__

print("------------------")
blockPrint()
for i in range(1, 10):
    medianFiltSeq.medFiltSeq()
enablePrint()
start = time.time()
medianFiltSeq.medFiltSeq()
end = time.time()
print("sequential median filter time:")
print(end - start)

print("------------------")
blockPrint()
for i in range(1, 10):
    medianFilt.medFiltPara()
enablePrint()
start = time.time()
medianFilt.medFiltPara()
end = time.time()
print("parallel median filter time:")
print(end - start)

print("------------------")
blockPrint()
for i in range(1, 3):
    edgeSeq.edgeDetectSeq()
enablePrint()
start = time.time()
edgeSeq.edgeDetectSeq()
end = time.time()
print("sequential edge detect time:")
print(end - start)

print("------------------")
blockPrint()
for i in range(1, 10):
    edgeDetect.edgeDetectPara()
enablePrint()
start = time.time()
edgeDetect.edgeDetectPara()
end = time.time()
print("parallel edge detect time:")
print(end - start)
