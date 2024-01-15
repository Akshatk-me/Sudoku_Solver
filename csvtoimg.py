import numpy

def readcsv(fname: str) -> numpy.ndarray:

    with open(fname, 'rb') as csvfile:
        read = csvfile.read()

    (h, w, channels) = (28, 28, 1)
    data = numpy.frombuffer(read, dtype=int)
    data = data.reshape((h, w, channels))
    
    print(data[0])

# for data[i][0] is the number which is represented by the pixels

    return numpy.ndarray(1)










