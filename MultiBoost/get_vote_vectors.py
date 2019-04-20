import xml.etree.ElementTree as ET
import numpy as np

tree = ET.parse('shypSingleStump.xml')
multiboost = tree.getroot()

for t in range(1,31):
    print(multiboost[t].attrib['iter'])
    v = []
    for l in range(1,27):
    #for l in range(1,11):
        print(int(multiboost[t][2][l].text))
        v.append(int(multiboost[t][2][l].text))

    np_v = np.array(v)
    np.save('../data/isolet/single_stump/vote_vectors/v_{}'.format(t), np_v)
    #np.save('../data/pendigits/single_stump/vote_vectors/v_{}'.format(t), np_v)
