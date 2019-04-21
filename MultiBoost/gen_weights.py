import os

for i in range(31):
    #os.system('./multiboost --fileformat arff --traintest pendigitsTrain.arff pendigitsTest.arff {} --verbose 5 --learnertype SingleStumpLearner --outputinfo resultSingleStump.dta ham --shypname shypSingleStump.xml --weights weight.txt'.format(i))
    #os.system('./multiboost --fileformat arff --traintest isoletTrain.arff isoletTest.arff {} --verbose 5 --learnertype SingleStumpLearner --outputinfo resultSingleStump.dta ham --shypname shypSingleStump.xml --weights weight.txt'.format(i))
    os.system('./multiboost --fileformat arff --traintest good_data/pendigitsTrainbinary.arff good_data/pendigitsTestbinary.arff {} --verbose 5 --learnertype SingleStumpLearner --outputinfo resultSingleStump.dta edg --shypname shypSingleStump.xml --weights weight.txt'.format(i))


    #os.system('mv weight.txt ../data/pendigits/single_stump/weights/weight_{}.txt'.format(i))
    #os.system('mv weight.txt ../data/isolet/single_stump/weights/weight_{}.txt'.format(i))
    os.system('mv weight.txt ../data/pendigits_binary/single_stump/weights/weight_{}.txt'.format(i))
