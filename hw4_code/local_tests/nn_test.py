import numpy as np

from NN import *

class NN_Test():
  def __init__(self):

    # Sample data: 
    self.x_train = np.array([[ 0.03807591,  0.05068012, -0.01375064, -0.01599922, -0.03596778,
                  -0.02198168, -0.01394774, -0.00259226, -0.02595242, -0.0010777 ],
                  [ 0.01991321,  0.05068012,  0.10480869,  0.07007254, -0.03596778,
                  -0.0266789 , -0.02499266, -0.00259226,  0.00371174,  0.04034337],
                  [ 0.05260606, -0.04464164, -0.00405033, -0.03091833, -0.0469754 ,
                  -0.0583069 , -0.01394774, -0.02583997,  0.03605579,  0.02377494],
                  [ 0.04897352,  0.05068012,  0.08109682,  0.02187235,  0.04383748,
                    0.06413415, -0.05444576,  0.07120998,  0.03243323,  0.04862759],
                  [ 0.0562386 ,  0.05068012,  0.02181716,  0.05630106, -0.00707277,
                    0.01810133, -0.03235593, -0.00259226, -0.02364456,  0.02377494],
                  [ 0.03444337, -0.04464164,  0.01858372,  0.05630106,  0.01219057,
                  -0.05454912, -0.06917231,  0.07120998,  0.13008061,  0.00720652],
                  [ 0.04170844,  0.05068012,  0.07139652,  0.00810087,  0.03833367,
                    0.01590929, -0.01762938,  0.03430886,  0.07341008,  0.08590655],
                  [ 0.01264814,  0.05068012, -0.07195249, -0.04698506, -0.05110326,
                  -0.09713731,  0.11859122, -0.0763945 , -0.02028875, -0.03835666],
                  [-0.00551455, -0.04464164,  0.00888341, -0.05042793,  0.0259501 ,
                    0.04722413, -0.04340085,  0.07120998,  0.01482271,  0.00306441]]).T


    self.y_train = np.array([[1., 2., 3., 4., 5., 6., 7., 8., 9.]])

    self.yh = np.array([[2.16402836, 3.75766397, 4.68946889, 5.81078464, 5.91976053,
              7.09269363, 8.59720718, 8.5714377 , 9.98050705]])

    # Sample outputs for misc helper functions:
    self.leaky_relu = np.array([[ 3.8075910e-02,  1.9913210e-02,  5.2606060e-02,  4.8973520e-02,
                    5.6238600e-02,  3.4443370e-02,  4.1708440e-02,  1.2648140e-02,
                  -2.7572750e-04],
                  [ 5.0680120e-02,  5.0680120e-02, -2.2320820e-03,  5.0680120e-02,
                    5.0680120e-02, -2.2320820e-03,  5.0680120e-02,  5.0680120e-02,
                  -2.2320820e-03],
                  [-6.8753200e-04,  1.0480869e-01, -2.0251650e-04,  8.1096820e-02,
                    2.1817160e-02,  1.8583720e-02,  7.1396520e-02, -3.5976245e-03,
                    8.8834100e-03],
                  [-7.9996100e-04,  7.0072540e-02, -1.5459165e-03,  2.1872350e-02,
                    5.6301060e-02,  5.6301060e-02,  8.1008700e-03, -2.3492530e-03,
                  -2.5213965e-03],
                  [-1.7983890e-03, -1.7983890e-03, -2.3487700e-03,  4.3837480e-02,
                  -3.5363850e-04,  1.2190570e-02,  3.8333670e-02, -2.5551630e-03,
                    2.5950100e-02],
                  [-1.0990840e-03, -1.3339450e-03, -2.9153450e-03,  6.4134150e-02,
                    1.8101330e-02, -2.7274560e-03,  1.5909290e-02, -4.8568655e-03,
                    4.7224130e-02],
                  [-6.9738700e-04, -1.2496330e-03, -6.9738700e-04, -2.7222880e-03,
                  -1.6177965e-03, -3.4586155e-03, -8.8146900e-04,  1.1859122e-01,
                  -2.1700425e-03],
                  [-1.2961300e-04, -1.2961300e-04, -1.2919985e-03,  7.1209980e-02,
                  -1.2961300e-04,  7.1209980e-02,  3.4308860e-02, -3.8197250e-03,
                    7.1209980e-02],
                  [-1.2976210e-03,  3.7117400e-03,  3.6055790e-02,  3.2433230e-02,
                  -1.1822280e-03,  1.3008061e-01,  7.3410080e-02, -1.0144375e-03,
                    1.4822710e-02],
                  [-5.3885000e-05,  4.0343370e-02,  2.3774940e-02,  4.8627590e-02,
                    2.3774940e-02,  7.2065200e-03,  8.5906550e-02, -1.9178330e-03,
                    3.0644100e-03]])

    self.tanh = np.array([[ 0.03805752,  0.01991058,  0.05255759,  0.0489344 ,  0.05617938,
                  0.03442976,  0.04168427,  0.01264747, -0.00551449],
                [ 0.05063677,  0.05063677, -0.04461201,  0.05063677,  0.05063677,
                -0.04461201,  0.05063677,  0.05063677, -0.04461201],
                [-0.01374977,  0.1044266 , -0.00405031,  0.0809195 ,  0.0218137 ,
                  0.01858158,  0.07127545, -0.07182858,  0.00888318],
                [-0.01599786,  0.06995808, -0.03090848,  0.02186886,  0.05624165,
                  0.05624165,  0.00810069, -0.04695052, -0.05038523],
                [-0.03595228, -0.03595228, -0.04694088,  0.04380942, -0.00707265,
                  0.01218997,  0.0383149 , -0.05105882,  0.02594428],
                [-0.02197814, -0.02667257, -0.05824091,  0.06404636,  0.01809935,
                -0.05449508,  0.01590795, -0.09683294,  0.04718906],
                [-0.01394684, -0.02498746, -0.01394684, -0.05439203, -0.03234464,
                -0.0690622 , -0.01762755,  0.11803838, -0.04337362],
                [-0.00259225, -0.00259225, -0.02583422,  0.07108986, -0.00259225,
                  0.07108986,  0.0342954 , -0.07624623,  0.07108986],
                [-0.0259466 ,  0.00371172,  0.03604017,  0.03242186, -0.02364015,
                  0.12935185,  0.07327849, -0.02028597,  0.01482162],
                [-0.0010777 ,  0.0403215 ,  0.02377046,  0.0485893 ,  0.02377046,
                  0.0072064 ,  0.08569584, -0.03833786,  0.0030644 ]])

    self.nloss = 0.9142519719206229



    # Output for forward function
    self.forward_u1 = np.array([[ 0.01402937, -0.04725561,  0.07669296, -0.0880401 , -0.04026252, 0.00440673, -0.02149594,  0.1661655 , -0.04567273],
                  [-0.0163818 , -0.03705478,  0.07380859, -0.0352923 , -0.01835815,
                    0.04668763, -0.00553572,  0.0087384 ,  0.00799142],
                  [-0.00353575,  0.05614368, -0.03219106,  0.01297088,  0.01602512,
                    -0.02683109,  0.03167426,  0.00569138, -0.04266634],
                  [-0.00883828, -0.02442314,  0.04095987,  0.00284903, -0.03261088,
                    0.07629807,  0.02610006, -0.03690876,  0.04176993],
                  [-0.02175949,  0.00688673, -0.00725445,  0.0178652 ,  0.00364731,
                    0.09683463,  0.00940061, -0.05622873,  0.02051729],
                  [-0.00410651, -0.04770276,  0.00573411, -0.00064907, -0.01541225,
                    0.00076352,  0.00814272,  0.00849936,  0.02550554],
                  [-0.01845564, -0.00463578, -0.0368322 ,  0.0404096 , -0.03629645,
                    0.07118243,  0.04946754,  0.01288536,  0.020288  ],
                  [-0.02161116,  0.03274394,  0.00986106, -0.01892744, -0.02288224,
                    0.08057598,  0.00911098, -0.01570855, -0.00682689],
                  [-0.01059912,  0.0269362 , -0.00049896,  0.04478966,  0.01578753,
                    0.03827847,  0.0434621 , -0.07005823,  0.01998369],
                  [ 0.02324345, -0.01068456,  0.01567869,  0.02620895,  0.02318995,
                    -0.01824398,  0.02115369, -0.00186478,  0.01252829],
                  [ 0.02044841,  0.04939207, -0.01616666, -0.02104525,  0.02388324,
                    -0.06491152, -0.00783262,  0.06445125, -0.06415026],
                  [-0.01352575,  0.02065031,  0.02825134, -0.04014358, -0.04071984,
                    0.00577602,  0.02131676,  0.05314985, -0.03289264],
                  [-0.01357879,  0.0803966 , -0.02264742,  0.01334491,  0.04070769,
                    -0.0506604 , -0.00504726, -0.02858073, -0.03310328],
                  [-0.02193155, -0.04478489,  0.01797578, -0.06766471, -0.02695712,
                    0.01887477, -0.02739064,  0.05720527, -0.03530926],
                  [-0.02142205,  0.0432955 , -0.01813506,  0.05855105,  0.01610038,
                    0.04087837,  0.05477399, -0.06787374,  0.01324538]])
    self.forward_o1 = np.array([[ 1.40293725e-02, -2.36278030e-03,  7.66929580e-02,
                                  -4.40200520e-03, -2.01312595e-03,  4.40673298e-03,
                                  -1.07479701e-03,  1.66165497e-01, -2.28363633e-03],
                                [-8.19090233e-04, -1.85273883e-03,  7.38085902e-02,
                                  -1.76461500e-03, -9.17907564e-04,  4.66876268e-02,
                                  -2.76786002e-04,  8.73839893e-03,  7.99141527e-03],
                                [-1.76787635e-04,  5.61436775e-02, -1.60955310e-03,
                                  1.29708797e-02,  1.60251222e-02, -1.34155444e-03,
                                  3.16742636e-02,  5.69137911e-03, -2.13331691e-03],
                                [-4.41914192e-04, -1.22115717e-03,  4.09598716e-02,
                                  2.84902518e-03, -1.63054378e-03,  7.62980702e-02,
                                  2.61000573e-02, -1.84543816e-03,  4.17699346e-02],
                                [-1.08797427e-03,  6.88673452e-03, -3.62722445e-04,
                                  1.78652032e-02,  3.64730674e-03,  9.68346285e-02,
                                  9.40061467e-03, -2.81143640e-03,  2.05172943e-02],
                                [-2.05325430e-04, -2.38513780e-03,  5.73411089e-03,
                                  -3.24536854e-05, -7.70612421e-04,  7.63521354e-04,
                                  8.14272249e-03,  8.49936353e-03,  2.55055414e-02],
                                [-9.22782026e-04, -2.31788760e-04, -1.84160995e-03,
                                  4.04095986e-02, -1.81482227e-03,  7.11824273e-02,
                                  4.94675354e-02,  1.28853577e-02,  2.02880015e-02],
                                [-1.08055793e-03,  3.27439366e-02,  9.86105721e-03,
                                  -9.46372003e-04, -1.14411221e-03,  8.05759846e-02,
                                  9.11097531e-03, -7.85427583e-04, -3.41344301e-04],
                                [-5.29956104e-04,  2.69361964e-02, -2.49481383e-05,
                                  4.47896588e-02,  1.57875328e-02,  3.82784725e-02,
                                  4.34620987e-02, -3.50291137e-03,  1.99836905e-02],
                                [ 2.32434490e-02, -5.34228017e-04,  1.56786924e-02,
                                  2.62089520e-02,  2.31899465e-02, -9.12198841e-04,
                                  2.11536923e-02, -9.32392309e-05,  1.25282922e-02],
                                [ 2.04484142e-02,  4.93920702e-02, -8.08333216e-04,
                                  -1.05226232e-03,  2.38832403e-02, -3.24557621e-03,
                                  -3.91631032e-04,  6.44512481e-02, -3.20751283e-03],
                                [-6.76287336e-04,  2.06503137e-02,  2.82513442e-02,
                                  -2.00717925e-03, -2.03599189e-03,  5.77602179e-03,
                                  2.13167613e-02,  5.31498450e-02, -1.64463193e-03],
                                [-6.78939626e-04,  8.03965958e-02, -1.13237106e-03,
                                  1.33449088e-02,  4.07076888e-02, -2.53301989e-03,
                                  -2.52362979e-04, -1.42903652e-03, -1.65516393e-03],
                                [-1.09657768e-03, -2.23924454e-03,  1.79757840e-02,
                                  -3.38323532e-03, -1.34785601e-03,  1.88747740e-02,
                                  -1.36953207e-03,  5.72052749e-02, -1.76546298e-03],
                                [-1.07110231e-03,  4.32954987e-02, -9.06752872e-04,
                                  5.85510475e-02,  1.61003831e-02,  4.08783700e-02,
                                  5.47739852e-02, -3.39368691e-03,  1.32453818e-02]])
    self.forward_u2 = np.array([[-0.01034704,  0.01480894, -0.04354555,  0.00906054,  0.00011559,
                    0.02205182,  0.01805583, -0.04646774,  0.0181819 ]])
    self.forward_o2 = np.array([[-4.96619957e-03,  2.07992385e-02, -6.16670093e-02,
                                  1.49762634e-02,  7.37984630e-05, -2.82631077e-02,
                                  2.23046983e-02, -1.55731087e-02,  3.31241534e-02]])
    self.forward_o2_without_dropout = np.array([[-0.01034667,  0.01480785, -0.04351804,  0.00906029,  0.00011559, 0.02204824,  0.01805387, -0.04643432,  0.0181799 ]])

    # Outputs for dropout
    self.u1_after_dropout = np.array([[ 0.02004196, -0.06750801,  0.        , -0.12577157, -0.        ,
                                          0.        , -0.        ,  0.23737929, -0.06524676],
                                        [-0.02340257, -0.0529354 ,  0.10544084, -0.        , -0.02622593,
                                          0.        , -0.00790817,  0.01248343,  0.01141631],
                                        [-0.        ,  0.        , -0.04598723,  0.01852983,  0.02289303,
                                        -0.03833013,  0.04524894,  0.00813054, -0.        ],
                                        [-0.        , -0.        ,  0.0585141 ,  0.        , -0.04658697,
                                          0.10899724,  0.0372858 , -0.0527268 ,  0.05967133],
                                        [-0.03108499,  0.00983819, -0.        ,  0.02552171,  0.00521044,
                                          0.13833519,  0.        , -0.08032676,  0.        ],
                                        [-0.00586644, -0.0681468 ,  0.        , -0.        , -0.        ,
                                          0.        ,  0.01163246,  0.        ,  0.        ],
                                        [-0.0263652 , -0.        , -0.05261743,  0.        , -0.05185207,
                                          0.10168919,  0.        ,  0.01840766,  0.02898286],
                                        [-0.03087309,  0.        ,  0.01408723, -0.0270392 , -0.03268891,
                                          0.11510854,  0.01301569, -0.02244079, -0.        ],
                                        [-0.        ,  0.03848029, -0.0007128 ,  0.        ,  0.02255361,
                                          0.05468353,  0.06208871, -0.10008319,  0.02854813],
                                        [ 0.03320493, -0.01526366,  0.02239813,  0.        ,  0.0331285 ,
                                        -0.02606283,  0.03021956, -0.00266397,  0.01789756],
                                        [ 0.        ,  0.0705601 , -0.02309523, -0.03006464,  0.03411891,
                                        -0.        , -0.01118946,  0.09207321, -0.        ],
                                        [-0.0193225 ,  0.02950044,  0.04035906, -0.05734797, -0.0581712 ,
                                          0.00825146,  0.03045251,  0.        , -0.04698949],
                                        [-0.01939827,  0.11485229, -0.        ,  0.        ,  0.05815384,
                                        -0.072372  , -0.        , -0.04082961, -0.0472904 ],
                                        [-0.03133079, -0.06397841,  0.        , -0.        , -0.        ,
                                          0.        , -0.        ,  0.08172181, -0.0504418 ],
                                        [-0.03060293,  0.06185071, -0.        ,  0.        ,  0.02300054,
                                          0.05839767,  0.07824856, -0.        ,  0.01892197]])

    self.dropout_mask = np.array([[1, 1, 0, 1, 0, 0, 0, 1, 1],
                                    [1, 1, 1, 0, 1, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1, 1, 1, 1, 0],
                                    [0, 0, 1, 0, 1, 1, 1, 1, 1],
                                    [1, 1, 0, 1, 1, 1, 0, 1, 0],
                                    [1, 1, 0, 0, 0, 0, 1, 0, 0],
                                    [1, 0, 1, 0, 1, 1, 0, 1, 1],
                                    [1, 0, 1, 1, 1, 1, 1, 1, 0],
                                    [0, 1, 1, 0, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 0, 1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1, 0, 1, 1, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 0, 1],
                                    [1, 1, 0, 0, 1, 1, 0, 1, 1],
                                    [1, 1, 0, 0, 0, 0, 0, 1, 1],
                                    [1, 1, 0, 0, 1, 1, 1, 0, 1]])


    # Outputs for backward function
    self.dLoss_theta2 = np.array([[-0.03402052, -0.08920497, -0.04144692, -0.10581599, -0.14459094,
                                    -0.05745014, -0.16694243, -0.08941738, -0.10131233, -0.06021393,
                                    -0.01388765, -0.0319441 , -0.03321787, -0.09361998, -0.16360135]])
    self.dLoss_theta2_without_dropout = np.array([[-0.17153379, -0.07003818, -0.05300721, -0.12521658, -0.10125687,
                                                    -0.04073596, -0.13375642, -0.06933973, -0.11055812, -0.06046782,
                                                    -0.07727722, -0.07802206, -0.04092946, -0.06389679, -0.12395528]])
    self.dLoss_b2 = np.array([[-4.99841158]])
    self.dLoss_b2_without_dropout = np.array([[-4.99856367]])
    self.dLoss_theta1 = np.array([[ 1.47862444e-02,  1.97064777e-03,  1.65583715e-02,
                                      4.33188536e-03,  5.87098546e-03,  3.79623956e-03,
                                    -1.79595725e-02,  1.89406736e-02,  1.94524956e-02,
                                      1.44109352e-02],
                                    [ 1.27204264e-01,  8.58961223e-02,  4.38359376e-02,
                                      2.76781933e-02, -1.08647004e-02, -8.48153767e-02,
                                    -2.44631652e-03,  2.44824171e-02,  1.22522572e-01,
                                      7.17690681e-02],
                                    [-3.46304574e-02, -1.46409337e-02, -1.02444335e-02,
                                    -8.99855472e-05,  3.05129973e-03,  1.75773202e-02,
                                      1.26537462e-02, -3.05509853e-02, -3.17599613e-02,
                                    -6.59155604e-03],
                                    [-9.47635368e-02, -7.44007415e-02, -5.50571144e-02,
                                      3.35952732e-02,  1.88461386e-03,  1.35488863e-02,
                                      4.12716621e-03, -4.02545351e-02, -5.23794984e-02,
                                    -7.01864382e-02],
                                    [-2.28775226e-02, -1.01511243e-02, -1.31361619e-02,
                                      6.45605803e-04, -9.47908909e-04,  8.67788771e-03,
                                      8.36975016e-03, -1.58828000e-02, -2.47441793e-02,
                                    -1.49498515e-02],
                                    [ 4.33980451e-03,  3.40726702e-03,  2.52140350e-03,
                                    -1.53853395e-03, -8.63080465e-05, -6.20486739e-04,
                                    -1.89008295e-04,  1.84350246e-03,  2.39877901e-03,
                                      3.21426818e-03],
                                    [ 4.40890394e-03,  5.92018515e-04,  1.64415817e-03,
                                    -2.26105505e-03, -4.85881434e-04, -4.61769078e-03,
                                    -4.69763661e-04,  3.39601816e-03,  7.58799415e-03,
                                      2.87350456e-03],
                                    [ 4.99270642e-03,  2.96458815e-03,  7.25541149e-04,
                                      1.75409654e-03, -2.18290954e-03, -4.94605923e-03,
                                      2.39048793e-04,  4.56274431e-06,  3.52860960e-03,
                                      9.75314875e-04],
                                    [-6.66736587e-04, -7.66760922e-05, -1.49256928e-05,
                                      1.64688468e-04,  1.02593664e-04,  3.83803264e-04,
                                      2.07135608e-04, -4.59422412e-04, -6.05483124e-04,
                                    -1.05258436e-04],
                                    [ 2.88212531e-02,  8.99501957e-04,  3.24996185e-02,
                                    -1.78032045e-03,  7.23313342e-03,  2.47579601e-02,
                                    -3.95204210e-02,  3.79207918e-02,  1.20056480e-02,
                                      2.26422909e-02],
                                    [ 1.42550488e-02, -3.44400147e-03,  1.29247779e-02,
                                      5.08636161e-03,  8.66798842e-03,  8.86587770e-03,
                                    -2.53910713e-02,  2.82900446e-02,  1.90991249e-02,
                                      8.10371572e-03],
                                    [ 3.74505010e-02,  1.32263743e-02,  4.01885528e-02,
                                    -5.59500480e-03,  1.87823816e-02,  2.77878450e-02,
                                    -3.86942874e-02,  4.35769445e-02,  3.02229733e-02,
                                      4.12961456e-02],
                                    [-1.24944606e-02, -4.69987899e-03, -5.29063738e-03,
                                      1.77748534e-03, -1.25990889e-03,  4.35357888e-03,
                                      4.21389639e-03, -9.00550019e-03, -1.38735314e-02,
                                    -7.61346401e-03],
                                    [ 1.13606107e-02,  1.30649382e-03,  2.54320804e-04,
                                    -2.80614804e-03, -1.74810667e-03, -6.53967330e-03,
                                    -3.52941032e-03,  7.82815770e-03,  1.03169050e-02,
                                      1.79351206e-03],
                                    [-2.40924519e-02, -8.32099493e-03, -1.08109741e-02,
                                      3.21159716e-03, -3.24132918e-03,  8.21705070e-03,
                                      8.10288613e-03, -1.79774726e-02, -2.81432848e-02,
                                    -1.51758646e-02]])
    self.dLoss_theta1_without_dropout = np.array([[ 1.12541229e-02,  4.98098483e-03,  6.47295785e-03,
                                                    -3.14400853e-04,  4.73065325e-04, -4.25554645e-03,
                                                    -4.13786747e-03,  7.82718228e-03,  1.21789463e-02,
                                                      7.36106210e-03],
                                                    [ 8.83601463e-02,  3.91074943e-02,  5.08215084e-02,
                                                    -2.46847361e-03,  3.71420516e-03, -3.34118180e-02,
                                                    -3.24878782e-02,  6.14540090e-02,  9.56212658e-02,
                                                      5.77943327e-02],
                                                    [-3.76988952e-02, -1.66852296e-02, -2.16830189e-02,
                                                      1.05317535e-03, -1.58466726e-03,  1.42551668e-02,
                                                      1.38609675e-02, -2.62193799e-02, -4.07968551e-02,
                                                    -2.46579774e-02],
                                                    [-7.93653870e-02, -3.51264858e-02, -4.56480535e-02,
                                                      2.21719148e-03, -3.33611183e-03,  3.00106098e-02,
                                                      2.91807239e-02, -5.51982021e-02, -8.58873495e-02,
                                                    -5.19110682e-02],
                                                    [-1.60170946e-02, -7.08903802e-03, -9.21244410e-03,
                                                      4.47461632e-04, -6.73276107e-04,  6.05657949e-03,
                                                      5.88909639e-03, -1.11398036e-02, -1.73333219e-02,
                                                    -1.04764120e-02],
                                                    [ 3.63462864e-03,  1.60865758e-03,  2.09050480e-03,
                                                    -1.01538819e-04,  1.52781056e-04, -1.37437018e-03,
                                                    -1.33636461e-03,  2.52786477e-03,  3.93330937e-03,
                                                      2.37732672e-03],
                                                    [ 4.95157324e-03,  2.19152673e-03,  2.84796293e-03,
                                                    -1.38329647e-04,  2.08138620e-04, -1.87234937e-03,
                                                    -1.82057314e-03,  3.44379270e-03,  5.35847575e-03,
                                                      3.23870980e-03],
                                                    [ 4.32047079e-03,  1.91220583e-03,  2.48497600e-03,
                                                    -1.20698851e-04,  1.81610325e-04, -1.63370920e-03,
                                                    -1.58853210e-03,  3.00486432e-03,  4.67551157e-03,
                                                      2.82592025e-03],
                                                    [-6.31786596e-04, -2.79623697e-04, -3.63380429e-04,
                                                      1.76499090e-05, -2.65570524e-05,  2.38898866e-04,
                                                      2.32292576e-04, -4.39404198e-04, -6.83704550e-04,
                                                    -4.13237034e-04],
                                                    [ 4.07180061e-02,  1.80214640e-02,  2.34195005e-02,
                                                    -1.13751875e-03,  1.71157512e-03, -1.53967899e-02,
                                                    -1.49710212e-02,  2.83191554e-02,  4.40640657e-02,
                                                      2.66327082e-02],
                                                    [ 1.87653332e-02,  8.30538645e-03,  1.07931299e-02,
                                                    -5.24237812e-04,  7.88797894e-04, -7.09577705e-03,
                                                    -6.89955693e-03,  1.30511888e-02,  2.03074009e-02,
                                                      1.22739714e-02],
                                                    [ 3.61820560e-02,  1.60138887e-02,  2.08105887e-02,
                                                    -1.01080016e-03,  1.52090716e-03, -1.36816011e-02,
                                                    -1.33032626e-02,  2.51644264e-02,  3.91553674e-02,
                                                      2.36658479e-02],
                                                    [-9.02890969e-03, -3.99612325e-03, -5.19309700e-03,
                                                      2.52236175e-04, -3.79528830e-04,  3.41412164e-03,
                                                      3.31971065e-03, -6.27955839e-03, -9.77087307e-03,
                                                    -5.90560148e-03],
                                                    [ 1.07650933e-02,  4.76454422e-03,  6.19168597e-03,
                                                    -3.00739075e-04,  4.52509039e-04, -4.07062858e-03,
                                                    -3.95806315e-03,  7.48706477e-03,  1.16497300e-02,
                                                      7.04119909e-03],
                                                    [-1.79711953e-02, -7.95390737e-03, -1.03363710e-02,
                                                      5.02052376e-04, -7.55416430e-04,  6.79548793e-03,
                                                      6.60757173e-03, -1.24988702e-02, -1.94480036e-02,
                                                    -1.17545442e-02]])
    self.dLoss_b1 = np.array([[ 0.46990807],
                                [ 3.39785684],
                                [-1.48848877],
                                [-3.49222859],
                                   [-0.81374004],
                                   [ 0.1599306 ],
                                   [ 0.20132282],
                                   [ 0.14176566],
                                   [-0.02570987],
                                   [ 1.05624521],
                                   [ 0.5291676 ],
                                   [ 1.18378655],
                                   [-0.43853627],
                                   [ 0.4380738 ],
                                   [-0.85246917]])
    self.dLoss_b1_without_dropout = np.array([[ 0.40024402],
                                                [ 3.14245906],
                                                [-1.34073154],
                                                [-2.82256753],
                                                [-0.56963536],
                                                [ 0.12926271],
                                                [ 0.17609881],
                                                [ 0.15365414],
                                                [-0.02246899],
                                                [ 1.44810384],
                                                [ 0.66737431],
                                                [ 1.28678634],
                                                [-0.32110607],
                                                [ 0.38285207],
                                                [-0.63913142]])


    # Local test for update_weights
    self.theta = np.array([[-6.23905153e-01,  2.61875479e-01, -5.87029005e-01,  8.76199863e-01,
                1.23255464e-01, -3.97125683e-01,  8.86089920e-01,  3.18971826e-01,
                2.64867626e-01,  1.04003845e+00],
                [ 5.73265449e-01, -1.08898467e-01,  9.37554843e-01,  3.09317802e-01,
                2.91730876e+00,  1.09868850e+00,  1.15321262e+00,  1.29099337e+00,
                7.98396113e-02,  1.31289541e+00],
                [ 2.33570276e-02, -8.31173403e-01, -5.63986458e-01,  5.27950554e-01,
                -1.56111989e+00,  2.08352921e-01, -7.28350085e-01,  7.18216383e-01,
                -7.46173711e-01,  1.87230326e+00],
                [ 7.67818129e-01, -1.26885896e+00,  1.75875935e+00, -2.27252509e-01,
                -7.27476110e-01, -1.02379932e+00,  5.67764740e-01,  1.50452187e+00,
                -5.78426973e-01, -9.97620842e-01],
                [-1.13970009e+00,  1.49640531e+00,  1.67072922e+00, -3.48471140e-01,
                5.37705087e-01, -2.90545028e-03, -6.06303023e-02,  9.64022632e-01,
                4.40956001e-01,  3.29489967e-01],
                [-2.92578935e-01,  8.15600360e-01, -2.82005902e-01,  4.99224881e-02,
                2.19477494e-01, -1.20115566e+00, -2.99094967e-01, -3.12603014e-01,
                1.01203086e-01, -1.11181809e+00],
                [-1.18655170e+00,  1.62346210e+00,  1.15644361e+00,  8.89039357e-01,
                1.82481879e+00,  4.19595238e-01, -9.10787349e-02,  4.82175748e-01,
                -1.87928699e+00, -1.09808315e+00],
                [ 7.58637063e-01,  3.26154832e-02, -1.27763634e+00,  6.58536872e-01,
                9.98901649e-01,  6.67879561e-01, -3.02967794e-02, -8.36049480e-01,
                1.07594938e-01,  4.26924329e-01],
                [-3.58588798e-01,  6.03035910e-01,  3.14431934e-01,  3.33114550e-01,
                -2.03253979e+00,  1.08109299e+00,  1.72439172e+00, -4.02467625e-01,
                -1.47689898e+00,  6.38929319e-01],
                [-4.65659732e-01, -9.67012371e-01,  1.21771626e+00, -1.38337935e+00,
                7.17428622e-01, -1.24773412e+00,  1.46226798e+00,  5.16552466e-01,
                -2.57394997e-01,  1.49369958e-01],
                [ 5.82087387e-01,  8.29894377e-01,  8.27791428e-01,  5.46730266e-01,
                -4.77381662e-01,  6.64079548e-01, -1.31132438e+00,  1.00409310e+00,
                8.73005837e-01,  1.39408104e+00],
                [-5.88779607e-01,  1.86211698e-01,  8.58286001e-01,  3.17857879e-01,
                -4.26667284e-01,  3.07331067e-01,  6.80320347e-02,  9.95703941e-01,
                -6.28462553e-01,  3.39487806e-01],
                [ 2.92931126e-01,  7.57328124e-01, -7.28922452e-02,  1.27314636e-01,
                -7.09496749e-02,  3.40658626e-02,  8.35916276e-03, -3.26744551e-01,
                2.82729979e+00, -8.43911488e-01],
                [-1.17340991e+00, -7.95626613e-01, -7.10052549e-01,  1.14365712e-02,
                1.43093280e+00,  1.68838378e+00,  2.37324364e-01, -2.49821271e+00,
                3.84359353e-01, -1.31065457e+00],
                [-5.00701575e-01, -1.14972279e+00,  4.25622580e-01, -6.26607445e-01,
                7.72119677e-01,  4.77302396e-01, -2.40069567e-01,  8.05603706e-02,
                9.17419766e-01, -3.72131916e-01]])
    self.dLoss_theta = np.array([[ 0.2590319 ,  0.01444842, -1.47958003, -0.2407005 , -0.85567139,
                  -2.04820046,  0.48388365,  1.55868825,  2.36973019,  1.56241953],
                  [-0.87080155,  1.17524499,  1.119899  , -1.98782953,  0.86128852,
                  0.62717704,  0.16280825,  0.28861672,  0.05830738,  1.63193585],
                  [-0.40178883, -0.19993939,  0.00738898,  0.27566408, -1.7632498 ,
                  1.38797381,  0.22619976,  0.5691246 ,  0.19731599, -0.18644127],
                  [-0.35524151,  0.09611414,  0.15205234,  1.15526176,  0.34605775,
                  -0.13348867,  1.98656511, -1.27942616, -1.34020918,  0.35460205],
                  [-0.21237329, -1.77459599, -0.31222966, -0.71065577,  1.1311286 ,
                  -0.62125177,  1.05061465,  0.4597817 , -0.20633091,  0.02117183],
                  [ 0.42865874, -2.30803851,  0.32706841, -0.37911961,  1.79791937,
                  -0.69126896,  1.14256392, -2.51492462,  0.81462501,  0.27610275],
                  [-0.24701649, -0.12088931, -0.26056059,  0.42300321, -0.13424856,
                  -1.78773771, -0.18581086,  2.23472174,  0.0468462 ,  0.29078795],
                  [-0.43805451,  0.17405447,  0.17794556, -0.26120192,  0.8632634 ,
                  -0.92307796, -0.13019521,  0.50505375, -0.26700418, -1.22387965],
                  [ 0.55826422, -0.98216096, -0.44730816, -0.82814759, -0.11072841,
                  -0.42938597, -0.47458987,  0.68097893,  1.7626089 , -0.35751421],
                  [ 0.52265517, -0.35541349,  0.09894225,  1.12775134,  0.05029324,
                  -0.81554735, -0.72992661, -0.61674643, -0.01330422,  0.85801145],
                  [-1.35879684, -1.03728917, -0.92454121, -1.74940542,  1.32592268,
                  -0.03637864,  1.90077932, -1.42433368,  1.29418231, -0.70164853],
                  [-0.40736969, -0.98896446, -0.94986386, -1.32374541,  0.21633296,
                  -1.31420103, -0.24180187, -0.00920154,  0.66661069,  0.10039172],
                  [ 0.32132583,  0.51441156, -0.01725087,  0.36334792, -0.97972019,
                  -0.77547043,  1.89751081, -0.01173421, -0.71050067,  1.3798799 ],
                  [-0.15684261, -0.64649103, -1.44899155,  0.77949187, -1.08630091,
                  -0.53903258,  0.64409999,  0.18363357, -0.08642687, -0.21398778],
                  [ 1.14592735,  2.23027415, -0.54876083,  0.56890409,  1.92880031,
                  1.07905705, -0.68683163, -0.43068063, -0.59796854, -0.91344341]])
    self.dLoss = {'theta': self.dLoss_theta}
    self.theta_after = np.array([[-6.24164185e-01,  2.61861031e-01, -5.85549425e-01,  8.76440563e-01,
                  1.24111135e-01, -3.95077482e-01,  8.85606037e-01,  3.17413137e-01,
                  2.62497896e-01,  1.03847604e+00],
                  [ 5.74136250e-01, -1.10073712e-01,  9.36434944e-01,  3.11305632e-01,
                  2.91644747e+00,  1.09806132e+00,  1.15304981e+00,  1.29070475e+00,
                  7.97813039e-02,  1.31126348e+00],
                  [ 2.37588164e-02, -8.30973464e-01, -5.63993847e-01,  5.27674890e-01,
                  -1.55935664e+00,  2.06964948e-01, -7.28576285e-01,  7.17647258e-01,
                  -7.46371027e-01,  1.87248970e+00],
                  [ 7.68173371e-01, -1.26895507e+00,  1.75860730e+00, -2.28407771e-01,
                  -7.27822167e-01, -1.02366583e+00,  5.65778175e-01,  1.50580129e+00,
                  -5.77086763e-01, -9.97975444e-01],
                  [-1.13948771e+00,  1.49817991e+00,  1.67104145e+00, -3.47760484e-01,
                  5.36573959e-01, -2.28419851e-03, -6.16809169e-02,  9.63562851e-01,
                  4.41162332e-01,  3.29468795e-01],
                  [-2.93007594e-01,  8.17908398e-01, -2.82332971e-01,  5.03016077e-02,
                  2.17679574e-01, -1.20046439e+00, -3.00237531e-01, -3.10088089e-01,
                  1.00388461e-01, -1.11209419e+00],
                  [-1.18630468e+00,  1.62358299e+00,  1.15670417e+00,  8.88616353e-01,
                  1.82495304e+00,  4.21382975e-01, -9.08929241e-02,  4.79941027e-01,
                  -1.87933384e+00, -1.09837393e+00],
                  [ 7.59075118e-01,  3.24414287e-02, -1.27781428e+00,  6.58798073e-01,
                  9.98038386e-01,  6.68802639e-01, -3.01665842e-02, -8.36554534e-01,
                  1.07861942e-01,  4.28148208e-01],
                  [-3.59147062e-01,  6.04018071e-01,  3.14879242e-01,  3.33942698e-01,
                  -2.03242906e+00,  1.08152238e+00,  1.72486631e+00, -4.03148603e-01,
                  -1.47866159e+00,  6.39286834e-01],
                  [-4.66182388e-01, -9.66656958e-01,  1.21761731e+00, -1.38450710e+00,
                  7.17378329e-01, -1.24691857e+00,  1.46299791e+00,  5.17169212e-01,
                  -2.57381692e-01,  1.48511947e-01],
                  [ 5.83446184e-01,  8.30931667e-01,  8.28715970e-01,  5.48479671e-01,
                  -4.78707585e-01,  6.64115926e-01, -1.31322516e+00,  1.00551744e+00,
                  8.71711655e-01,  1.39478269e+00],
                  [-5.88372237e-01,  1.87200663e-01,  8.59235865e-01,  3.19181625e-01,
                  -4.26883617e-01,  3.08645268e-01,  6.82738365e-02,  9.95713143e-01,
                  -6.29129163e-01,  3.39387414e-01],
                  [ 2.92609800e-01,  7.56813713e-01, -7.28749943e-02,  1.26951288e-01,
                  -6.99699547e-02,  3.48413330e-02,  6.46165195e-03, -3.26732817e-01,
                  2.82801029e+00, -8.45291368e-01],
                  [-1.17325306e+00, -7.94980122e-01, -7.08603557e-01,  1.06570793e-02,
                  1.43201910e+00,  1.68892282e+00,  2.36680264e-01, -2.49839635e+00,
                  3.84445780e-01, -1.31044058e+00],
                  [-5.01847503e-01, -1.15195306e+00,  4.26171340e-01, -6.27176349e-01,
                  7.70190877e-01,  4.76223339e-01, -2.39382735e-01,  8.09910512e-02,
                  9.18017735e-01, -3.71218472e-01]])


    # Outputs for GD
    self.gd_loss = [15.83627282125746, 15.501008200433239, 15.194663053917482]
    self.gd_loss_with_momentum = [15.8257298 , 15.51312132, 15.03633248]

        # Outputs for BGD
    self.batch_y =np.array([[[1., 2., 3., 4., 5., 6.]],
                   [[7., 8., 9., 1., 2., 3.]],
                   [[4., 5., 6., 7., 8., 9.]]])

    self.bgd_loss = [7.573928445695169, 17.12996447900625, 21.86319152295093]
