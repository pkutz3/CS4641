import unittest
import numpy as np
import pickle

from NN import dlnet

class TestNN(unittest.TestCase):
    
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.setUp()

    def setUp(self):
        self.nn = dlnet(x=np.random.randn(3, 30), y=np.random.randn(1, 30))

    def assertAllClose(self, student, truth, msg=None):
        self.assertTrue(np.allclose(student, truth), msg=msg)
    
    def assertDictAllClose(self, student, truth):
        for key in truth:
            if key not in student:
                self.fail('Key ' + key + ' missing.')
            self.assertAllClose(student[key], truth[key], msg=(key + ' is incorrect.'))
        
        for key in student:
            if key not in truth:
                self.fail('Extra key ' + key + '.')
    
    def test_leaky_relu(self):
        alpha = 0.05
        u = np.array([
            [ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799, -0.97727788],
            [ 0.95008842, -0.15135721, -0.10321885,  0.4105985 ,  0.14404357,  1.45427351],
            [ 0.76103773,  0.12167502,  0.44386323,  0.33367433,  1.49407907, -0.20515826],
            [ 0.3130677 , -0.85409574, -2.55298982,  0.6536186 ,  0.8644362 , -0.74216502],
            [ 2.26975462, -1.45436567,  0.04575852, -0.18718385,  1.53277921,  1.46935877]
        ])
        student = self.nn.Leaky_Relu(alpha, u)
        truth = np.array([
            [ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799, -0.04886389],
            [ 0.95008842, -0.00756786, -0.00516094,  0.4105985 ,  0.14404357,  1.45427351],
            [ 0.76103773,  0.12167502,  0.44386323,  0.33367433,  1.49407907, -0.01025791],
            [ 0.3130677 , -0.04270479, -0.12764949,  0.6536186 ,  0.8644362 , -0.03710825],
            [ 2.26975462, -0.07271828,  0.04575852, -0.00935919,  1.53277921,  1.46935877]
        ])
        self.assertAllClose(student, truth)
        print_success_message('test_leaky_relu')

    def test_tanh(self):
        u = np.array([
            [ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799, -0.97727788],
            [ 0.95008842, -0.15135721, -0.10321885,  0.4105985 ,  0.14404357,  1.45427351],
            [ 0.76103773,  0.12167502,  0.44386323,  0.33367433,  1.49407907, -0.20515826],
            [ 0.3130677 , -0.85409574, -2.55298982,  0.6536186 ,  0.8644362 , -0.74216502],
            [ 2.26975462, -1.45436567,  0.04575852, -0.18718385,  1.53277921,  1.46935877]
        ])
        student = self.nn.Tanh(u)
        truth = np.array([
            [ 0.94295388,  0.38008347,  0.75251907,  0.97762674,  0.95337222, -0.7518851 ],
            [ 0.73982308, -0.15021189, -0.10285384,  0.38898074,  0.14305554,  0.89653467],
            [ 0.6416878 ,  0.12107809,  0.41684154,  0.32181845,  0.90407255, -0.20232755],
            [ 0.30322532, -0.69320313, -0.98795222,  0.57410094,  0.69853626, -0.63045143],
            [ 0.97886837, -0.89655275,  0.04572661, -0.18502789,  0.91089899,  0.89945506]
        ])
        self.assertAllClose(student, truth)
        print_success_message('test_tanh')

    def test_dropout(self):
        np.random.seed(0)
        u = np.array([
            [ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799, -0.97727788],
            [ 0.95008842, -0.15135721, -0.10321885,  0.4105985 ,  0.14404357,  1.45427351],
            [ 0.76103773,  0.12167502,  0.44386323,  0.33367433,  1.49407907, -0.20515826],
            [ 0.3130677 , -0.85409574, -2.55298982,  0.6536186 ,  0.8644362 , -0.74216502],
            [ 2.26975462, -1.45436567,  0.04575852, -0.18718385,  1.53277921,  1.46935877]
        ])
        student, _ = self.nn._dropout(u, prob=0.3)

        truth = np.array([
            [ 2.52007479,  0.57165316,  1.39819711,  3.201276  ,  2.66793999, -1.39611126],
            [ 1.35726917, -0.21622459, -0.1474555 ,  0.58656929,  0.20577653, 2.07753359],
            [ 1.08719676,  0.17382146,  0.        ,  0.        ,  0.        , -0.29308323],
            [ 0.44723957, -1.22013677, -3.64712831,  0.93374086,  1.23490886, -1.06023574],
            [ 0.        , -2.07766524,  0.        , -0.2674055 ,  2.18968459, 2.09908396]
        ])

        self.assertAllClose(student, truth)
        print_success_message('test_dropout')
    
    def test_loss(self):
        y = np.array([[
             1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,
            -0.97727788,  0.95008842, -0.15135721, -0.10321885,  0.4105985 ,
             0.14404357,  1.45427351,  0.76103773,  0.12167502,  0.44386323,
             0.33367433,  1.49407907, -0.20515826,  0.3130677 , -0.85409574,
            -2.55298982,  0.6536186 ,  0.8644362 , -0.74216502,  2.26975462,
            -1.45436567,  0.04575852, -0.18718385,  1.53277921,  1.46935877
        ]])
        yh = np.array([[
             0.15494743,  0.37816252, -0.88778575, -1.98079647, -0.34791215,
             0.15634897,  1.23029068,  1.20237985, -0.38732682, -0.30230275,
            -1.04855297, -1.42001794, -1.70627019,  1.9507754 , -0.50965218,
            -0.4380743 , -1.25279536,  0.77749036, -1.61389785, -0.21274028,
            -0.89546656,  0.3869025 , -0.51080514, -1.18063218, -0.02818223,
             0.42833187,  0.06651722,  0.3024719 , -0.63432209, -0.36274117
        ]])
        student = self.nn.nloss(y, yh)

        truth = 1.46445481
        self.assertAllClose(student, truth)
        print_success_message('test_loss')

    def test_forward_without_dropout(self):
        # load nn parameters
        file = open('local_tests/test_data/nn_param.pickle', 'rb')
        self.nn.nInit(param=pickle.load(file))
        file.close()

        x = np.array([
            [-1.10061918,  1.14472371,  0.90159072,  0.50249434,  0.90085595,
             -0.68372786, -0.12289023, -0.93576943, -0.26788808,  0.53035547,
             -0.69166075, -0.39675353, -0.6871727 , -0.84520564, -0.67124613,
             -0.0126646 , -1.11731035,  0.2344157 ,  1.65980218,  0.74204416,
             -0.19183555, -0.88762896, -0.74715829,  1.6924546 ,  0.05080775,
             -0.63699565,  0.19091548,  2.10025514,  0.12015895,  0.61720311],
            [ 0.30017032, -0.35224985, -1.1425182 , -0.34934272, -0.20889423,
              0.58662319,  0.83898341,  0.93110208,  0.28558733,  0.88514116,
             -0.75439794,  1.25286816,  0.51292982, -0.29809284,  0.48851815,
             -0.07557171,  1.13162939,  1.51981682,  2.18557541, -1.39649634,
             -1.44411381, -0.50446586,  0.16003707,  0.87616892,  0.31563495,
             -2.02220122, -0.30620401,  0.82797464,  0.23009474,  0.76201118],
            [-0.22232814, -0.20075807,  0.18656139,  0.41005165,  0.19829972,
              0.11900865, -0.67066229,  0.37756379,  0.12182127,  1.12948391,
              1.19891788,  0.18515642, -0.37528495, -0.63873041,  0.42349435,
              0.07734007, -0.34385368,  0.04359686, -0.62000084,  0.69803203,
             -0.44712856,  1.2245077 ,  0.40349164,  0.59357852, -1.09491185,
              0.16938243,  0.74055645, -0.9537006 , -0.26621851,  0.03261455]
        ])

        student = self.nn.forward(x, use_dropout=False)

        truth = np.array([[
            -0.02276188, -0.80744553, -0.79884405, -0.42247819, -0.67697607,
             0.02494442, -0.06870545,  0.08052316,  0.02539117, -0.01424645,
             0.29772667,  0.00696395, -0.03870995, -0.05636024,  0.10763478,
             0.005136  , -0.04076357, -0.00843497, -0.63710829, -0.71398696,
            -0.34038965,  0.30828856,  0.13069885, -0.79939701, -0.22600913,
             0.03688138, -0.07342173, -0.92209785, -0.07896259, -0.25048218
        ]])

        self.assertAllClose(student, truth)
        print_success_message('test_forward_without_dropout')

    def test_forward(self):
        # control random seed
        np.random.seed(0)

        # load nn parameters
        file = open('local_tests/test_data/nn_param.pickle', 'rb')
        self.nn.nInit(param=pickle.load(file))
        file.close()

        x = np.array([
            [-1.10061918,  1.14472371,  0.90159072,  0.50249434,  0.90085595,
             -0.68372786, -0.12289023, -0.93576943, -0.26788808,  0.53035547,
             -0.69166075, -0.39675353, -0.6871727 , -0.84520564, -0.67124613,
             -0.0126646 , -1.11731035,  0.2344157 ,  1.65980218,  0.74204416,
             -0.19183555, -0.88762896, -0.74715829,  1.6924546 ,  0.05080775,
             -0.63699565,  0.19091548,  2.10025514,  0.12015895,  0.61720311],
            [ 0.30017032, -0.35224985, -1.1425182 , -0.34934272, -0.20889423,
              0.58662319,  0.83898341,  0.93110208,  0.28558733,  0.88514116,
             -0.75439794,  1.25286816,  0.51292982, -0.29809284,  0.48851815,
             -0.07557171,  1.13162939,  1.51981682,  2.18557541, -1.39649634,
             -1.44411381, -0.50446586,  0.16003707,  0.87616892,  0.31563495,
             -2.02220122, -0.30620401,  0.82797464,  0.23009474,  0.76201118],
            [-0.22232814, -0.20075807,  0.18656139,  0.41005165,  0.19829972,
              0.11900865, -0.67066229,  0.37756379,  0.12182127,  1.12948391,
              1.19891788,  0.18515642, -0.37528495, -0.63873041,  0.42349435,
              0.07734007, -0.34385368,  0.04359686, -0.62000084,  0.69803203,
             -0.44712856,  1.2245077 ,  0.40349164,  0.59357852, -1.09491185,
              0.16938243,  0.74055645, -0.9537006 , -0.26621851,  0.03261455]
        ])

        student = self.nn.forward(x, use_dropout=True)

        truth = np.array([[
            0.05565557, -0.7097448 , -0.9159206 , -0.22745133, -0.53696253,
            0.02344555, -0.08687536,  0.06650571,  0.03626499,  0.01324989,
            0.41248723, -0.01417449, -0.05097394,  0.0577288 ,  0.12090724,
            0.02648439,  0.03612589,  0.03732259, -0.66645408, -0.41645874,
            -0.4671949 ,  0.42617936,  0.1652627 , -0.91773564,  0.0353842 ,
            0.20133724, -0.24914303, -0.87730836, -0.11256001, -0.35036333
        ]])

        self.assertAllClose(student, truth)
        print_success_message('test_forward')

    def test_compute_gradients_without_dropout(self):
        nn_param = open('local_tests/test_data/nn_param.pickle', 'rb')
        self.nn.nInit(param=pickle.load(nn_param))
        nn_param.close()

        cache = open('local_tests/test_data/test_compute_gradients_cache.pickle', 'rb')
        self.nn.ch = pickle.load(cache)
        cache.close()
        

        y = np.array([[
             0.77741921, -0.11877117, -0.19899818,  1.86647138, -0.4189379 ,
            -0.47918492, -1.95210529, -1.40232915,  0.45112294, -0.6949209 ,
             0.5154138 , -1.11487105, -0.76730983,  0.67457071,  1.46089238,
             0.5924728 ,  1.19783084,  1.70459417,  1.04008915, -0.91844004,
            -0.10534471,  0.63019567, -0.4148469 ,  0.45194604, -1.57915629,
            -0.82862798,  0.52887975, -2.23708651, -1.1077125 , -0.01771832
        ]])
        yh = np.array([[
            -1.71939447,  0.057121  , -0.79954749, -0.2915946 , -0.25898285,
             0.1892932 , -0.56378873,  0.08968641, -0.6011568 ,  0.55607351,
             1.69380911,  0.19686978,  0.16986926, -1.16400797,  0.69336623,
            -0.75806733, -0.8088472 ,  0.55743945,  0.18103874,  1.10717545,
             1.44287693, -0.53968156,  0.12837699,  1.76041518,  0.96653925,
             0.71304905,  1.30620607, -0.60460297,  0.63658341,  1.40925339
        ]])
        student = self.nn.compute_gradients(y, yh, use_dropout=False)

        dLoss_theta1 = np.array([
            [-0.01583567,  0.07980993,  0.03973422],
            [-0.00995091, -0.00129345,  0.0037702 ],
            [-0.01691969,  0.06357043, -0.0248607 ],
            [ 0.00313408,  0.00083636, -0.00102607],
            [-0.00938507, -0.03755391,  0.01916318]
        ])
        dLoss_b1 = np.array([[-0.1413308 ], [-0.00364325], [-0.05855322], [ 0.00132441], [ 0.03434403]])
        dLoss_theta2 = np.array([[0.11725097, 0.23069846, 0.05794615, 0.06345147, 0.08118983]])
        dLoss_b2 = np.array([[0.19071892]])
        truth = {'theta1': dLoss_theta1, 'b1': dLoss_b1, 'theta2': dLoss_theta2, 'b2': dLoss_b2}

        self.assertDictAllClose(student, truth)
        print_success_message('test_compute_gradients_withou_dropout')

    def test_compute_gradients(self):
        nn_param = open('local_tests/test_data/nn_param.pickle', 'rb')
        self.nn.nInit(param=pickle.load(nn_param))
        nn_param.close()

        cache = open('local_tests/test_data/test_compute_gradients_cache.pickle', 'rb')
        self.nn.ch = pickle.load(cache)
        cache.close()
        

        y = np.array([[
             0.77741921, -0.11877117, -0.19899818,  1.86647138, -0.4189379 ,
            -0.47918492, -1.95210529, -1.40232915,  0.45112294, -0.6949209 ,
             0.5154138 , -1.11487105, -0.76730983,  0.67457071,  1.46089238,
             0.5924728 ,  1.19783084,  1.70459417,  1.04008915, -0.91844004,
            -0.10534471,  0.63019567, -0.4148469 ,  0.45194604, -1.57915629,
            -0.82862798,  0.52887975, -2.23708651, -1.1077125 , -0.01771832
        ]])
        yh = np.array([[
            -1.71939447,  0.057121  , -0.79954749, -0.2915946 , -0.25898285,
             0.1892932 , -0.56378873,  0.08968641, -0.6011568 ,  0.55607351,
             1.69380911,  0.19686978,  0.16986926, -1.16400797,  0.69336623,
            -0.75806733, -0.8088472 ,  0.55743945,  0.18103874,  1.10717545,
             1.44287693, -0.53968156,  0.12837699,  1.76041518,  0.96653925,
             0.71304905,  1.30620607, -0.60460297,  0.63658341,  1.40925339
        ]])
        student = self.nn.compute_gradients(y, yh, use_dropout=True)
        
        dLoss_theta1 = np.array([
            [-0.01309931,  0.12461298,  0.00451363],
            [ 0.01219336, -0.01356142,  0.00260407],
            [-0.01859597,  0.00936985,  0.00055253],
            [ 0.00681515, -0.00066948, -0.00230718],
            [-0.02382557, -0.06597251,  0.00320531]
       ])
        dLoss_b1 = np.array([[-0.13008372], [-0.02136174], [-0.06632419], [-0.00018006], [ 0.02419189]])
        dLoss_theta2 = np.array([[0.11725097, 0.23069846, 0.05794615, 0.06345147, 0.08118983]])
        dLoss_b2 = np.array([[0.19071892]])
        truth = {'theta1': dLoss_theta1, 'b1': dLoss_b1, 'theta2': dLoss_theta2, 'b2': dLoss_b2}

        self.assertDictAllClose(student, truth)
        print_success_message('test_compute_gradients')

    def test_update_weights(self):
        nn_param = open('local_tests/test_data/nn_param.pickle', 'rb')
        self.nn.nInit(param=pickle.load(nn_param))
        nn_param.close()

        dLoss_file = open('local_tests/test_data/dLoss.pickle', 'rb')
        dLoss = pickle.load(dLoss_file)
        dLoss_file.close()
        
        self.nn.update_weights(dLoss)
        student = self.nn.param

        theta1 = np.array([
            [ 0.94882242, -0.36464497, -0.31395601],
            [-0.62450367,  0.49063477, -1.32195671],
            [ 1.00859644, -0.43012531,  0.18687619],
            [-0.14927761,  0.85106502, -1.18545526],
            [-0.17927593, -0.21328183,  0.66129455]
        ])
        b1 = np.array([[0.00012665], [0.0111731], [-0.00234416], [-0.01659802], [-0.00742044]])
        theta2 = np.array([[-0.48996797, -0.06823595, -0.38511864, 0.00195402, 0.26013481]])
        b2 = np.array([[0.00636996]])
        truth = {'theta1': theta1, 'b1': b1, 'theta2': theta2, 'b2': b2}

        self.assertDictAllClose(student, truth)
        print_success_message('test_update_weights')
    
    def test_update_weights_with_momentum(self):
        nn_param = open('local_tests/test_data/nn_param.pickle', 'rb')
        self.nn.nInit(param=pickle.load(nn_param))
        nn_param.close()

        change_file = open('local_tests/test_data/nn_change.pickle', 'rb')
        self.nn.change = pickle.load(change_file)
        change_file.close()

        dLoss_file = open('local_tests/test_data/dLoss.pickle', 'rb')
        dLoss = pickle.load(dLoss_file)
        dLoss_file.close()
        
        self.nn.update_weights(dLoss, use_momentum=True)
        student = self.nn.param

        theta1 = np.array([
            [ 0.95811233, -0.37082579, -0.32209426],
            [-0.62619372,  0.49663111, -1.32627343],
            [ 1.00950105, -0.42710571,  0.19302648],
            [-0.1520303 ,  0.84710098, -1.1823376 ],
            [-0.18187881, -0.20756012,  0.65728525]
        ])
        b1 = np.array([[-0.00010619], [0.01210595], [-0.00183543], [-0.02094245], [-0.0111725]])
        theta2 = np.array([[-0.4926153, -0.06892446, -0.38550774, -0.00113789, 0.25897234]])
        b2 = np.array([[0.0029572]])
        truth = {'theta1': theta1, 'b1': b1, 'theta2': theta2, 'b2': b2}

        self.assertDictAllClose(student, truth)
        print_success_message('test_update_weights_with_momentum')


def print_array(array):
    print(np.array2string(array, separator=', '))

def print_dict_arrays(dict_arrays):
    for key in dict_arrays:
        print(key)
        print_array(dict_arrays[key])

def print_success_message(test_name):
    print(test_name + ' passed!')