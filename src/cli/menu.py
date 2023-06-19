from cli.validation import Validation
import numpy as np
class MenuCLI:
    def __init__(self, models):
        self.__options = range(7)
        self.__models = models

    def __showHeader(self):
        print('Welcome to Menu!')
        print('Choose the model:')
        print('1 - Perceptron')
        print('2 - Adaline')
        print('3 - Decision Tree')
        print('4 - Random Forest')
        print('5 - K-Nearest Neighboor')
        print('6 - Exit')
    
    def run(self):
        while True:
            self.__showHeader()
            try:
                option = int(input('Option: '))
                if not(option in self.__options):
                    continue
                elif option == max(self.__options):
                    break
                inputs = self.__features_input(option)
                print(inputs)
                result = self.__models[option - 1].predict(inputs)
                print(f'Result for {inputs} was: {result}')
            except:
                print('Enter with the correct option!')  
    
    def __features_input(self, option) -> list:
        age = Validation.map_age_in_range(int(input('Type age: ')))
        meno_pause = Validation.map_menopause_in_range(str(input('Menopause (lt40 , ge40 ou premeno): ')))
        tumor_size = Validation.map_tumorsize_in_range(int(input('Tumor size in mmm: ')))
        inv_nodes = Validation.map_invnodes_in_range(int(input('Linfonodes number: ')))
        node_caps = Validation.map_nodecaps_in_range(str(input('Tumor in node caps(yes or no): ')))
        deg_malig = Validation.map_degmalig_in_range(str(input('Malig degree (minor, medium, bigger): ')))
        breast = Validation.map_breast_in_range(str(input('Breast cancer (l or r):')))
        breast_quad = Validation.map_breastquad_in_range(str(input('Quadrant of tumor (lu ll ru rl or c): ')))
        irradiat = Validation.map_irradiat_in_range(str(input('Radioterapy(yes or no): ')))
        inputs = []
        if option <= 2:
            inputs.append(1)
        inputs.append(age)
        inputs.append(meno_pause)
        inputs.append(tumor_size)
        inputs.append(inv_nodes)
        inputs.append(node_caps)
        inputs.append(deg_malig)
        inputs.append(breast)
        inputs.append(breast_quad)
        inputs.append(irradiat)
        if option <= 2:
            inputs = np.array(inputs)
        else:
            inputs = np.array(inputs).reshape(1,9)
        return inputs