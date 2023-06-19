'''
    # age
    '10-19' : '1',
    '20-29' : '2',
    '30-39' : '3',
    '40-49' : '4',
    '50-59' : '5',
    '60-69' : '6',
    '70-79' : '7',
    '80-89' : '8',
    '90-99' : '9',
    # menopause
    'lt40' : '1',
    'ge40' : '2',
    'premeno' : '3',
    # tumor-size
    '0-4' : '1',
    '5-9' : '2',
    '10-14' : '3',
    '15-19' : '4',
    '20-24' : '5',
    '25-29' : '6',
    '30-34' : '7',
    '35-39' : '8',
    '40-44' : '9',
    '45-49' : '10',
    '50-54' : '11',
    '55-59' : '12',
    # inv-nodes
    '0-2' : '1',
    '3-5' : '2',
    '6-8' : '3',
    '9-11' : '4',
    '12-14' : '5',
    '15-17' : '6',
    '18-20' : '7',
    '21-23' : '8',
    '24-26' : '9',
    '27-29' : '10',
    '30-32' : '11',
    '33-35' : '12',
    '36-39' : '13',
    # node-caps
    'no' : '1',
    'yes' : '2',
    # deg-malig
    '1' : '1',
    '2' : '2',
    '3' : '3',
    # breast
    'left' : '1',
    'right' : '2',
    # breast-quad
    'left_up' : '1',
    'left_low' : '2',
    'right_up' : '3',
    'right_low' : '4',
    'central' : '5',
    # class
    'no-recurrence-events' : '1',
    'recurrence-events' : '2'
'''

class Validation:
    @staticmethod
    def map_age_in_range(age):
        return age // 10
    
    @staticmethod
    def map_menopause_in_range(menopause):
        values = {
            'lt40': 1,
            'ge40': 2,
            'premeno': 3
        }
        return values[menopause]
    
    @staticmethod
    def map_tumorsize_in_range(tumor_size):
        return (tumor_size // 5) + 1
    
    @staticmethod
    def map_invnodes_in_range(inv_nodes):
        return (inv_nodes // 3) + 1
    
    @staticmethod
    def map_nodecaps_in_range(node_caps):
        return 1 if node_caps == 'no' else 2

    @staticmethod
    def map_degmalig_in_range(degree_malig):
        values = {
            'minor': 1,
            'medium': 2,
            'bigger': 3
        }
        return values[degree_malig]
    
    @staticmethod
    def map_breast_in_range(breast):
        values = {
            'l': 1,
            'r': 2
        }
        return values[breast]
    
    @staticmethod
    def map_breastquad_in_range(breast_quad):
        values = {
            'lu': 1,
            'll': 2,
            'ru': 3,
            'rl': 4,
            'c': 5
        }
        return values[breast_quad]
    
    @staticmethod
    def map_irradiat_in_range(irradiat):
        return 1 if irradiat == 'no' else 2
    
    @staticmethod
    def map_result_in_range(result):
        return 'no-recurrence-events' if result == 1 else 'recurrence-events'