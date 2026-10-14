
dataset_population_coding = {
    'SC': [('01942', '74R', '20230307', '01'),
           ('02042', '253LR', '20230725', 'full'),
           ('02043', '252R', '20230724', 'full'),
           ('02044', '243LL', '20230731', 'full'),
           ('02030', '255L', '20230720', 'full'),
           ('02031', '258LL', '20230723', 'full'),
           ('02032', '265L', '20230724', 'full'),
           ('02285', '498L', '20240619', 'full'),
            ('02286', '499R', '20240621', 'full'),
            ('02289', '483L', '20240626', 'full'),
            ('02290', '486LL', '20240719', 'full'),
            ('02291', '485LR', '20240702', 'full'),
            ('02292', '484R', '20240731', 'full'),
            ('02293', '487RR', '20240626', 'full'),
            ('02045', '107LR', '20230823', 'full'),
            ('02047', '108LL', '20230825', 'full'),
            ('02034', '83L', '20230705', 'full'),
            ('02046', '106R', '20230905', 'full'),
            ('02254', '285L', '20240416', 'full'),
            ('02255', '385LL', '20240422', 'full'),
            ('02368', '464L', '20240823', 'full'),
            ('02369', '426LR', '20240828', 'full'),
            ('02370', '425R', '20240828', 'full'),
            ('02268', '146RR', '20240621', 'full'),
            ('02341', '156LL', '20240729', 'full'),
            ('02367', '157RR', '20240823', 'full')
            ],
    'LP': [('02042', '253LR', '20230728', 'full'), 
           ('02043', '252R', '20230726', 'full'),
           ('02161', '346L', '20240214', 'full'), 
           ('02162', '348LR', '20240215', 'full'),
           ('02356', '662L', '20240909','full'),
           ('02357', '667LR', '20240909','full'), 
           ('02252', '386L', '20240502','full'),
           ('02251', '379R', '20240426', 'full'),
           ('02341', '156LL', '20240726', 'full'),
           ('02368', '464L', '20240820', 'full'),
           ('02367', '157RR', '20240820', 'full'),
           ('02369', '426LR', '20240826', 'full')]

}

def get_cluster_keys(dataset_name, as_keys=True):
    pass

dataset_extra = {
    'SC': [('02205', '115LR', '20240326'),
    ]
}
def tuplesdict_to_keys(tuples_dict):
    return {k: [{'experiment_id': rec[0], 'mouse_id': rec[1], 'session_id': rec[2], 'session_split': rec[3]} for rec in v] for k, v in tuples_dict.items()}

def get_dataset(name, as_keys=True):
    if name == 'population_coding':
        if as_keys:
            return tuplesdict_to_keys(dataset_population_coding)
        else:
            return dataset_population_coding
    else:
        raise ValueError(f'Unknown dataset {name}')