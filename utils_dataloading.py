import os
import scipy as sp
import numpy as np
# from collections import defaultdict
# from utils import reindex_csr_matrix
# from utils_modelloading import load_network
folder_path = r"data" 



import pandas as pd
def load_gt_clusters():
    def load_gt_rm(path, group = True):
        # -1 is affil
        # -2 is group
        gt_rm = {}
        if group:
            ix = -2
        else:
            ix = -1
        first_line = True
        with open(path,"r") as f:
            for line in f:
                if first_line:
                    first_line = False
                    continue
                line = line.replace("\n","")
                list_line = line.split(",")
                gt_rm[(list_line[0],)] = "empty" if list_line[ix] == "" else list_line[ix]
        return gt_rm


    def load_gt_sociopatterns(path, separator = "\t",gt_index = 1):
        gt_sociopatterns = {}
        with open(path,"r") as f:
            for line in f:
                line = line.replace("\n","")
                list_line = line.split(separator)
                gt_sociopatterns[(list_line[0],)] = list_line[gt_index]
        return gt_sociopatterns
    

    def load_gt_enron(path):
        # update lod gt_clusters for enron and Bovet paths
        #ppp = os.path.join(folder_path, "enron_metadata.txt")
        roles = []
        i = 0
        with open(path, "r") as f:
            for line in f:
                line = line.replace("\n","")
                line = line.split(",")
                try:
                    role = line[2]
                    if role in ["",'N/A',]:
                        roles.append("unknown")
                    elif "President" in role:
                        roles.append("President")
                    elif "Director" in role:
                        roles.append("President")
                    else:
                        roles.append(line[2])
                    i +=1
                except IndexError:
                    if line[-1] == 'xxx':
                        roles.append("unknown")
                        i += 1
                    else:
                        assert False, "data have been cleaned, this should not happen"
#                         print(i, line)
#                         break
        return dict(zip([(str(i),) for i in range(len(roles))],roles))



    gt_primaryschool = load_gt_sociopatterns(
        os.path.join(folder_path, "primaryschool_labels.txt"), gt_index = -1
    )

    gt_highschool2011 = load_gt_sociopatterns(
        os.path.join(folder_path, "highschool2011_labels.txt"), gt_index = -1
    )

    gt_highschool2012 = load_gt_sociopatterns(
        os.path.join(folder_path, "highschool2012_labels.txt"), gt_index = -1
    )
    # two nodes, ("2",) ("478",), probably never observed
    # del gt_highschool2012[("2",)]
    # del gt_highschool2012[("478",)]

    gt_workplace2016 = load_gt_sociopatterns(
        os.path.join(folder_path, "workplace2016_labels.txt")
    )

    gt_workplace2018 = load_gt_sociopatterns(
        os.path.join(folder_path, "workplace2018_labels.txt")
    )

    gt_hospital = load_gt_sociopatterns(
        os.path.join(folder_path, "hospital_labels.txt")
    )

    # gt_rm = load_gt_rm(
    #     os.path.join(folder_path, "rm-groups.csv"),group = False
    # )

    # if filename == "temporal_clusters.ngram":
    gt_synthetic = {node: "red" if len(node[0])==1 else "blue" if node[0][0]=="1" else "green" for node in [(str(i),) for i in range(0,30)]} 

    
    # # gt_enron = load_gt_enron(
    # #     os.path.join(folder_path, "enron_metadata.txt")
    # #     )
    
    gt_sms = load_gt_sociopatterns(
        os.path.join(folder_path, "sms_genders.csv"), separator = ","
    )

    # there aren t sms recorded for all the  700+ students
    no_sms_students = {('727',), ('347',), ('734',), ('715',), ('53',), ('113',), ('463',), ('198',), ('581',), ('443',), ('762',), ('603',), ('369',), ('316',), ('769',), ('230',), ('776',), ('621',), ('384',), ('725',), ('726',), ('107',), ('579',), ('135',), ('125',), ('231',), ('345',), ('398',), ('375',), ('818',), ('745',), ('565',), ('594',), ('705',), ('343',), ('753',), ('78',), ('824',), ('770',), ('737',), ('160',), ('428',), ('487',), ('831',), ('318',), ('642',), ('691',), ('434',), ('750',), ('101',), ('456',), ('626',), ('338',), ('184',), ('662',), ('248',), ('514',), ('196',), ('808',), ('385',), ('328',), ('826',), ('117',), ('307',), ('411',), ('298',), ('309',), ('751',), ('646',), ('817',), ('602',), ('690',), ('16',), ('40',), ('518',), ('216',), ('839',), ('17',), ('206',), ('712',), ('410',), ('44',), ('226',), ('349',), ('281',), ('815',), ('674',), ('209',), ('346',), ('417',), ('138',), ('784',), ('32',), ('765',), ('228',), ('723',), ('511',), ('84',), ('630',), ('838',), ('110',), ('795',), ('467',), ('610',), ('493',), ('728',), ('814',), ('303',), ('460',), ('127',), ('555',), ('793',), ('738',), ('124',), ('718',), ('816',), ('264',), ('132',), ('576',), ('841',), ('181',), ('80',), ('834',), ('168',), ('655',), ('404',), ('238',), ('620',), ('504',), ('719',), ('142',), ('496',), ('633',), ('809',), ('490',), ('529',), ('758',), ('735',), ('313',), ('329',), ('241',), ('515',), ('756',), ('319',), ('287',), ('810',), ('425',), ('847',), ('597',), ('819',), ('429',), ('544',), ('741',), ('693',), ('277',), ('7',), ('502',), ('754',), ('156',), ('761',), ('478',), ('30',), ('573',), ('627',), ('660',), ('498',), ('497',), ('439',), ('587',), ('825',), ('686',), ('803',), ('387',), ('724',), ('407',), ('828',), ('302',), ('717',), ('729',), ('86',), ('189',), ('426',), ('807',), ('832',), ('591',), ('647',), ('790',), ('740',), ('433',), ('775',), ('45',), ('733',), ('516',), ('34',), ('223',), ('791',), ('840',), ('247',), ('823',), ('739',), ('2',), ('768',), ('400',), ('679',), ('722',), ('697',), ('730',), ('525',), ('833',), ('530',), ('757',), ('450',), ('431',), ('293',), ('517',), ('835',), ('130',), ('501',), ('261',), ('251',), ('777',), ('295',), ('792',), ('508',), ('87',), ('721',), ('820',), ('483',), ('520',), ('41',), ('103',), ('783',), ('77',), ('563',), ('767',), ('590',), ('684',), ('510',), ('447',), ('38',), ('320',), ('635',), ('822',), ('763',), ('150',), ('716',), ('353',), ('10',), ('710',), ('827',), ('629',)}
    # filtering gt of observed students only
    gt_sms = {k:v for k,v in gt_sms.items() if k not in no_sms_students}
    # gt_sms_filtered = {k:v for k,v in gt_sms_filtered.items() if k not in no_sms_students}


    gt_modules_dict = {
        "temporal_clusters.ngram":gt_synthetic,
        "temporal_clusters_3.ngram":gt_synthetic,
        # "sociopatterns_primary.ngram":gt_primaryschool,
        "highschool2011_delta4_ts900_full.ngram": gt_highschool2011,
        "highschool2012_delta4_ts900_full.ngram": gt_highschool2012,
        "workplace2016_delta4_ts900_full.ngram": gt_workplace2016,
        "workplace2018_delta4_ts900_full.ngram": gt_workplace2018,
        "hospital_delta4_ts900_full.ngram":gt_hospital,
        # "reality_mining_6.ngram":gt_rm,
        # "enron_8h.subpaths":gt_enron,
        "sms.ngram":gt_sms,
        }
    return gt_modules_dict






# using those vectors for multi-order prediction
# https://dongr0510.medium.com/multi-label-classification-example-with-multioutputclassifier-and-xgboost-in-python-98c84c7d379f
def load_target_y_tube(station_to_index = None, drop_Waterloo = True):
    """
    Only two stations in the "Waterloo & City" lines. Remove to avoid issues with classification experiments.  
    """
    folder_path = r"C:\Users\vince\Documents\Working\Datasets"
    filename = "tube_station_to_line.csv"
    file_path = os.path.join(folder_path, filename)
    # load ground truth tube stations
    station_to_line = {}
    with open(file_path) as f:
        for line in f: 
            # print(line)
            line = line.replace("\n","").split(",")
            station_name = line[0]
            # ignoring stations that are not present in the ngram
            if station_name in ['Battersea Power Station','Heathrow Terminals 2 & 3','Nine Elms']:
                continue
            station_to_line[station_name] = [station.strip() for station in line[1:]]
            #print(station_lines) 

    all_lines = [
        "Bakerloo", 
        "Central", 
        "Circle", 
        "District", 
        "Hammersmith & City", 
        "Jubilee", 
        "Metropolitan", 
        "Northern", 
        "Piccadilly", 
        "Victoria", 
        "Waterloo & City"] # only two stations on this line
    if drop_Waterloo:
        del all_lines[all_lines.index("Waterloo & City")]
    

    target_y = np.zeros((len(station_to_line), len(all_lines)))
    
    line_to_index = {}
    last_line_index = 0

    if station_to_index is None:
        station_to_index = {}
        last_station_index = 0
    else:
        last_station_index = max(station_to_index.values())
    

    # node_to_index_2
    # print(station_to_line)
    # print(station_to_index) # is it just that ndes are not tuple nodes???
    for station in station_to_line:
        # ignoring stations that are not present in the ngram
        if station in ['Battersea Power Station','Heathrow Terminals 2 & 3','Nine Elms']:
            continue
        if (station,) not in station_to_index:
            station_to_index[(station,)] = last_station_index
            last_station_index +=1
            print("missing station", station)
        for line in station_to_line[station]:
            if drop_Waterloo and (line == "Waterloo & City"):
                continue
            if line not in line_to_index:
                line_to_index[line] = last_line_index
                last_line_index += 1
            target_y[station_to_index[(station,)],line_to_index[line]] += 1
    return target_y #, station_to_index



def clean_objects_gt(gt_groups_rm, embedding_matrix, target_y, index_to_node, list_accepted):
    """
    cleaning for classification experiment. The classes that are kept are those with at least 2 observations
    """
    node_to_index = {v:k for k,v in index_to_node.items()}
    
    new_gt_groups_rm = {}
    ix_to_be_removed = []
    for id_,group in gt_groups_rm.items():
        if group in list_accepted:#['1styeargrad ','mlgrad','sloan','mlfrosh','mlstaff','grad','mlurop']:
            new_gt_groups_rm[id_] = group
        else:
            ix = node_to_index[id_]
            ix_to_be_removed.append(ix)
            # adjusting indexes after node removal


    # new_index_to_node = copy.deepcopy(index_to_node)
    # print(sorted(ix_to_be_removed, reverse=True))
    for ix in sorted(ix_to_be_removed, reverse=True):
        embedding_matrix = np.delete(
            embedding_matrix,
            obj = ix,
            axis = 0) 
        del target_y[ix]

        new_index_to_node = {}
        for ix_inner in index_to_node.keys():
            if ix_inner == ix:
                continue
            elif ix_inner < ix:
                new_index_to_node[ix_inner] = index_to_node[ix_inner]
            elif ix_inner > ix:
                new_index_to_node[ix_inner-1] = index_to_node[ix_inner]
        index_to_node = new_index_to_node    
    return new_gt_groups_rm, embedding_matrix, target_y, index_to_node

