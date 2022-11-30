import torch
def gt_list_from_gt_dict(dict_start, network):
    last_ix = 0
    gt_clusters_int = {}
    str_to_int = {}
    for k,v in dict_start.items():
        if v not in str_to_int:
            str_to_int[v] = last_ix
            last_ix += 1
        gt_clusters_int[k] = str_to_int[v]

    reverse_index = network.index_to_node 
    ground_truth = torch.tensor(
        [gt_clusters_int[reverse_index[i]] for i in range(len(gt_clusters_int))]
        ).type(torch.LongTensor)
    return ground_truth



def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score
    from sklearn.metrics.cluster import adjusted_mutual_info_score
    
    return {'F1-score-weighted':f1_score(y_true, y_pred, average='weighted'),
            'F1-score-macro':f1_score(y_true, y_pred, average='macro'),
            'F1-score-micro':f1_score(y_true, y_pred, average='micro'),
            'Accuracy':accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Precision-weighted':precision_score(y_true, y_pred, average='weighted'),
            'Precision-macro':precision_score(y_true, y_pred, average='macro'),
            'Precision-micro':precision_score(y_true, y_pred, average='micro'),
            'Recall-weighted':recall_score(y_true, y_pred, average='weighted'),
            'Recall-macro':recall_score(y_true, y_pred, average='macro'),
            'Recall-micro':recall_score(y_true, y_pred, average='micro'),
            'AMI': adjusted_mutual_info_score(y_true, y_pred)}


def evaluate_learning_classification(model, data, device=None):
    # evaluating on test nodes
    model.eval()
    if device is None:
        _, pred = model(data).max(dim=1)
    else:
        _, pred = model(data,device).max(dim=1)
    metrics_train = calculate_metrics(
        data.y[data.train_mask].cpu(),
        pred[data.train_mask].cpu().numpy()

        )
    metrics_test = calculate_metrics(
        data.y[data.test_mask].cpu(),
        pred[data.test_mask].cpu().numpy()
        )
    metrics_test = {k+"_test":v for k,v in metrics_test.items()}
    metrics_train = {k+"_train":v for k,v in metrics_train.items()}
    return metrics_test, metrics_train 
