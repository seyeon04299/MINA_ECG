from utils import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, accuracy_score, f1_score

def metric_binary (y, y_prob): 
    fpr, tpr, _ = roc_curve(y, y_prob)
    auroc = np.round(auc(fpr, tpr),4)

    prec, rec, _ = precision_recall_curve(y, y_prob,)
    auprc  = np.round(auc(rec,prec),4)
    result_text = 'auroc: {:.4f}, auprc: {:.4f}, \n\n'.format(auroc, auprc, ) 

    return auroc, auprc, result_text

def metric_multi (y, y_prob, num_classes):
    from sklearn.preprocessing import label_binarize
    y = label_binarize(y,classes=range(num_classes)) ## transform to n by num_classes matrix
    result_text = ''
    fpr, tpr, aurocs = dict(), dict(), dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:,i], y_prob[:,i])
        aurocs[i] = auc(fpr[i], tpr[i])
    auroc_macro = np.mean(list(aurocs.values()))
    micro_fpr, micro_tpr, _ = roc_curve(y.ravel(), y_prob.ravel())
    auroc_micro  = auc(micro_fpr, micro_tpr)

    result_aurocs = '\t '.join([str(np.round(x,4)) for x in (aurocs.values())])
    result_text += 'auroc_micro: {:.4f}, auroc_macro: {:.4f}, Label_auroc:{} \n'.format(auroc_micro,auroc_macro,result_aurocs) 

    prec, rec, auprcs = dict(), dict(), dict()
    for i in range(num_classes):
        prec[i], rec[i], _ = precision_recall_curve(y[:,i], y_prob[:,i])
        auprcs[i] = auc(rec[i],prec[i],)
    auprc_macro = np.mean(list(auprcs.values()))
    micro_prec, micro_rec, _ = precision_recall_curve(y.ravel(), y_prob.ravel())
    auprc_micro  = auc(micro_rec,micro_prec, )
                     
    result_auprcs = '\t '.join([str(np.round(x,4)) for x in (auprcs.values())])
    result_text += 'auprc_micro: {:.4f}, auprc_macro: {:.4f}, Label_auprc:{} \n\n'.format(auprc_micro,auprc_macro,result_auprcs)  

    return auroc_micro, auprc_micro, result_text

def metric_roc_prc (y, y_prob, f_name=None):
    result_text = ''
    auroc, auprc, fpr, tpr, prec, rec = 0, 0, 0, 0, 0, 0
    num_classes = y_prob.shape[1]
    if num_classes <=2:
        auroc, auprc, result_text = metric_binary(y, y_prob[:,-1])
    else:
        auroc, auprc, result_text = metric_multi(y,y_prob, num_classes)

    return result_text, auroc, auprc, 

def task_eval(y, y_prob, mode='train'):
    contents = ''
    y_hat = np.argmax(y_prob, axis=1)
    # print('y_prob : ', y_prob)
    # print('___')

    if mode == 'train':
        contents= 'Precision:{:.4f} | Recall:{:.4f}'.format(\
                        precision_score(y, y_hat, average='macro'), \
                        recall_score(y, y_hat, average='macro'))
        full_metrics = None

    elif mode == 'test':
        contents, auroc, auprc = metric_roc_prc(y, y_prob)  
        report_text, full_metrics = report_parser(y, y_hat)
        contents += report_text
        full_metrics['AUROC'], full_metrics['AUPRC'] = auroc, auprc

    return contents, full_metrics

def report_parser(y, y_hat):
    full_metrics = dict()
    text_full_metrics = classification_report(y, y_hat, digits=4)
    temp_text_list = [x.strip() for x in text_full_metrics.split('\n')]
    # print(temp_text_list)
    for row in temp_text_list:       
        if row.find('1.0 ')>-1:
            full_metrics['precision'] = float(row.split()[1])
            full_metrics['sensitivity'] = float(row.split()[2])
            full_metrics['f1_score'] = float(row.split()[3])
        elif row.find('0.0 ')>-1:
            full_metrics['NPV'] = float(row.split()[1]) 
            full_metrics['specificity'] = float(row.split()[2]) 
        elif row.find('accuracy')>-1:
            full_metrics['accuracy'] = float(row.split()[1])

    return text_full_metrics, full_metrics

def prob_round(y_prob, n_classes):
    labels = list(range(0,n_classes))
    labels.sort(reverse=True)
    y_hat = np.zeros(y_prob.shape, dtype=np.float32)
    for i in labels:
        bounds = (i/len(labels), (i+1)/len(labels))
        y_hat[(y_prob>bounds[0]) & (y_prob<=bounds[1])] = i
    return y_hat   

########################################
############ Calibration ###############
########################################

def calc_bins(labels, probs):
  # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(probs, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(probs[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (probs[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_calib_metrics(labels, probs):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(labels, probs)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE



import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def draw_reliability_graph(labels, probs, perform=None, fname=None):
    if fname is not None:

        ECE, MCE = get_calib_metrics(labels, probs)
        bins, _, bin_accs, _, _ = calc_bins(labels, probs)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca()

        # x/y limits
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)

        # x/y labels
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        # Create grid
        ax.set_axisbelow(True) 
        # ax.grid(color='gray', linestyle='dashed')
        
        # result
        fig.text(1.0,0.0,perform)

        # Error bars
        plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

        # Draw bars and identity line
        plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
        plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

        # Equally spaced axes
        plt.gca().set_aspect('equal', adjustable='box')

        # ECE and MCE legend
        ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
        MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
        plt.legend(handles=[ECE_patch, MCE_patch])

        #plt.show()
        plt.savefig(fname+'.png', bbox_inches='tight')