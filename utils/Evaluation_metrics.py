from sklearn.metrics import roc_auc_score

def precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp/(tp + fp)


def recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp/(tp + fn)


def accuracy(tp, tn, fp, fn):
    if tp + tn + fp + fn == 0:
        return 0
    return (tp + tn)/(tp + tn + fp + fn)


    # took this from another website:s https://scikit-learn.org/stable/modules/model_evaluation.html
def balanced_accuracy(tp, tn, fp, fn):
    if tp+fn == 0 or tn + fp ==0:
        return 0
    return 0.5 * (tp/(tp+fn) + tn/(tn + fp))


def AUC_of_the_ROC(y_pred, y_target):
    return roc_auc_score(y_target, y_pred)
    # #true positive rate
    # TPR = tp/(tp + fn)
    # #false positive rate
    # FPR = fp/(fp + tn)
    # # TODO: what's next?


def f1_score(tp, fp, fn):
    r = recall(tp, fn)
    p = precision(tp, fp)
    if r + p == 0:
        return 0
    return 2 * (r*p)/(r + p)



#
# class Evaluator(object):
#     # tp = true positive.
#     # fp = false positive.
#     # tn = true negative.
#     # fn = false negative.
#     def _init_(self):
#         self.tp = 0
#         self.tn = 0
#         self.fp = 0
#         self.fn = 0
#
#