#################################################################
#
# created for ramp project, August 2022
# Author: carolyn.johnston@dev.global
#
#################################################################

import tensorflow as tf
# adding logging
import logging
import functools

log = logging.getLogger()
log.addHandler(logging.NullHandler())


def get_sparse_categorical_accuracy_fn(cfg):
    return tf.keras.metrics.SparseCategoricalAccuracy()

def get_categorical_accuracy_fn(cfg):
    return tf.keras.metrics.CategoricalAccuracy()

#---
# TODO: manually changing here for OHE experiment
def get_iou_fn(cfg):
    return tf.keras.metrics.IoU(
        num_classes= 4,
        target_class_ids= [0, 1],
        name="iou"
    # num_classes: int,
    # target_class_ids: Union[List[int], Tuple[int, ...]],
    # name=None,
    # dtype=None
)
#---

def get_onehotiou_fn(cfg):
    return tf.keras.metrics.OneHotIoU(
        num_classes = 4,
        name = "ohe_iou",
        target_class_ids= [0, 1]
    # num_classes: int,
    # target_class_ids: Union[List[int], Tuple[int, ...]],
    # name=None,
    # dtype=None
)


#---
# TODO: manually changing here for OHE experiment
def get_precision_fn(cfg):
    return tf.keras.metrics.Precision(
        # E.g. buildings
        class_id=1,
        name="precision_1"
        # thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    )


def get_recall_fn(cfg):
    return tf.keras.metrics.Recall(
        # E.g. buildings
        class_id=1,
        name="recall_1"
        # thresholds=None,top_k=None, class_id=None, name=None, dtype=None
    )
#---

def get_mse_fn(cfg):
    return tf.keras.losses.MeanSquaredError(
        # reduction=losses_utils.ReductionV2.AUTO,
        name='mean_squared_error'
    )

def get_accuracy_fn(cfg):
    return tf.keras.metrics.Accuracy()

 
# def get_sparse_iou_fn(cfg):
#     return functools.partial(tf.keras.metrics.IoU(
#         # num_classes: int,
#         # target_class_ids: Union[List[int], Tuple[int, ...]],
#         # name: Optional[str] = None,
#         # dtype: Optional[Union[str, tf.dtypes.DType]] = None,
#         # ignore_class: Optional[int] = None,
#         sparse_y_true= True,
#         sparse_y_pred= False
#         # axis: int = -1
#     ))

def get_sparse_iou_fn(cfg):
    return tf.keras.metrics.IoU(
        # num_classes: int,
        # target_class_ids: Union[List[int], Tuple[int, ...]],
        # name: Optional[str] = None,
        # dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        # ignore_class: Optional[int] = None,
        sparse_y_true= True,
        sparse_y_pred= False
        # axis: int = -1
    )

# from keras.metrics import MeanIoU
# n_classes = 2
# IOU_keras = MeanIoU(num_classes=n_classes)
# IOU_keras.update_state(y_test, y_pred)
# print("Mean IoU =", IOU_keras.result().numpy())
def get_meanIoU_fn(cfg):
    return tf.keras.metrics.MeanIoU(
        num_classes=2
    )