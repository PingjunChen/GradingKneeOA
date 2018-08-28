
import numpy as np
import matplotlib.pyplot as plt

from plot_util import plot_confusion_matrix

# Show Confusion Matrix of ResNet-34 Cross-Entropy on Mannual Cropped Knee Joints
np.set_printoptions(precision=2)
plt.figure()
cnf_matrix = np.array([[556, 43, 39, 1, 0], [131, 52, 108, 5, 0], [41, 32, 335, 37, 2],
                       [1, 1, 33, 167, 21], [0, 0, 1, 7, 43]])
class_names=["0", "1", "2", "3", "4"]
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='VGG-19-Ordinal on Manual')

# # Show Confusion Matrix of Propsed Methods on Mannual Cropped Knee Joints
# np.set_printoptions(precision=2)
# plt.figure()
# cnf_matrix = np.array([[484, 125, 29, 1, 0], [125, 104, 60, 7, 0], [66, 92, 246, 43, 0],
#                        [0, 7, 34, 169, 13], [0, 0, 0, 3, 48]])
# class_names=["0", "1", "2", "3", "4"]
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Proposed on Manual')

# # Show Confusion Matrix of ResNet-34 Cross-Entropy on Mannual Cropped Knee Joints
# np.set_printoptions(precision=2)
# plt.figure()
# cnf_matrix = np.array([[545, 44, 49, 1, 0], [178, 36, 80, 2, 0], [123, 31, 276, 17, 0],
#                        [9, 3, 45, 162, 4], [0, 0, 1, 9, 41]])
# class_names=["0", "1", "2", "3", "4"]
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='ResNet-34-CE on Manual')

# # Show Confusion Matrix of Propsed Methods on Mannual Cropped Knee Joints
# np.set_printoptions(precision=2)
# plt.figure()
# cnf_matrix = np.array([[473, 114, 48, 1, 0], [119, 114, 58, 5, 0], [78, 98, 229, 36, 1],
#                        [0, 5, 36, 163, 19], [0, 0, 0, 7, 42]])
# class_names=["0", "1", "2", "3", "4"]
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Proposed on Automatic')

# # Show Confusion Matrix of ResNet-34 Cross-Entropy on Mannual Cropped Knee Joints
# np.set_printoptions(precision=2)
# plt.figure()
# cnf_matrix = np.array([[537, 47, 52, 0, 0], [176, 42, 75, 3, 0], [131, 51, 249, 11, 0],
#                        [6, 7, 48, 154, 8], [0, 0, 0, 8, 41]])
# class_names=["0", "1", "2", "3", "4"]
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='ResNet-34-CE on Automatic')



# plt.tight_layout()
plt.show()
