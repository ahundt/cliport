import bias_eval
import numpy as np
import os

def test_bar_plot():
    ''' Test bar charts. '''
    values=np.array([1,2,3,4,5,6,7,8,9])/10
    # x_labels=[str(values[i]) for i in range(values.shape[0])]
    x_labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    single_std_errs=np.full((values.shape[0], 1), 0.5)/10
    x_pos=np.array(list(range(values.shape[0])))
    y_label="y"
    title="test_plot"

    # save_path="/Users/athundt/Downloads/checkpoints_test_cfd-67-strings-2022-01-21-pairwise/checkpoints"
    # save_path="/home/willie/github/cliport/cliport_quickstart/"
    save_path="~/Downloads/"
    save_path = os.path.expanduser(save_path)

    bias_eval.make_bar_plot(x_pos, values, single_std_errs, x_labels, y_label, title, save_path, x_axis_label='Identity')

if __name__ == '__main__':
    test_bar_plot()