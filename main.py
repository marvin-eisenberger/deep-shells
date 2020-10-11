from shape_utils import *
from unsupervised_shells import *
from param import *


def load_model_matching(model_address, num_epoch=None):
    model = UnsupervisedShells()
    model.param.reset_karray()
    model.load_self(save_path(model_address), num_epoch)
    model.feat_mod.eval()

    return model


def train_faustremeshed_train():
    dataset = Faustremeshed_train()
    model = UnsupervisedShells(dataset, save_path())

    model.train()


def train_scaperemeshed_train():
    dataset = Scaperemeshed_train()
    model = UnsupervisedShells(dataset, save_path())

    model.train()


def demo_faust_scape():
    shape_x, shape_y = load_shape_pair(data_folder + "faust_scape_pair189.mat")
    model = load_model_matching("Faust_Scape", num_epoch=20)
    model.test_model(shape_x, shape_y, plot_result=True)


if __name__ == "__main__":
    demo_faust_scape()
