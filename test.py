import os
import argparse

from DermaClassifier.utils import config
from DermaClassifier.inference.test_setup import test_setup


def test(args):
    print("Start testing process.")
    test_setup(args)
    print("Finished with testing!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="../../scp_data/", help="Path to data and table information.")
    parser.add_argument("--pred", type=str, default="single", help="prediction per image or per patient {single, batch}")
    parser.add_argument("--saving", type=bool, default=False)
    parser.add_argument("--demo", type=bool, default=False)

    args = parser.parse_args()

    args.table_path = os.path.join(args.data_path, config.patho_panel_tabel)
    args.images_path = os.path.join(args.data_path, f"{config.in_imgs_size}x{config.in_imgs_size}")

    args.model = "./weights" + "/" + config.encode_label + "/" + config.preprocess

    test(args)
