from pathlib import Path
import argparse
from DermaClassifier.utils.plots_derma import plot_data, _plot_roc, _plot_confusion
from DermaClassifier.utils.statistics import statistic_calculation


def main(result_case: str, input_path: Path, output_path: Path, numbering: str = 'alphabetical'):

    match result_case:
        case 'statistic':
            statistic_calculation(Path(__file__).resolve().parent / input_path)
        case 'auroc':
            rows = ['Dermoscopic classifier (majority votes)', 'Dermoscopic classifier (soft-labels)']
            cols = ['holdout test dataset', 'external test dataset']
            files = [[input_path / 'pred_derma_darker_majority_holdout.csv', input_path / 'pred_derma_darker_majority_extern.csv'],
                     [input_path / 'pred_derma_darker_softlabel_holdout.csv', input_path / 'pred_derma_darker_softlabel_extern.csv']]

            figure = plot_data(rows, cols, files,
                               class_labels=['Invasive Melanoma', 'Non-invasive melanoma', 'Nevus'],
                               numbering=numbering,
                               func=_plot_roc,
                               scaling=(6,4.5),
                               use_softmax=False)

            Path(output_path).mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path / 'AUROCDerma.pdf')
            figure.savefig(output_path / 'AUROCDerma.png')

        case 'confusion':
            figure = plot_data(['Dermoscopic classifier (majority votes)'],
                               ['holdout test dataset', 'external test dataset'],
                               [[input_path / 'pred_derma_darker_majority_holdout.csv', input_path / 'pred_derma_darker_majority_extern.csv']],
                               class_labels=['IM', 'NIM', 'Nevus'],
                               numbering=numbering,
                               scaling=(7,5.25), func=_plot_confusion)

            figure.savefig(output_path / 'ConfusionDermaMajority.pdf')
            figure.savefig(output_path / 'ConfusionDermaMajority.png')

            figure = plot_data(['Dermoscopic classifier (soft-labels)'],
                               ['holdout test dataset', 'external test dataset'],
                               [[input_path / 'pred_derma_darker_softlabel_holdout.csv', input_path / 'pred_derma_darker_softlabel_extern.csv']],
                               class_labels=['IM', 'NIM', 'Nevus'],
                               numbering=numbering,
                               scaling=(7, 5.25), func=_plot_confusion)

            figure.savefig(output_path / 'ConfusionDermaSoftLabel.pdf')
            figure.savefig(output_path / 'ConfusionDermaSoftLabel.png')
        case _:
            print(f'Unknown case: {result_case}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots figures and computer results from paper.")
    parser.add_argument("--result_case", required=True, help="What to do: plot {auroc, confusion} or compute metrics {statistic}")
    parser.add_argument("--input_path", default="./predictions", help="Path to directory where predictions are stored (default: ./predictions)")
    parser.add_argument("--output_path", default="./results/plots", help="Path where output will be stored (default: ./plots)")
    parser.add_argument("--numbering", default="alphabetical", help="Numbering style to use (default: alphabetical)")

    args = parser.parse_args()

    main(args.result_case, Path(args.input_path), Path(args.output_path), args.numbering)
