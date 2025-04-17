# Expert-informed dermatoscopic classification for melanoma detection

This repository provides code to test, train, and tune a multi-classification networks for dermatoscopic RGB image analysis, and accompanies the paper:

Expert-informed melanoma classification with dermoscopic and histopathologic data: Results and recommendations for practice, Haggenmüller, S.; Heinlein, L.; Abels, J.; et al., Conference/Journal TBD, 2025

This Git repository provides only the model weights corresponding to the approach described in the paper, which incorporates a preprocessing step that darkens the lesion.

## System Requirements
Code was tested on Debian GNU/Linux 12 system with GPU (Nvidia V100 SXM2 15.7G) using Python 3.11.
Dependencies listed in `requirements.txt`.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.