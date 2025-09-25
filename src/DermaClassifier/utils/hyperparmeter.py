import torch

encode = lambda encode_label: {"ohe": 0, "sl": 1}[encode_label]
diagnosis_label = lambda encode_label : {"ohe": "majority", "sl": "dist"}[encode_label]
do_sampler = lambda preprocess, encode : {"normal":         (True, False),
              "rgb_darker":     (True, False),
              "rgb_contrast":   (True, True),
              "rgb_gray":       (False, False)}[preprocess][encode]
sampling = lambda preprocess, encode : {"normal":         (0.7215066330753371, 0),
            "rgb_darker":     (0.4725539635634674, 0),
            "rgb_contrast":   (0.5575380493697614, 0.7835607575192484),
            "rgb_gray":       (0, 0)}[preprocess][encode]
weight_loss = lambda preprocess, encode : {"normal": (torch.Tensor([3, 3, 1]), torch.Tensor([3, 2, 1])),
               "rgb_darker":   (torch.Tensor([2, 1, 1]), torch.Tensor([1, 1, 1])),
               "rgb_contrast": (torch.Tensor([2, 2, 1]), torch.Tensor([2, 3, 1])),
               "rgb_gray":     (torch.Tensor([1, 3, 1]), torch.Tensor([1, 1, 1]))}[preprocess][encode]
lr = lambda preprocess, encode : {"normal":         (0.0001475042939967624,  0.00010270052307614925),
      "rgb_darker":     (0.00015881381759176924, 0.00010839286325346421),    
      "rgb_contrast":   (0.00033942637522135555, 0.00020750786725355494),
      "rgb_gray":       (0.00021082263552915995, 0.0008159478485883724 )}[preprocess][encode]
wd = lambda preprocess, encode : {"normal":         (8.735781611575447e-05, 3.351148907132612e-05),
      "rgb_darker":     (3.843836771786579e-05, 5.7324337117752024e-05),
      "rgb_contrast":   (6.263480709722485e-05, 5.8316570447913375e-05),
      "rgb_gray":       (2.411087217117067e-05, 3.580662725076099e-05)}[preprocess][encode]
bs = lambda preprocess, encode : {"normal":         (12, 16),
      "rgb_darker":     (17, 11),
      "rgb_contrast":   (15, 18),
      "rgb_gray":       (24, 17)}[preprocess][encode]
