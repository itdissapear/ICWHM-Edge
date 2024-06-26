import re
import torch
import os

# Your log data as a multi-line string
log_data = """
2023-11-27 00:08:53.525 | INFO     | trainer_eeg2image:fit:49 - Epoch: 1/150. Train set: Average loss: 3.656558. Accuracy: 0.0383
	Validation set: Average loss: 3.639189. Accuracy: 0.0342
2023-11-27 00:09:28.248 | INFO     | trainer_eeg2image:fit:49 - Epoch: 2/150. Train set: Average loss: 3.462101. Accuracy: 0.0612
	Validation set: Average loss: 3.550124. Accuracy: 0.0651
2023-11-27 00:10:01.182 | INFO     | trainer_eeg2image:fit:49 - Epoch: 3/150. Train set: Average loss: 2.925923. Accuracy: 0.1142
	Validation set: Average loss: 3.160936. Accuracy: 0.1125
2023-11-27 00:10:32.843 | INFO     | trainer_eeg2image:fit:49 - Epoch: 4/150. Train set: Average loss: 2.441136. Accuracy: 0.1766
	Validation set: Average loss: 2.321732. Accuracy: 0.1820
2023-11-27 00:11:06.115 | INFO     | trainer_eeg2image:fit:49 - Epoch: 5/150. Train set: Average loss: 2.140141. Accuracy: 0.2273
	Validation set: Average loss: 2.042447. Accuracy: 0.2558
2023-11-27 00:11:39.780 | INFO     | trainer_eeg2image:fit:49 - Epoch: 6/150. Train set: Average loss: 2.010662. Accuracy: 0.2595
	Validation set: Average loss: 1.867203. Accuracy: 0.2909
2023-11-27 00:12:12.621 | INFO     | trainer_eeg2image:fit:49 - Epoch: 7/150. Train set: Average loss: 1.879082. Accuracy: 0.3012
	Validation set: Average loss: 1.722017. Accuracy: 0.3439
2023-11-27 00:12:45.827 | INFO     | trainer_eeg2image:fit:49 - Epoch: 8/150. Train set: Average loss: 1.747462. Accuracy: 0.3334
	Validation set: Average loss: 1.793815. Accuracy: 0.3424
2023-11-27 00:13:18.244 | INFO     | trainer_eeg2image:fit:49 - Epoch: 9/150. Train set: Average loss: 1.680132. Accuracy: 0.3564
	Validation set: Average loss: 1.682291. Accuracy: 0.3380
2023-11-27 00:13:51.278 | INFO     | trainer_eeg2image:fit:49 - Epoch: 10/150. Train set: Average loss: 1.605774. Accuracy: 0.3786
	Validation set: Average loss: 1.707724. Accuracy: 0.3649
2023-11-27 00:13:51.324 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.3649
2023-11-27 00:14:23.307 | INFO     | trainer_eeg2image:fit:49 - Epoch: 11/150. Train set: Average loss: 1.342961. Accuracy: 0.4727
	Validation set: Average loss: 1.222643. Accuracy: 0.5060
2023-11-27 00:14:56.405 | INFO     | trainer_eeg2image:fit:49 - Epoch: 12/150. Train set: Average loss: 1.214553. Accuracy: 0.5277
	Validation set: Average loss: 1.140470. Accuracy: 0.5462
2023-11-27 00:15:29.295 | INFO     | trainer_eeg2image:fit:49 - Epoch: 13/150. Train set: Average loss: 1.178816. Accuracy: 0.5382
	Validation set: Average loss: 1.104627. Accuracy: 0.5562
2023-11-27 00:16:01.923 | INFO     | trainer_eeg2image:fit:49 - Epoch: 14/150. Train set: Average loss: 1.137045. Accuracy: 0.5444
	Validation set: Average loss: 1.072718. Accuracy: 0.5749
2023-11-27 00:16:34.623 | INFO     | trainer_eeg2image:fit:49 - Epoch: 15/150. Train set: Average loss: 1.101743. Accuracy: 0.5598
	Validation set: Average loss: 1.021332. Accuracy: 0.5919
2023-11-27 00:17:07.976 | INFO     | trainer_eeg2image:fit:49 - Epoch: 16/150. Train set: Average loss: 1.076141. Accuracy: 0.5666
	Validation set: Average loss: 1.014390. Accuracy: 0.5948
2023-11-27 00:17:36.495 | INFO     | trainer_eeg2image:fit:49 - Epoch: 17/150. Train set: Average loss: 1.054055. Accuracy: 0.5775
	Validation set: Average loss: 1.027166. Accuracy: 0.5708
2023-11-27 00:18:09.707 | INFO     | trainer_eeg2image:fit:49 - Epoch: 18/150. Train set: Average loss: 1.036238. Accuracy: 0.5770
	Validation set: Average loss: 0.966499. Accuracy: 0.6044
2023-11-27 00:18:42.969 | INFO     | trainer_eeg2image:fit:49 - Epoch: 19/150. Train set: Average loss: 0.998406. Accuracy: 0.5984
	Validation set: Average loss: 0.954046. Accuracy: 0.6065
2023-11-27 00:19:16.762 | INFO     | trainer_eeg2image:fit:49 - Epoch: 20/150. Train set: Average loss: 0.997477. Accuracy: 0.6030
	Validation set: Average loss: 0.954666. Accuracy: 0.6102
2023-11-27 00:19:16.821 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6102
2023-11-27 00:19:49.760 | INFO     | trainer_eeg2image:fit:49 - Epoch: 21/150. Train set: Average loss: 0.932807. Accuracy: 0.6291
	Validation set: Average loss: 0.912061. Accuracy: 0.6331
2023-11-27 00:20:23.363 | INFO     | trainer_eeg2image:fit:49 - Epoch: 22/150. Train set: Average loss: 0.908280. Accuracy: 0.6447
	Validation set: Average loss: 0.899966. Accuracy: 0.6401
2023-11-27 00:20:56.873 | INFO     | trainer_eeg2image:fit:49 - Epoch: 23/150. Train set: Average loss: 0.916418. Accuracy: 0.6314
	Validation set: Average loss: 0.887982. Accuracy: 0.6449
2023-11-27 00:21:30.307 | INFO     | trainer_eeg2image:fit:49 - Epoch: 24/150. Train set: Average loss: 0.912743. Accuracy: 0.6383
	Validation set: Average loss: 0.898292. Accuracy: 0.6314
2023-11-27 00:22:04.012 | INFO     | trainer_eeg2image:fit:49 - Epoch: 25/150. Train set: Average loss: 0.912806. Accuracy: 0.6345
	Validation set: Average loss: 0.889286. Accuracy: 0.6471
2023-11-27 00:22:37.524 | INFO     | trainer_eeg2image:fit:49 - Epoch: 26/150. Train set: Average loss: 0.907520. Accuracy: 0.6406
	Validation set: Average loss: 0.888924. Accuracy: 0.6377
2023-11-27 00:23:11.548 | INFO     | trainer_eeg2image:fit:49 - Epoch: 27/150. Train set: Average loss: 0.898052. Accuracy: 0.6457
	Validation set: Average loss: 0.893005. Accuracy: 0.6346
2023-11-27 00:23:44.103 | INFO     | trainer_eeg2image:fit:49 - Epoch: 28/150. Train set: Average loss: 0.898886. Accuracy: 0.6368
	Validation set: Average loss: 0.882402. Accuracy: 0.6447
2023-11-27 00:24:17.775 | INFO     | trainer_eeg2image:fit:49 - Epoch: 29/150. Train set: Average loss: 0.893465. Accuracy: 0.6410
	Validation set: Average loss: 0.873104. Accuracy: 0.6580
2023-11-27 00:24:51.434 | INFO     | trainer_eeg2image:fit:49 - Epoch: 30/150. Train set: Average loss: 0.885861. Accuracy: 0.6419
	Validation set: Average loss: 0.871444. Accuracy: 0.6570
2023-11-27 00:24:51.434 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6580
2023-11-27 00:25:23.355 | INFO     | trainer_eeg2image:fit:49 - Epoch: 31/150. Train set: Average loss: 0.896470. Accuracy: 0.6466
	Validation set: Average loss: 0.876668. Accuracy: 0.6474
2023-11-27 00:25:56.495 | INFO     | trainer_eeg2image:fit:49 - Epoch: 32/150. Train set: Average loss: 0.886473. Accuracy: 0.6510
	Validation set: Average loss: 0.875849. Accuracy: 0.6459
2023-11-27 00:26:29.606 | INFO     | trainer_eeg2image:fit:49 - Epoch: 33/150. Train set: Average loss: 0.892832. Accuracy: 0.6433
	Validation set: Average loss: 0.877059. Accuracy: 0.6471
2023-11-27 00:27:02.722 | INFO     | trainer_eeg2image:fit:49 - Epoch: 34/150. Train set: Average loss: 0.884654. Accuracy: 0.6486
	Validation set: Average loss: 0.870837. Accuracy: 0.6570
2023-11-27 00:27:36.072 | INFO     | trainer_eeg2image:fit:49 - Epoch: 35/150. Train set: Average loss: 0.874411. Accuracy: 0.6582
	Validation set: Average loss: 0.880761. Accuracy: 0.6495
2023-11-27 00:28:08.979 | INFO     | trainer_eeg2image:fit:49 - Epoch: 36/150. Train set: Average loss: 0.880365. Accuracy: 0.6506
	Validation set: Average loss: 0.877297. Accuracy: 0.6434
2023-11-27 00:28:42.271 | INFO     | trainer_eeg2image:fit:49 - Epoch: 37/150. Train set: Average loss: 0.884850. Accuracy: 0.6436
	Validation set: Average loss: 0.869641. Accuracy: 0.6509
2023-11-27 00:29:15.462 | INFO     | trainer_eeg2image:fit:49 - Epoch: 38/150. Train set: Average loss: 0.882116. Accuracy: 0.6522
	Validation set: Average loss: 0.876053. Accuracy: 0.6483
2023-11-27 00:29:48.404 | INFO     | trainer_eeg2image:fit:49 - Epoch: 39/150. Train set: Average loss: 0.877982. Accuracy: 0.6484
	Validation set: Average loss: 0.872050. Accuracy: 0.6461
2023-11-27 00:30:22.119 | INFO     | trainer_eeg2image:fit:49 - Epoch: 40/150. Train set: Average loss: 0.874497. Accuracy: 0.6592
	Validation set: Average loss: 0.884392. Accuracy: 0.6375
2023-11-27 00:30:22.119 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6580
2023-11-27 00:30:54.923 | INFO     | trainer_eeg2image:fit:49 - Epoch: 41/150. Train set: Average loss: 0.879251. Accuracy: 0.6568
	Validation set: Average loss: 0.873957. Accuracy: 0.6417
2023-11-27 00:31:28.415 | INFO     | trainer_eeg2image:fit:49 - Epoch: 42/150. Train set: Average loss: 0.876233. Accuracy: 0.6583
	Validation set: Average loss: 0.869466. Accuracy: 0.6542
2023-11-27 00:32:01.263 | INFO     | trainer_eeg2image:fit:49 - Epoch: 43/150. Train set: Average loss: 0.879995. Accuracy: 0.6505
	Validation set: Average loss: 0.878070. Accuracy: 0.6482
2023-11-27 00:32:34.072 | INFO     | trainer_eeg2image:fit:49 - Epoch: 44/150. Train set: Average loss: 0.879953. Accuracy: 0.6506
	Validation set: Average loss: 0.874726. Accuracy: 0.6455
2023-11-27 00:33:07.129 | INFO     | trainer_eeg2image:fit:49 - Epoch: 45/150. Train set: Average loss: 0.881394. Accuracy: 0.6550
	Validation set: Average loss: 0.869880. Accuracy: 0.6444
2023-11-27 00:33:40.349 | INFO     | trainer_eeg2image:fit:49 - Epoch: 46/150. Train set: Average loss: 0.873036. Accuracy: 0.6537
	Validation set: Average loss: 0.879936. Accuracy: 0.6455
2023-11-27 00:34:13.298 | INFO     | trainer_eeg2image:fit:49 - Epoch: 47/150. Train set: Average loss: 0.881007. Accuracy: 0.6525
	Validation set: Average loss: 0.877368. Accuracy: 0.6490
2023-11-27 00:34:45.780 | INFO     | trainer_eeg2image:fit:49 - Epoch: 48/150. Train set: Average loss: 0.876006. Accuracy: 0.6483
	Validation set: Average loss: 0.880919. Accuracy: 0.6446
2023-11-27 00:35:18.778 | INFO     | trainer_eeg2image:fit:49 - Epoch: 49/150. Train set: Average loss: 0.875475. Accuracy: 0.6560
	Validation set: Average loss: 0.874458. Accuracy: 0.6430
2023-11-27 00:35:51.936 | INFO     | trainer_eeg2image:fit:49 - Epoch: 50/150. Train set: Average loss: 0.882741. Accuracy: 0.6482
	Validation set: Average loss: 0.872911. Accuracy: 0.6540
2023-11-27 00:35:51.937 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6580
2023-11-27 00:36:23.805 | INFO     | trainer_eeg2image:fit:49 - Epoch: 51/150. Train set: Average loss: 0.883860. Accuracy: 0.6493
	Validation set: Average loss: 0.871072. Accuracy: 0.6513
2023-11-27 00:36:57.287 | INFO     | trainer_eeg2image:fit:49 - Epoch: 52/150. Train set: Average loss: 0.880198. Accuracy: 0.6506
	Validation set: Average loss: 0.877207. Accuracy: 0.6345
2023-11-27 00:37:30.551 | INFO     | trainer_eeg2image:fit:49 - Epoch: 53/150. Train set: Average loss: 0.873032. Accuracy: 0.6529
	Validation set: Average loss: 0.872371. Accuracy: 0.6443
2023-11-27 00:38:03.763 | INFO     | trainer_eeg2image:fit:49 - Epoch: 54/150. Train set: Average loss: 0.873927. Accuracy: 0.6524
	Validation set: Average loss: 0.870044. Accuracy: 0.6424
2023-11-27 00:38:36.535 | INFO     | trainer_eeg2image:fit:49 - Epoch: 55/150. Train set: Average loss: 0.884550. Accuracy: 0.6484
	Validation set: Average loss: 0.871565. Accuracy: 0.6537
2023-11-27 00:39:09.289 | INFO     | trainer_eeg2image:fit:49 - Epoch: 56/150. Train set: Average loss: 0.871014. Accuracy: 0.6550
	Validation set: Average loss: 0.870429. Accuracy: 0.6444
2023-11-27 00:39:42.550 | INFO     | trainer_eeg2image:fit:49 - Epoch: 57/150. Train set: Average loss: 0.872446. Accuracy: 0.6615
	Validation set: Average loss: 0.874978. Accuracy: 0.6450
2023-11-27 00:40:15.666 | INFO     | trainer_eeg2image:fit:49 - Epoch: 58/150. Train set: Average loss: 0.883773. Accuracy: 0.6473
	Validation set: Average loss: 0.870392. Accuracy: 0.6468
2023-11-27 00:40:49.047 | INFO     | trainer_eeg2image:fit:49 - Epoch: 59/150. Train set: Average loss: 0.864833. Accuracy: 0.6618
	Validation set: Average loss: 0.868309. Accuracy: 0.6546
2023-11-27 00:41:20.066 | INFO     | trainer_eeg2image:fit:49 - Epoch: 60/150. Train set: Average loss: 0.884575. Accuracy: 0.6454
	Validation set: Average loss: 0.871362. Accuracy: 0.6375
2023-11-27 00:41:20.066 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6580
2023-11-27 00:41:53.175 | INFO     | trainer_eeg2image:fit:49 - Epoch: 61/150. Train set: Average loss: 0.867947. Accuracy: 0.6596
	Validation set: Average loss: 0.865983. Accuracy: 0.6545
2023-11-27 00:42:26.752 | INFO     | trainer_eeg2image:fit:49 - Epoch: 62/150. Train set: Average loss: 0.880249. Accuracy: 0.6483
	Validation set: Average loss: 0.866955. Accuracy: 0.6543
2023-11-27 00:42:59.691 | INFO     | trainer_eeg2image:fit:49 - Epoch: 63/150. Train set: Average loss: 0.875148. Accuracy: 0.6481
	Validation set: Average loss: 0.878400. Accuracy: 0.6481
2023-11-27 00:43:33.242 | INFO     | trainer_eeg2image:fit:49 - Epoch: 64/150. Train set: Average loss: 0.874565. Accuracy: 0.6564
	Validation set: Average loss: 0.865891. Accuracy: 0.6468
2023-11-27 00:44:05.732 | INFO     | trainer_eeg2image:fit:49 - Epoch: 65/150. Train set: Average loss: 0.875404. Accuracy: 0.6575
	Validation set: Average loss: 0.875158. Accuracy: 0.6398
2023-11-27 00:44:39.645 | INFO     | trainer_eeg2image:fit:49 - Epoch: 66/150. Train set: Average loss: 0.878442. Accuracy: 0.6477
	Validation set: Average loss: 0.868290. Accuracy: 0.6449
2023-11-27 00:45:12.686 | INFO     | trainer_eeg2image:fit:49 - Epoch: 67/150. Train set: Average loss: 0.879567. Accuracy: 0.6534
	Validation set: Average loss: 0.872079. Accuracy: 0.6498
2023-11-27 00:45:45.835 | INFO     | trainer_eeg2image:fit:49 - Epoch: 68/150. Train set: Average loss: 0.869817. Accuracy: 0.6569
	Validation set: Average loss: 0.874458. Accuracy: 0.6509
2023-11-27 00:46:18.833 | INFO     | trainer_eeg2image:fit:49 - Epoch: 69/150. Train set: Average loss: 0.871642. Accuracy: 0.6551
	Validation set: Average loss: 0.863882. Accuracy: 0.6498
2023-11-27 00:46:51.912 | INFO     | trainer_eeg2image:fit:49 - Epoch: 70/150. Train set: Average loss: 0.876212. Accuracy: 0.6575
	Validation set: Average loss: 0.872243. Accuracy: 0.6394
2023-11-27 00:46:51.912 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6580
2023-11-27 00:47:24.057 | INFO     | trainer_eeg2image:fit:49 - Epoch: 71/150. Train set: Average loss: 0.868144. Accuracy: 0.6582
	Validation set: Average loss: 0.866195. Accuracy: 0.6551
2023-11-27 00:47:57.585 | INFO     | trainer_eeg2image:fit:49 - Epoch: 72/150. Train set: Average loss: 0.873149. Accuracy: 0.6538
	Validation set: Average loss: 0.871391. Accuracy: 0.6355
2023-11-27 00:48:30.189 | INFO     | trainer_eeg2image:fit:49 - Epoch: 73/150. Train set: Average loss: 0.879174. Accuracy: 0.6525
	Validation set: Average loss: 0.879008. Accuracy: 0.6438
2023-11-27 00:49:03.325 | INFO     | trainer_eeg2image:fit:49 - Epoch: 74/150. Train set: Average loss: 0.875835. Accuracy: 0.6495
	Validation set: Average loss: 0.871177. Accuracy: 0.6581
2023-11-27 00:49:36.310 | INFO     | trainer_eeg2image:fit:49 - Epoch: 75/150. Train set: Average loss: 0.875544. Accuracy: 0.6554
	Validation set: Average loss: 0.875488. Accuracy: 0.6489
2023-11-27 00:50:09.739 | INFO     | trainer_eeg2image:fit:49 - Epoch: 76/150. Train set: Average loss: 0.871162. Accuracy: 0.6543
	Validation set: Average loss: 0.875020. Accuracy: 0.6482
2023-11-27 00:50:43.325 | INFO     | trainer_eeg2image:fit:49 - Epoch: 77/150. Train set: Average loss: 0.879650. Accuracy: 0.6518
	Validation set: Average loss: 0.869107. Accuracy: 0.6543
2023-11-27 00:51:16.943 | INFO     | trainer_eeg2image:fit:49 - Epoch: 78/150. Train set: Average loss: 0.885563. Accuracy: 0.6491
	Validation set: Average loss: 0.872684. Accuracy: 0.6536
2023-11-27 00:51:50.594 | INFO     | trainer_eeg2image:fit:49 - Epoch: 79/150. Train set: Average loss: 0.884265. Accuracy: 0.6518
	Validation set: Average loss: 0.867858. Accuracy: 0.6466
2023-11-27 00:52:23.279 | INFO     | trainer_eeg2image:fit:49 - Epoch: 80/150. Train set: Average loss: 0.873575. Accuracy: 0.6574
	Validation set: Average loss: 0.889523. Accuracy: 0.6263
2023-11-27 00:52:23.279 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6581
2023-11-27 00:52:54.996 | INFO     | trainer_eeg2image:fit:49 - Epoch: 81/150. Train set: Average loss: 0.878348. Accuracy: 0.6441
	Validation set: Average loss: 0.872315. Accuracy: 0.6509
2023-11-27 00:53:28.020 | INFO     | trainer_eeg2image:fit:49 - Epoch: 82/150. Train set: Average loss: 0.872797. Accuracy: 0.6489
	Validation set: Average loss: 0.871841. Accuracy: 0.6512
2023-11-27 00:54:01.568 | INFO     | trainer_eeg2image:fit:49 - Epoch: 83/150. Train set: Average loss: 0.877347. Accuracy: 0.6495
	Validation set: Average loss: 0.870995. Accuracy: 0.6510
2023-11-27 00:54:34.935 | INFO     | trainer_eeg2image:fit:49 - Epoch: 84/150. Train set: Average loss: 0.877814. Accuracy: 0.6548
	Validation set: Average loss: 0.876349. Accuracy: 0.6410
2023-11-27 00:55:07.905 | INFO     | trainer_eeg2image:fit:49 - Epoch: 85/150. Train set: Average loss: 0.871980. Accuracy: 0.6542
	Validation set: Average loss: 0.870215. Accuracy: 0.6593
2023-11-27 00:55:40.817 | INFO     | trainer_eeg2image:fit:49 - Epoch: 86/150. Train set: Average loss: 0.874032. Accuracy: 0.6522
	Validation set: Average loss: 0.871132. Accuracy: 0.6485
2023-11-27 00:56:13.854 | INFO     | trainer_eeg2image:fit:49 - Epoch: 87/150. Train set: Average loss: 0.870363. Accuracy: 0.6479
	Validation set: Average loss: 0.870120. Accuracy: 0.6493
2023-11-27 00:56:47.073 | INFO     | trainer_eeg2image:fit:49 - Epoch: 88/150. Train set: Average loss: 0.873473. Accuracy: 0.6519
	Validation set: Average loss: 0.869113. Accuracy: 0.6501
2023-11-27 00:57:20.101 | INFO     | trainer_eeg2image:fit:49 - Epoch: 89/150. Train set: Average loss: 0.876219. Accuracy: 0.6519
	Validation set: Average loss: 0.868876. Accuracy: 0.6379
2023-11-27 00:57:53.576 | INFO     | trainer_eeg2image:fit:49 - Epoch: 90/150. Train set: Average loss: 0.867966. Accuracy: 0.6611
	Validation set: Average loss: 0.873303. Accuracy: 0.6443
2023-11-27 00:57:53.577 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6593
2023-11-27 00:58:25.552 | INFO     | trainer_eeg2image:fit:49 - Epoch: 91/150. Train set: Average loss: 0.878717. Accuracy: 0.6500
	Validation set: Average loss: 0.878854. Accuracy: 0.6358
2023-11-27 00:58:58.608 | INFO     | trainer_eeg2image:fit:49 - Epoch: 92/150. Train set: Average loss: 0.887386. Accuracy: 0.6433
	Validation set: Average loss: 0.879864. Accuracy: 0.6378
2023-11-27 00:59:31.966 | INFO     | trainer_eeg2image:fit:49 - Epoch: 93/150. Train set: Average loss: 0.881315. Accuracy: 0.6542
	Validation set: Average loss: 0.873510. Accuracy: 0.6388
2023-11-27 01:00:04.910 | INFO     | trainer_eeg2image:fit:49 - Epoch: 94/150. Train set: Average loss: 0.885938. Accuracy: 0.6487
	Validation set: Average loss: 0.870547. Accuracy: 0.6537
2023-11-27 01:00:38.286 | INFO     | trainer_eeg2image:fit:49 - Epoch: 95/150. Train set: Average loss: 0.880651. Accuracy: 0.6474
	Validation set: Average loss: 0.872779. Accuracy: 0.6406
2023-11-27 01:01:11.663 | INFO     | trainer_eeg2image:fit:49 - Epoch: 96/150. Train set: Average loss: 0.875997. Accuracy: 0.6606
	Validation set: Average loss: 0.871360. Accuracy: 0.6496
2023-11-27 01:01:45.123 | INFO     | trainer_eeg2image:fit:49 - Epoch: 97/150. Train set: Average loss: 0.869075. Accuracy: 0.6600
	Validation set: Average loss: 0.869833. Accuracy: 0.6493
2023-11-27 01:02:18.168 | INFO     | trainer_eeg2image:fit:49 - Epoch: 98/150. Train set: Average loss: 0.878154. Accuracy: 0.6509
	Validation set: Average loss: 0.870293. Accuracy: 0.6554
2023-11-27 01:02:51.630 | INFO     | trainer_eeg2image:fit:49 - Epoch: 99/150. Train set: Average loss: 0.876156. Accuracy: 0.6532
	Validation set: Average loss: 0.868988. Accuracy: 0.6518
2023-11-27 01:03:25.019 | INFO     | trainer_eeg2image:fit:49 - Epoch: 100/150. Train set: Average loss: 0.874760. Accuracy: 0.6509
	Validation set: Average loss: 0.866361. Accuracy: 0.6577
2023-11-27 01:03:25.019 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6593
2023-11-27 01:03:57.400 | INFO     | trainer_eeg2image:fit:49 - Epoch: 101/150. Train set: Average loss: 0.874467. Accuracy: 0.6489
	Validation set: Average loss: 0.874324. Accuracy: 0.6501
2023-11-27 01:04:30.679 | INFO     | trainer_eeg2image:fit:49 - Epoch: 102/150. Train set: Average loss: 0.874888. Accuracy: 0.6596
	Validation set: Average loss: 0.875831. Accuracy: 0.6439
2023-11-27 01:05:03.722 | INFO     | trainer_eeg2image:fit:49 - Epoch: 103/150. Train set: Average loss: 0.870870. Accuracy: 0.6568
	Validation set: Average loss: 0.869674. Accuracy: 0.6503
2023-11-27 01:05:36.504 | INFO     | trainer_eeg2image:fit:49 - Epoch: 104/150. Train set: Average loss: 0.873319. Accuracy: 0.6538
	Validation set: Average loss: 0.875094. Accuracy: 0.6491
2023-11-27 01:06:09.386 | INFO     | trainer_eeg2image:fit:49 - Epoch: 105/150. Train set: Average loss: 0.869406. Accuracy: 0.6619
	Validation set: Average loss: 0.875666. Accuracy: 0.6381
2023-11-27 01:06:42.947 | INFO     | trainer_eeg2image:fit:49 - Epoch: 106/150. Train set: Average loss: 0.870380. Accuracy: 0.6556
	Validation set: Average loss: 0.873668. Accuracy: 0.6494
2023-11-27 01:07:16.266 | INFO     | trainer_eeg2image:fit:49 - Epoch: 107/150. Train set: Average loss: 0.885897. Accuracy: 0.6488
	Validation set: Average loss: 0.873299. Accuracy: 0.6386
2023-11-27 01:07:49.606 | INFO     | trainer_eeg2image:fit:49 - Epoch: 108/150. Train set: Average loss: 0.880847. Accuracy: 0.6538
	Validation set: Average loss: 0.865596. Accuracy: 0.6567
2023-11-27 01:08:22.496 | INFO     | trainer_eeg2image:fit:49 - Epoch: 109/150. Train set: Average loss: 0.882775. Accuracy: 0.6520
	Validation set: Average loss: 0.879822. Accuracy: 0.6545
2023-11-27 01:08:55.299 | INFO     | trainer_eeg2image:fit:49 - Epoch: 110/150. Train set: Average loss: 0.875741. Accuracy: 0.6477
	Validation set: Average loss: 0.872411. Accuracy: 0.6525
2023-11-27 01:08:55.300 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6593
2023-11-27 01:09:27.126 | INFO     | trainer_eeg2image:fit:49 - Epoch: 111/150. Train set: Average loss: 0.886335. Accuracy: 0.6516
	Validation set: Average loss: 0.875609. Accuracy: 0.6484
2023-11-27 01:09:59.916 | INFO     | trainer_eeg2image:fit:49 - Epoch: 112/150. Train set: Average loss: 0.875169. Accuracy: 0.6520
	Validation set: Average loss: 0.876640. Accuracy: 0.6448
2023-11-27 01:10:33.361 | INFO     | trainer_eeg2image:fit:49 - Epoch: 113/150. Train set: Average loss: 0.876455. Accuracy: 0.6573
	Validation set: Average loss: 0.873973. Accuracy: 0.6425
2023-11-27 01:11:06.463 | INFO     | trainer_eeg2image:fit:49 - Epoch: 114/150. Train set: Average loss: 0.870209. Accuracy: 0.6539
	Validation set: Average loss: 0.870299. Accuracy: 0.6424
2023-11-27 01:11:39.025 | INFO     | trainer_eeg2image:fit:49 - Epoch: 115/150. Train set: Average loss: 0.876298. Accuracy: 0.6551
	Validation set: Average loss: 0.872678. Accuracy: 0.6482
2023-11-27 01:12:11.747 | INFO     | trainer_eeg2image:fit:49 - Epoch: 116/150. Train set: Average loss: 0.873025. Accuracy: 0.6546
	Validation set: Average loss: 0.869647. Accuracy: 0.6514
2023-11-27 01:12:45.076 | INFO     | trainer_eeg2image:fit:49 - Epoch: 117/150. Train set: Average loss: 0.877247. Accuracy: 0.6475
	Validation set: Average loss: 0.880163. Accuracy: 0.6481
2023-11-27 01:13:17.769 | INFO     | trainer_eeg2image:fit:49 - Epoch: 118/150. Train set: Average loss: 0.873275. Accuracy: 0.6601
	Validation set: Average loss: 0.877644. Accuracy: 0.6423
2023-11-27 01:13:50.972 | INFO     | trainer_eeg2image:fit:49 - Epoch: 119/150. Train set: Average loss: 0.880315. Accuracy: 0.6496
	Validation set: Average loss: 0.871340. Accuracy: 0.6457
2023-11-27 01:14:23.468 | INFO     | trainer_eeg2image:fit:49 - Epoch: 120/150. Train set: Average loss: 0.874471. Accuracy: 0.6561
	Validation set: Average loss: 0.870644. Accuracy: 0.6459
2023-11-27 01:14:23.468 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6593
2023-11-27 01:14:55.604 | INFO     | trainer_eeg2image:fit:49 - Epoch: 121/150. Train set: Average loss: 0.875205. Accuracy: 0.6546
	Validation set: Average loss: 0.870807. Accuracy: 0.6464
2023-11-27 01:15:29.068 | INFO     | trainer_eeg2image:fit:49 - Epoch: 122/150. Train set: Average loss: 0.867890. Accuracy: 0.6583
	Validation set: Average loss: 0.867455. Accuracy: 0.6480
2023-11-27 01:16:02.210 | INFO     | trainer_eeg2image:fit:49 - Epoch: 123/150. Train set: Average loss: 0.880270. Accuracy: 0.6481
	Validation set: Average loss: 0.865640. Accuracy: 0.6535
2023-11-27 01:16:34.636 | INFO     | trainer_eeg2image:fit:49 - Epoch: 124/150. Train set: Average loss: 0.882969. Accuracy: 0.6511
	Validation set: Average loss: 0.873085. Accuracy: 0.6509
2023-11-27 01:17:07.372 | INFO     | trainer_eeg2image:fit:49 - Epoch: 125/150. Train set: Average loss: 0.878222. Accuracy: 0.6519
	Validation set: Average loss: 0.871052. Accuracy: 0.6398
2023-11-27 01:17:40.333 | INFO     | trainer_eeg2image:fit:49 - Epoch: 126/150. Train set: Average loss: 0.879801. Accuracy: 0.6519
	Validation set: Average loss: 0.871465. Accuracy: 0.6478
2023-11-27 01:18:13.453 | INFO     | trainer_eeg2image:fit:49 - Epoch: 127/150. Train set: Average loss: 0.871588. Accuracy: 0.6543
	Validation set: Average loss: 0.878967. Accuracy: 0.6363
2023-11-27 01:18:46.671 | INFO     | trainer_eeg2image:fit:49 - Epoch: 128/150. Train set: Average loss: 0.870082. Accuracy: 0.6589
	Validation set: Average loss: 0.872202. Accuracy: 0.6487
2023-11-27 01:19:19.704 | INFO     | trainer_eeg2image:fit:49 - Epoch: 129/150. Train set: Average loss: 0.887162. Accuracy: 0.6428
	Validation set: Average loss: 0.871501. Accuracy: 0.6515
2023-11-27 01:19:52.819 | INFO     | trainer_eeg2image:fit:49 - Epoch: 130/150. Train set: Average loss: 0.878468. Accuracy: 0.6510
	Validation set: Average loss: 0.874801. Accuracy: 0.6442
2023-11-27 01:19:52.820 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6593
2023-11-27 01:20:24.560 | INFO     | trainer_eeg2image:fit:49 - Epoch: 131/150. Train set: Average loss: 0.881520. Accuracy: 0.6511
	Validation set: Average loss: 0.871690. Accuracy: 0.6514
2023-11-27 01:20:57.616 | INFO     | trainer_eeg2image:fit:49 - Epoch: 132/150. Train set: Average loss: 0.873821. Accuracy: 0.6472
	Validation set: Average loss: 0.874321. Accuracy: 0.6487
2023-11-27 01:21:30.715 | INFO     | trainer_eeg2image:fit:49 - Epoch: 133/150. Train set: Average loss: 0.882018. Accuracy: 0.6507
	Validation set: Average loss: 0.870766. Accuracy: 0.6511
2023-11-27 01:22:04.426 | INFO     | trainer_eeg2image:fit:49 - Epoch: 134/150. Train set: Average loss: 0.871815. Accuracy: 0.6583
	Validation set: Average loss: 0.866957. Accuracy: 0.6580
2023-11-27 01:22:37.462 | INFO     | trainer_eeg2image:fit:49 - Epoch: 135/150. Train set: Average loss: 0.882463. Accuracy: 0.6497
	Validation set: Average loss: 0.864877. Accuracy: 0.6509
2023-11-27 01:23:09.780 | INFO     | trainer_eeg2image:fit:49 - Epoch: 136/150. Train set: Average loss: 0.878279. Accuracy: 0.6527
	Validation set: Average loss: 0.871214. Accuracy: 0.6496
2023-11-27 01:23:42.897 | INFO     | trainer_eeg2image:fit:49 - Epoch: 137/150. Train set: Average loss: 0.875371. Accuracy: 0.6546
	Validation set: Average loss: 0.871766. Accuracy: 0.6518
2023-11-27 01:24:16.583 | INFO     | trainer_eeg2image:fit:49 - Epoch: 138/150. Train set: Average loss: 0.889896. Accuracy: 0.6518
	Validation set: Average loss: 0.875517. Accuracy: 0.6413
2023-11-27 01:24:50.008 | INFO     | trainer_eeg2image:fit:49 - Epoch: 139/150. Train set: Average loss: 0.875845. Accuracy: 0.6554
	Validation set: Average loss: 0.875646. Accuracy: 0.6368
2023-11-27 01:25:22.764 | INFO     | trainer_eeg2image:fit:49 - Epoch: 140/150. Train set: Average loss: 0.875603. Accuracy: 0.6564
	Validation set: Average loss: 0.872617. Accuracy: 0.6536
2023-11-27 01:25:22.764 | INFO     | trainer_eeg2image:fit:58 - Best val accuracy: 0.6593
2023-11-27 01:25:54.558 | INFO     | trainer_eeg2image:fit:49 - Epoch: 141/150. Train set: Average loss: 0.876791. Accuracy: 0.6548
	Validation set: Average loss: 0.868449. Accuracy: 0.6435
2023-11-27 01:26:27.683 | INFO     | trainer_eeg2image:fit:49 - Epoch: 142/150. Train set: Average loss: 0.880132. Accuracy: 0.6493
	Validation set: Average loss: 0.871910. Accuracy: 0.6383
2023-11-27 01:27:00.986 | INFO     | trainer_eeg2image:fit:49 - Epoch: 143/150. Train set: Average loss: 0.873123. Accuracy: 0.6536
	Validation set: Average loss: 0.868304. Accuracy: 0.6500
2023-11-27 01:27:34.243 | INFO     | trainer_eeg2image:fit:49 - Epoch: 144/150. Train set: Average loss: 0.878135. Accuracy: 0.6491
	Validation set: Average loss: 0.880747. Accuracy: 0.6367
2023-11-27 01:28:07.156 | INFO     | trainer_eeg2image:fit:49 - Epoch: 145/150. Train set: Average loss: 0.877983. Accuracy: 0.6577
	Validation set: Average loss: 0.875347. Accuracy: 0.6392
2023-11-27 01:28:40.733 | INFO     | trainer_eeg2image:fit:49 - Epoch: 146/150. Train set: Average loss: 0.871372. Accuracy: 0.6514
	Validation set: Average loss: 0.871894. Accuracy: 0.6436
2023-11-27 01:29:13.702 | INFO     | trainer_eeg2image:fit:49 - Epoch: 147/150. Train set: Average loss: 0.887831. Accuracy: 0.6489
	Validation set: Average loss: 0.873964. Accuracy: 0.6437
2023-11-27 01:29:47.228 | INFO     | trainer_eeg2image:fit:49 - Epoch: 148/150. Train set: Average loss: 0.867014. Accuracy: 0.6528
	Validation set: Average loss: 0.867109. Accuracy: 0.6492
2023-11-27 01:30:20.262 | INFO     | trainer_eeg2image:fit:49 - Epoch: 149/150. Train set: Average loss: 0.879772. Accuracy: 0.6477
	Validation set: Average loss: 0.873881. Accuracy: 0.6448
2023-11-27 01:30:53.252 | INFO     | trainer_eeg2image:fit:49 - Epoch: 150/150. Train set: Average loss: 0.881898. Accuracy: 0.6495
	Validation set: Average loss: 0.867424. Accuracy: 0.6555
2023-11-27 01:30:53.253 | INFO     | trainer_eeg2image:fit:62 - =====================================
2023-11-27 01:30:53.253 | INFO     | trainer_eeg2image:fit:63 - Training complete in 82m 36s
2023-11-27 01:30:53.253 | INFO     | trainer_eeg2image:fit:64 - Best Val Accuracy: 0.6593
2023-11-27 01:30:53.253 | INFO     | trainer_eeg2image:fit:65 - ***********
2023-11-27 01:30:57.477 | INFO     | trainer_eeg2image:fit:67 - Test Accuracy: 0.6441
2023-11-27 01:30:57.477 | INFO     | trainer_eeg2image:fit:68 - =====================================
"""

# Regular expression patterns to match lines in the log
epoch_pattern = r"Epoch: (\d+)/(\d+)\. Train set: Average loss: ([\d\.]+)\. Accuracy: ([\d\.]+)\s+Validation set: Average loss: ([\d\.]+)\. Accuracy: ([\d\.]+)"
best_val_accuracy_pattern = r"Best Val Accuracy: ([\d\.]+)"
test_accuracy_pattern = r"Test Accuracy: ([\d\.]+)"

# Extract data
epochs_data = re.findall(epoch_pattern, log_data)
best_val_accuracy = re.search(best_val_accuracy_pattern, log_data).group(1)
test_accuracy = re.search(test_accuracy_pattern, log_data).group(1)

total_epochs = epochs_data[0][1]
print(f"Total epochs: {int(total_epochs)}")
print(f"Best val acc: {float(best_val_accuracy)}")
print(f"Test acc: {float(test_accuracy)}")

# Process data
training_data = {
    'epochs': [],
    'total_epochs': int(total_epochs),
    'best_val_accuracy': float(best_val_accuracy),
    'test_accuracy': float(test_accuracy)
}

for epoch, _, train_loss, train_acc, val_loss, val_acc in epochs_data:
    training_data['epochs'].append({
        'epoch': int(epoch),
        'train_loss': float(train_loss),
        'train_accuracy': float(train_acc),
        'validation_loss': float(val_loss),
        'validation_accuracy': float(val_acc)
    })

# Save data to a .pth file
saved_path = 'training_data_results/EEG2Image_efficientnet_v2_s_5_95_canny_50_130.pth'
torch.save(training_data, saved_path)

print(f"Data saved to {saved_path}")
