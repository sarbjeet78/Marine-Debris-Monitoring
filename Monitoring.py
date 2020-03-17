import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from PIL import Image
from darkflow.net.build import TFNet
from utils.ImgSM import SplitImg, MergeImg
from utils.exif import get_GPSInfo

# list of UAV Img
MPath = 'UAVImg/'
ExportImgPath = 'result/'

ListImg = os.listdir(MPath)
ListImg = [Img for Img in ListImg if Img.endswith('.jpg')]

# Define Options for YOLO
options = {
	'model' : 'cfg/yolo_SP1.cfg',
	'load' : 125000,
	'threshold' : 0.5,
	'gpu' : 0.7,
	'backup' : 'weight/'
}

# Define Class
tfnet = TFNet(options)


# Monitoring
DataTable = np.zeros((len(ListImg), 4)) # lat, lon, n_styrofoam, n_PET

print('================================================')
print('Data Table : %d rows' % len(ListImg))
print('================================================')

count_Styrofoam = 0
count_PET = 0

RawImgIdx = 0
for Img_i, Img in enumerate(ListImg):

	# Load Img
	RawImg = cv2.imread(MPath + Img, cv2.IMREAD_COLOR)

	print('================================================')
	print('Image %d Loaded' % Img_i)
	print('')

	# print(Img)
	GPSImg = Image.open(MPath + Img)
	ImgLat, ImgLon = get_GPSInfo(GPSImg)

	print('Image GPS Information Loaded')
	print('Lat : %f   Lon : %f' % (ImgLat, ImgLon))

	DataTable[Img_i, 0] = ImgLat
	DataTable[Img_i, 1] = ImgLon

	Img_count_Styrofoam = 0
	Img_count_PET = 0

	ExportImgName = ExportImgPath + str(RawImgIdx) + '.jpg'

	# BGR to RGB
	RawImg = cv2.cvtColor(RawImg, cv2.COLOR_BGR2RGB)

	# Raw Img의 H또는 W가 608의 배수가 아닌 경우 Resize
	RawImg_H, RawImg_W, _ = RawImg.shape

	if (RawImg_H % 608 != 0) or (RawImg_W % 608 != 0):

		RawImg = cv2.resize(RawImg,
			dsize=(ceil(RawImg_H/608)*608, ceil(RawImg_W/608)*608),
			interpolation=cv2.INTER_AREA)

	# Split Img
	# Result of Function 'SplitImg' = Tensor(shape = (n of Piece, 608, 608, 3))
	sImg = SplitImg(RawImg, 608)

	# Detection
	dImg = sImg

	for idx in range(sImg.shape[0]):

		InputImg = sImg[idx, :, :, :]

		result = tfnet.return_predict(InputImg)

		# Result
		if len(result) != 0: # Detection Result가 1개라도 존재 한다면 실행

			for i in range(0, len(result)):

				tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
				br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])

				label = result[i]['label']
				confidence = result[i]['confidence']

				if label == 'Styrofoam':
					count_Styrofoam = count_Styrofoam + 1
					Img_count_Styrofoam = Img_count_Styrofoam + 1
					Img2 = cv2.rectangle(InputImg, tl, br, color=(0, 255, 0), thickness=5)

				if label == 'PET':
					count_PET = count_PET + 1
					Img_count_PET = Img_count_PET + 1
					Img2 = cv2.rectangle(InputImg, tl, br, color=(255, 0, 0), thickness=5)

				# txt = label + ' : ' + str(confidence * 100)[0:4]
				# Img2 = cv2.putText(InputImg, txt, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

			Img3 = cv2.cvtColor(Img2, cv2.COLOR_RGB2BGR)
			dImg[idx, :, :, :] = Img3

			# print('Finish : %s [count : %d] [Styrofoam : %d] [PET : %d]' % (str(idx), len(result), count_Styrofoam, count_PET))

		else:
			Img3 = cv2.cvtColor(InputImg, cv2.COLOR_RGB2BGR)
			dImg[idx, :, :, :] = Img3
			idx = idx + 1

			# print('Finish : %s [count : %d] [Styrofoam : %d] [PET : %d]' % (str(idx), 0, count_Styrofoam, count_PET))


	# Merge
	DataTable[Img_i, 2] = Img_count_Styrofoam
	DataTable[Img_i, 3] = Img_count_PET

	print('')
	print('[Styrofoam : %d]  [PET : %d]' % (Img_count_Styrofoam, Img_count_PET))
	print('')

	ResultImg = MergeImg(RawImg, dImg, 608)

	cv2.imwrite(ExportImgName, ResultImg)
	RawImgIdx = RawImgIdx + 1
	print('Finish : %d / %d  [Styrofoam : %d] [PET : %d]' % (RawImgIdx, len(ListImg), count_Styrofoam, count_PET))

	# Report
	if Img_i % 20 == 0:
		ReportName = 'Monitoring_report_' + str(Img_i) + '.csv'
		np.savetxt(ReportName, 
				DataTable, 
				delimiter=',', 
				fmt='%3.5f',
				header='lat,lon,n_styrofoam,n_PET',
				comments='')

	print('')
	print(DataTable[Img_i, :])

print('===============================================')
print('Total Styrofoam : %d , Total PET : %d' % (count_Styrofoam, count_PET))

np.savetxt('Monitoring_report.csv', 
			DataTable, 
			delimiter=',', 
			fmt='%3.5f',
			header='lat,lon,n_styrofoam,n_PET',
			comments='')
