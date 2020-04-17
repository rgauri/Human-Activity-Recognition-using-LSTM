<h1 align="center">
	Human Activity Recognition
</h1>

**_Updates done so far:_**


1) Data Preprocessing
2) Input Pipeline
3) Model architecture
4) Metrics
	- Confusion Matrix
	- ROC AUC Curve
5) Hyperparamter Optimization
	- Neurons in LSTM layer
	- Optimizer
	- Dropout rate


**_Results obtained so far:_**
<h1 align="left">
	Accuracy Test Data Set : 77.23 %
</h1>



<h1 align="left">
	1. Training Data Visualization
</h1>


![Acc_TrainData](https://media.github.tik.uni-stuttgart.de/user/986/files/f02dd600-314d-11ea-8767-b73ab9c98f07)

![Gyro_TrainData](https://media.github.tik.uni-stuttgart.de/user/986/files/eefca900-314d-11ea-9ff5-3e62a886b102)

<h1 align="left">
	2. Test Data Visualization
</h1>

![Acc_TestData](https://media.github.tik.uni-stuttgart.de/user/986/files/30153000-36ef-11ea-9bfd-72b7ea1c1812)
![Gyro_TestData](https://media.github.tik.uni-stuttgart.de/user/986/files/30153000-36ef-11ea-95cd-6c3a3997c100)

<h1 align="left">
	3. Posture Data Visualization
</h1>

<h1 align="center">
	WALKING
</h1>
 

  ![acceleration signals of experiment 1 when user 1 was performing activity_ 1(WALKING)](https://media.github.tik.uni-stuttgart.de/user/986/files/f02dd600-314d-11ea-8023-42266df19ff4)

  ![gyroscope signals of experiment 1 when user 1 was performing activity_ 1(WALKING)](https://media.github.tik.uni-stuttgart.de/user/986/files/eefca900-314d-11ea-87ce-24ec7f275818)


<h1 align="center">
WALKING_UPSTAIRS
</h1>
  
![gyroscope signals of experiment 1 when user 1 was performing activity_ 2(WALKING_UPSTAIRS)](https://media.github.tik.uni-stuttgart.de/user/986/files/eefca900-314d-11ea-8ce7-efafbc00f96a)

![acceleration signals of experiment 1 when user 1 was performing activity_ 2(WALKING_UPSTAIRS)](https://media.github.tik.uni-stuttgart.de/user/986/files/f02dd600-314d-11ea-8f52-54c696f73bbd)
  
<h1 align="center">
  WALKING_DOWNSTAIRS
</h1>

![acceleration signals of experiment 1 when user 1 was performing activity_ 3(WALKING_DOWNSTAIRS)](https://media.github.tik.uni-stuttgart.de/user/986/files/f02dd600-314d-11ea-8d18-d125155fc4ae)

![gyroscope signals of experiment 1 when user 1 was performing activity_ 3(WALKING_DOWNSTAIRS)](https://media.github.tik.uni-stuttgart.de/user/986/files/ef953f80-314d-11ea-9454-c5643735efee)

<h1 align="center">
  SITTING
</h1>

![acceleration signals of experiment 1 when user 1 was performing activity_ 4(SITTING)](https://media.github.tik.uni-stuttgart.de/user/986/files/f0c66c80-314d-11ea-8465-cb0f722fddd5)

![gyroscope signals of experiment 1 when user 1 was performing activity_ 4(SITTING)](https://media.github.tik.uni-stuttgart.de/user/986/files/ef953f80-314d-11ea-9a6e-5a2c6e3a5713)

<h1 align="center">
STANDING
</h1>

![acceleration signals of experiment 1 when user 1 was performing activity_ 5(STANDING)](https://media.github.tik.uni-stuttgart.de/user/986/files/ee641280-314d-11ea-92e3-380ef5f668fa)

![gyroscope signals of experiment 1 when user 1 was performing activity_ 5(STANDING)](https://media.github.tik.uni-stuttgart.de/user/986/files/ef953f80-314d-11ea-9387-3c6c70f31378)

<h1 align="center">
LAYING
</h1>
  
![acceleration signals of experiment 1 when user 1 was performing activity_ 6(LAYING)](https://media.github.tik.uni-stuttgart.de/user/986/files/ee641280-314d-11ea-9f27-557907e744e9)

![gyroscope signals of experiment 1 when user 1 was performing activity_ 6(LAYING)](https://media.github.tik.uni-stuttgart.de/user/986/files/ef953f80-314d-11ea-9c2f-5a9e91c77f1f)

<h1 align="center">
STAND_TO_SIT
</h1>

![acceleration signals of experiment 1 when user 1 was performing activity_ 7(STAND_TO_SIT)](https://media.github.tik.uni-stuttgart.de/user/986/files/eefca900-314d-11ea-8b12-5ccfc50f9e1d)

![gyroscope signals of experiment 1 when user 1 was performing activity_ 7(STAND_TO_SIT)](https://media.github.tik.uni-stuttgart.de/user/986/files/ef953f80-314d-11ea-859b-c70b3971db40)

<h1 align="center">
SIT_TO_STAND
</h1>

  
![acceleration signals of experiment 1 when user 1 was performing activity_ 8(SIT_TO_STAND)](https://media.github.tik.uni-stuttgart.de/user/986/files/eefca900-314d-11ea-95e4-843cc814ede1)

![gyroscope signals of experiment 1 when user 1 was performing activity_ 8(SIT_TO_STAND)](https://media.github.tik.uni-stuttgart.de/user/986/files/ef953f80-314d-11ea-8c40-7ea7a878abb6)

<h1 align="center">
 SIT_TO_LIE
</h1>

  
![acceleration signals of experiment 1 when user 1 was performing activity_ 9(SIT_TO_LIE)](https://media.github.tik.uni-stuttgart.de/user/986/files/eefca900-314d-11ea-8220-68e43d0c0c84)

![gyroscope signals of experiment 1 when user 1 was performing activity_ 9(SIT_TO_LIE)](https://media.github.tik.uni-stuttgart.de/user/986/files/ef953f80-314d-11ea-8d4e-f72f6d8c1031)

<h1 align="center">
LIE_TO_SIT
</h1>

  
![acceleration signals of experiment 1 when user 1 was performing activity_ 10(LIE_TO_SIT)](https://media.github.tik.uni-stuttgart.de/user/986/files/eefca900-314d-11ea-84c0-e69518a44c9e)
  
![gyroscope signals of experiment 1 when user 1 was performing activity_ 10(LIE_TO_SIT)](https://media.github.tik.uni-stuttgart.de/user/986/files/ef953f80-314d-11ea-8e75-ca708d8a0ec4)

<h1 align="center">
STAND_TO_LIE
</h1>

  
![acceleration signals of experiment 1 when user 1 was performing activity_ 11(STAND_TO_LIE)](https://media.github.tik.uni-stuttgart.de/user/986/files/eefca900-314d-11ea-9a48-8e7f8c817b26)
  
![gyroscope signals of experiment 1 when user 1 was performing activity_ 11(STAND_TO_LIE)](https://media.github.tik.uni-stuttgart.de/user/986/files/f02dd600-314d-11ea-9786-25bdcffbeb9d)

<h1 align="center">
 LIE_TO_STAND
</h1>

![acceleration signals of experiment 1 when user 1 was performing activity_ 12(LIE_TO_STAND)](https://media.github.tik.uni-stuttgart.de/user/986/files/eefca900-314d-11ea-8ce4-5233957daa25)

![gyroscope signals of experiment 1 when user 1 was performing activity_ 12(LIE_TO_STAND)](https://media.github.tik.uni-stuttgart.de/user/986/files/f02dd600-314d-11ea-878d-6ddd9c4f8b7a)

<h1 align="left">
	4. Histogram: Training Examples per Activity
</h1>

<p align="center">
  <img src="https://media.github.tik.uni-stuttgart.de/user/986/files/f02dd600-314d-11ea-9e7a-748d881f462b"  >
</p>

<h1 align="left">
	5. Metrics
</h1>


![Metrics_HAR](https://media.github.tik.uni-stuttgart.de/user/986/files/f02dd600-314d-11ea-86a7-c568d0760885)

<h1 align="left">
	6. Confusion Matrix
</h1>

![Confusion matrix](https://media.github.tik.uni-stuttgart.de/user/986/files/a5d05600-322e-11ea-8a48-94ff0fc160ed)

<h1 align="left">
	7. ROC-AUC Curve
</h1>

<p align="center">
  <img src="https://media.github.tik.uni-stuttgart.de/user/986/files/d25df500-36a8-11ea-9ab7-e02978b35685"  >
</p>















